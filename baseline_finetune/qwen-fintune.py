import torch
import sys
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel

# --- Import Check & Debugging ---
try:
    import trl
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    print(f"Successfully imported trl version: {trl.__version__}")
except ImportError as e:
    print(f"Error importing trl: {e}")
    print("Attempting to import from trl.trainer...")
    try:
        from trl.trainer import SFTTrainer
        # Fallback for some older/specific versions
        from trl.trainer import DataCollatorForCompletionOnlyLM 
        print("Import successful from trl.trainer path.")
    except ImportError:
        print("CRITICAL: Could not import SFTTrainer or DataCollator. Please run `pip install -U trl`")
        sys.exit(1)

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" 
NEW_MODEL_NAME = "Qwen2.5-7B-HotpotQA-Finetune" # Local directory name
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "fullwiki"
NUM_SAMPLES = 1000

# QLoRA / Training Config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACC_STEPS = 4
NUM_EPOCHS = 1
MAX_SEQ_LENGTH = 1024 # Reduced slightly for stability, increase to 2048 if VRAM allows

def format_data_for_instruction(sample):
    """
    Formats the HotpotQA data into the chat structure.
    """
    user_prompt = f"""You are an expert at giving concise answers. Do not give any explanations, only a short answer.
    Question: {sample['question']}
    Answer: """
    
    # Create the standard messages format
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": sample['answer']}
    ]
    
    # We return a string here because DataCollatorForCompletionOnlyLM needs text to find the split token
    # Qwen chat template will be applied in the main loop or here. 
    # To be safe with the Collator, we will format it to a string manually or let the tokenizer do it.
    return {"messages": messages}

def main():
    print(f"Loading Tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split="train").select(range(NUM_SAMPLES))
    
    # --- Formatting for Chat ---
    # We apply the chat template column so the Collator can see the raw text
    def apply_template(examples):
        outputs = []
        for q, a in zip(examples['question'], examples['answer']):
            user_prompt = f"""You are an expert at giving concise answers. Do not give any explanations, only a short answer.
            Question: {q}
            Answer: """
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": a}
            ]
            # Apply template converts it to the string: "<|im_start|>user..."
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)
            outputs.append(formatted)
        return {"text": outputs}

    dataset = dataset.map(apply_template, batched=True, remove_columns=dataset.column_names)

    # --- Load Base Model in 4-bit (QLoRA) ---
    print("Loading Base Model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else "eager"
    )

    # --- LoRA Configuration ---
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="paged_adamw_32bit",
        save_strategy="no",
        report_to="none",
    )

    # --- Data Collator ---
    # This ensures we calculate loss ONLY on the assistant's answer, not the prompt.
    # Qwen 2.5 uses <|im_start|>assistant\n to mark the start of the response.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text", # We use the pre-formatted text column
        data_collator=collator,    # USE THE COLLATOR!
        packing=False,
    )

    print("Starting training...")
    trainer.train()

    print("Training complete. Saving adapter...")
    trainer.model.save_pretrained("./final_adapter")
    tokenizer.save_pretrained("./final_adapter")

    # --- Merge and Push (Optional: Uncomment to run immediately) ---
    # Note: Merging requires reloading the model in non-4bit mode.
    # It is often safer to do this in a separate script if VRAM is tight,
    # but here is the code if you have enough RAM (CPU).
    
    print("Cleaning up memory for merge...")
    del model
    del trainer
    torch.cuda.empty_cache()

    print("Reloading base model for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="cpu", # Load on CPU to merge if VRAM is tight
    )

    print("Merging adapter...")
    model = PeftModel.from_pretrained(base_model, "./final_adapter")
    model = model.merge_and_unload()

    print("Saving full merged model locally...")
    model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)
    print(f"Model saved to {NEW_MODEL_NAME}. You can now push this folder to HF.")

if __name__ == "__main__":
    main()