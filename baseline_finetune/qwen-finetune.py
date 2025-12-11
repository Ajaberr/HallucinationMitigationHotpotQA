import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
import huggingface_hub
from transformers.trainer_utils import get_last_checkpoint

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# HF_USERNAME = "fsiddiqui2"  # Add if pushing
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "train"
NUM_SAMPLES = 50
OUTPUT_DIR="./qwen_hotpot_finetuned"

# --- 1. Load Tokenizer & Dataset ---
print(f"Loading tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "right" 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading {DATASET_NAME} dataset...")
dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT).select(range(NUM_SAMPLES))

# --- 2. Corrected Pre-Tokenization Logic ---
MAX_LEN = 512 

# MATCHING THE EVAL SCRIPT SYSTEM PROMPT EXACTLY
SYSTEM_PROMPT = "You are a concise encyclopedia. Answer the question directly with a short phrase or entity name. Do not explain."

def tokenize_batch(batch):
    """
    Tokenizes using the Chat Template to ensure Training matches Eval.
    """
    input_ids_list = []
    prompt_len_list = []
    
    # Iterate over the batch
    for question, answer in zip(batch["question"], batch["answer"]):
        # 1. Format the PROMPT (Input) using the Chat Template
        # This matches what you do in the Eval script
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        
        # Generate the prompt text with special tokens (e.g. <|im_start|>user...)
        # add_generation_prompt=True ensures it ends with <|im_start|>assistant
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 2. Format the FULL training sequence (Prompt + Answer)
        # We append the answer and the EOS token manually or via template
        full_text = prompt_text + str(answer).strip() + tokenizer.eos_token
        
        # 3. Tokenize
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        # 4. Length Handling
        if len(full_ids) > MAX_LEN:
            full_ids = full_ids[:MAX_LEN]
        
        input_ids_list.append(full_ids)
        
        # We save the length of the prompt so we can mask it in the collator
        # (We only want to calculate loss on the answer, not the system prompt/question)
        prompt_len_list.append(len(prompt_ids))

    return {
        "input_ids": input_ids_list,
        "prompt_len": prompt_len_list,
    }

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=1024,
    remove_columns=dataset.column_names,
    desc="Running tokenizer"
)

# --- 3. Collator (Unchanged, this works great) ---
class PromptMaskedCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        prompt_len = torch.tensor([f["prompt_len"] for f in features], dtype=torch.long)
        feats_to_pad = [{"input_ids": f["input_ids"]} for f in features]

        batch = self.tok.pad(
            feats_to_pad,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]
        labels = input_ids.clone()
        
        T = input_ids.size(1)
        ar = torch.arange(T, device=input_ids.device).unsqueeze(0)
        plen = prompt_len.unsqueeze(1).to(device=input_ids.device)

        labels[ar < plen] = -100
        labels[attn == 0] = -100

        batch["labels"] = labels
        return batch

collator = PromptMaskedCollator(tokenizer)

# --- 4. Load Model with QLoRA ---
print("Loading model with quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    # Qwen 2.5 supports flash_attention_2 if your GPU allows, otherwise sdpa/eager
    attn_implementation="sdpa" if torch.cuda.is_available() else "eager"
)

# Prepare LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 5. Training Setup ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,                     
    per_device_train_batch_size=4,          
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="steps",      
    save_steps=100,              
    save_total_limit=2,         
    fp16=True,                              
    optim="adamw_torch",
    report_to="none",
    group_by_length=True,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collator,
)

# --- 6. Train & Save ---
print("Checking for existing checkpoints...")
last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

if last_checkpoint:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("No checkpoint found. Starting training from scratch...")
    trainer.train()

print("Saving adapter...")
trainer.save_model("./qwen_hotpot_adapter")