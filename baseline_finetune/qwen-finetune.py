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
# 1. Login to Hugging Face (Required to push the model)
# You can also run `huggingface-cli login` in terminal beforehand
# huggingface_hub.login(token="YOUR_HF_TOKEN_HERE") 

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_USERNAME = "fsiddiqui2" 
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "fullwiki"
SPLIT = "train"
NUM_SAMPLES = 10000
OUTPUT_DIR="./qwen_hotpot_finetuned"

# Using the prompt format from your EVAL code
PROMPT_PREFIX = """You are an expert at giving concise answers. Do not give any explanations, only a short answer.
        Question: """
PROMPT_SUFFIX = """
        Answer: """

# --- 1. Load Tokenizer & Dataset ---
print(f"Loading tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "right" # Important for training
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading {DATASET_NAME} dataset...")
dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT).select(range(NUM_SAMPLES))

# --- 2. Pre-Tokenization Logic (Adapted from Reference) ---
MAX_LEN = 512 

def tokenize_batch(batch):
    """
    Tokenizes the input (Prompt) and Output (Answer) separately to allow
    masking the prompt during loss calculation.
    """
    # 1. Construct the full prompt text (System instruction + Question + "Answer:")
    prompts = [f"{PROMPT_PREFIX}{q}{PROMPT_SUFFIX}" for q in batch["question"]]
    
    # 2. Tokenize prompt (no padding yet)
    enc_prompts = tokenizer(prompts, padding=False, add_special_tokens=False)
    
    # 3. Tokenize answers (target)
    # HotpotQA answers are strings in the 'answer' column
    answers = [str(a).strip() for a in batch["answer"]]
    enc_answers = tokenizer(answers, padding=False, add_special_tokens=False)

    input_ids_list = []
    prompt_len_list = []

    for prompt_ids, answer_ids in zip(enc_prompts["input_ids"], enc_answers["input_ids"]):
        # Concatenate: Prompt + Answer + EOS
        combined_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        
        # Truncate if necessary (keep prompt, truncate answer if needed, or cut from left)
        if len(combined_ids) > MAX_LEN:
            combined_ids = combined_ids[:MAX_LEN]
            
        input_ids_list.append(combined_ids)
        # Record length of prompt so we can mask it later
        prompt_len_list.append(len(prompt_ids))

    return {
        "input_ids": input_ids_list,
        "prompt_len": prompt_len_list,
        # We don't strictly need attention_mask here as the Collator builds it
    }

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=1024,
    remove_columns=dataset.column_names,
    desc="Running tokenizer"
)

# --- 3. Collator (From Reference) ---
class PromptMaskedCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        prompt_len = torch.tensor([f["prompt_len"] for f in features], dtype=torch.long)
        feats_to_pad = [{"input_ids": f["input_ids"]} for f in features]

        # Pad the input_ids
        batch = self.tok.pad(
            feats_to_pad,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]

        # create labels (clone input_ids)
        labels = input_ids.clone()
        
        # Create a range matrix to compare against prompt_len
        T = input_ids.size(1)
        ar = torch.arange(T, device=input_ids.device).unsqueeze(0)
        plen = prompt_len.unsqueeze(1).to(device=input_ids.device)

        # MASKING LOGIC:
        # 1. Mask the prompt (set to -100)
        labels[ar < plen] = -100
        # 2. Mask the padding (set to -100)
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
    attn_implementation="sdpa" if torch.cuda.is_available() else "eager"
)

print(f"Device: {model.device}")

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
    num_train_epochs=1,                     # Kept low for speed, increase for better results
    per_device_train_batch_size=4,          # Adjust based on VRAM
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="steps",      # Save frequently (not just at end of epoch)
    save_steps=50,              # Save every 50 steps (adjust based on speed)
    save_total_limit=2,         # Only keep the last 2 checkpoints to save disk space

    fp16=True,                              # Use fp16 (or bf16 if Ampere GPU)
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

# Check if a checkpoint exists in the output directory
last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

if last_checkpoint:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("No checkpoint found. Starting training from scratch...")
    trainer.train()

print("Saving adapter...")
trainer.save_model("./qwen_hotpot_adapter")
