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
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import get_last_checkpoint

# ==========================================
# 1. Configuration
# ==========================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_ID = "fsiddiqui2/hotpotqa-abstention-70-30" # CHANGED: HF Dataset ID
OUTPUT_DIR = "./qwen_abstention_finetuned"

# --- Training Mode ---
# Set to "COT" to train on reasoning ("target" column)
# Set to "SHORT" to train on direct answers ("short_target" column)
TRAINING_MODE = "COT" 

# Delimiters (Must match your generation script)
DELIMITER_PROMPT = " ###\n"
DELIMITER_END = " END"

MAX_LEN = 2048 

# ==========================================
# 2. Load & Format Dataset
# ==========================================

print(f"Loading tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "right" 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading HF dataset: {DATASET_ID}...")
# CHANGED: Load directly from Hub
dataset = load_dataset(DATASET_ID, split="train")

def tokenize_batch(batch):
    input_ids_list = []
    prompt_len_list = []

    # Decide which column to use based on mode
    target_col = "target" if TRAINING_MODE == "COT" else "short_target"

    for q, target in zip(batch["question"], batch[target_col]):
        
        # 1. Format Prompt
        # Matches the format you used to generate the data: "Question ###\n"
        prompt_text = f"{q}{DELIMITER_PROMPT}"
        
        # 2. Format Completion
        # The 'target' column already contains "Reasoning --> Answer" or "I don't know."
        # We just need to add the END token.
        completion_text = f"{target}{DELIMITER_END}"
        
        # 3. Tokenize
        enc_prompt = tokenizer(prompt_text, padding=False, add_special_tokens=False)
        enc_completion = tokenizer(completion_text, padding=False, add_special_tokens=False)
        
        prompt_ids = enc_prompt["input_ids"]
        completion_ids = enc_completion["input_ids"]

        # 4. Concatenate & EOS
        combined_ids = prompt_ids + completion_ids + [tokenizer.eos_token_id]
        
        # 5. Truncate
        if len(combined_ids) > MAX_LEN:
            combined_ids = combined_ids[:MAX_LEN]
            
        input_ids_list.append(combined_ids)
        prompt_len_list.append(len(prompt_ids))

    return {
        "input_ids": input_ids_list,
        "prompt_len": prompt_len_list,
    }

print(f"Tokenizing dataset in {TRAINING_MODE} mode...")
tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=1024,
    remove_columns=dataset.column_names,
    desc="Running tokenizer"
)

# ==========================================
# 3. Collator (Standard Masking)
# ==========================================
# This ensures we only calculate loss on the Completion, not the Prompt.
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
        
        # Mask prompt so the model only learns to generate the answer/reasoning
        T = input_ids.size(1)
        ar = torch.arange(T, device=input_ids.device).unsqueeze(0)
        plen = prompt_len.unsqueeze(1).to(device=input_ids.device)

        labels[ar < plen] = -100 
        labels[attn == 0] = -100 

        batch["labels"] = labels
        return batch

collator = PromptMaskedCollator(tokenizer)

# ==========================================
# 4. Load Model (QLoRA)
# ==========================================
print("Loading model...")
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

# ==========================================
# 5. Training
# ==========================================
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

print("Starting training...")
trainer.train()

print("Saving adapter...")
trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
print("Done.")