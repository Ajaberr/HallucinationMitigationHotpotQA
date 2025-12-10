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
from transformers.trainer_utils import get_last_checkpoint

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# Your generated dataset from the previous step
DATASET_ID = "fsiddiqui2/hotpot-qa-cot-reasoning" 
OUTPUT_DIR = "./qwen_hotpot_cot_finetuned"

# --- Paper Parameters ---
# The paper mentions specific delimiters to minimize token usage (Section 3, Step 2)
# Prompt: "<q> ###"
# Completion: "<reasoning> --> <a> END"
DELIMITER_PROMPT = " ###\n"
DELIMITER_ANSWER = " --> "
DELIMITER_END = " END"

# CoT requires longer context. Appendix E.2 suggests 128 was too short. 
# We use 2048 to be safe for full traces.
MAX_LEN = 2048 

# --- 1. Load Tokenizer & Dataset ---
print(f"Loading tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "right" 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading {DATASET_ID} dataset...")
# Load the dataset (assuming it was pushed to hub successfully)
# We filter out any entries where reasoning generation might have failed (None)
dataset = load_dataset(DATASET_ID, split="train")
dataset = dataset.filter(lambda x: x['reasoning_trace'] is not None and len(x['reasoning_trace']) > 0)

print(f"Loaded {len(dataset)} samples containing CoT reasoning.")

# --- 2. Pre-Tokenization Logic (Fine-tune-CoT Style) ---

def tokenize_batch(batch):
    """
    Implements the formatting from "Language Models are Reasoning Teachers".
    Structure: Question ### Rationale --> Answer END
    """
    input_ids_list = []
    prompt_len_list = []

    for q, r, a in zip(batch["question"], batch["reasoning_trace"], batch["answer"]):
        # 1. Format the Prompt part (Question + Delimiter)
        # Paper: "<q> ###"
        prompt_text = f"{q}{DELIMITER_PROMPT}"
        
        # 2. Format the Completion part (Rationale + Delimiter + Answer + END)
        # Paper: "<rationale> --> <answer> END"
        # Note: 'r' is the teacher-generated reasoning trace.
        completion_text = f"{r}{DELIMITER_ANSWER}{a}{DELIMITER_END}"
        
        # 3. Tokenize
        # We tokenize separately to easily calculate the length of the prompt for masking
        enc_prompt = tokenizer(prompt_text, padding=False, add_special_tokens=False)
        enc_completion = tokenizer(completion_text, padding=False, add_special_tokens=False)
        
        prompt_ids = enc_prompt["input_ids"]
        completion_ids = enc_completion["input_ids"]

        # 4. Concatenate: Prompt + Completion + EOS
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

print("Tokenizing and formatting dataset...")
tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=1024,
    remove_columns=dataset.column_names,
    desc="Applying Fine-tune-CoT formatting"
)

# --- 3. Collator (Standard Masking) ---
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
        
        # Masking Logic:
        # We mask the "Question ###" part so the model learns to generate 
        # "Rationale --> Answer END" given the question.
        T = input_ids.size(1)
        ar = torch.arange(T, device=input_ids.device).unsqueeze(0)
        plen = prompt_len.unsqueeze(1).to(device=input_ids.device)

        labels[ar < plen] = -100 # Mask prompt
        labels[attn == 0] = -100 # Mask padding

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

print(f"Using device {model.device}")

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
# Parameters roughly aligned with standard fine-tuning practices for CoT
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,                     # CoT usually benefits from slightly more epochs than simple QA
    per_device_train_batch_size=4,          
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="steps",      
    save_steps=50,              
    save_total_limit=2,         
    fp16=True,                              
    optim="adamw_torch",
    report_to="none",
    group_by_length=True,       # Helps efficiency with varying CoT lengths
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
trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
print("Done.")