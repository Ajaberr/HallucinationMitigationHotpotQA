import torch
import os
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_ID = "fsiddiqui2/hotpotqa-abstention-90-10" 
BASE_OUTPUT_DIR = "./qwen_abstention_finetuned"

# --- Hyperparameters ---
# Define variables here to easily log them in the run name
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
NUM_EPOCHS = 1
TRAINING_MODE = "COT"  # "COT" or "SHORT"

# --- Dynamic Run Name Generation ---
# 1. Extract the specific dataset name (e.g., "hotpotqa-abstention-70-30")
dataset_name = DATASET_ID.split("/")[-1] 

# 2. Create a timestamp to avoid overwriting identical runs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 3. Construct the Run Name
# Format: [DATASET]_[MODE]_lr[LR]_bs[BS]_[TIME]
run_name = f"{dataset_name}_{TRAINING_MODE}_lr{LEARNING_RATE}_bs{BATCH_SIZE}_{timestamp}"

print(f"--- LOGGING TO RUN: {run_name} ---")

# Delimiters
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
dataset = load_dataset(DATASET_ID, split="train")

def tokenize_batch(batch):
    input_ids_list = []
    prompt_len_list = []

    target_col = "target" if TRAINING_MODE == "COT" else "short_target"

    for q, target in zip(batch["question"], batch[target_col]):
        
        # Format Prompt & Completion
        prompt_text = f"{q}{DELIMITER_PROMPT}"
        completion_text = f"{target}{DELIMITER_END}"
        
        # Tokenize
        enc_prompt = tokenizer(prompt_text, padding=False, add_special_tokens=False)
        enc_completion = tokenizer(completion_text, padding=False, add_special_tokens=False)
        
        prompt_ids = enc_prompt["input_ids"]
        completion_ids = enc_completion["input_ids"]

        # Concatenate & EOS
        combined_ids = prompt_ids + completion_ids + [tokenizer.eos_token_id]
        
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
# 3. Collator
# ==========================================
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
        
        # Mask Prompt
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
# 5. Training with Dynamic Logging
# ==========================================

# Define distinct paths for checkpoints and logs based on the run_name
run_output_dir = os.path.join(BASE_OUTPUT_DIR, run_name)
run_logging_dir = os.path.join("./runs", run_name)

training_args = TrainingArguments(
    output_dir=run_output_dir,          # Checkpoints (weights) go here
    logging_dir=run_logging_dir,        # TensorBoard logs go here
    report_to="tensorboard",            # Enable TensorBoard
    run_name=run_name,                  # Name visible in some UI trackers
    
    num_train_epochs=NUM_EPOCHS,                     
    per_device_train_batch_size=BATCH_SIZE,          
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="steps",      
    save_steps=100,              
    save_total_limit=2,         
    fp16=True,                              
    optim="adamw_torch",
    group_by_length=True,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collator,
)

print(f"Starting training...\nLog Directory: {run_logging_dir}")
trainer.train()

print("Saving adapter...")
trainer.save_model(f"{run_output_dir}/final_adapter")
print("Done.")