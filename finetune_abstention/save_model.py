import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NEW_MODEL_NAME = "Qwen2.5-7B-Instruct-HotpotQA-Abstention-CoT-10000-90-10"
HF_USERNAME = "fsiddiqui2" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "right" # Important for training
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --- 7. Merge & Push to Hub ---
print("Merging model and pushing to Hugging Face...")

# Clear memory for merge
torch.cuda.empty_cache()

# 1. Reload model in FP16 to merge (cannot merge 4bit directly)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 2. Load the adapter using the base_model object
# We use PeftModel directly instead of AutoPeftModel to utilize the base_model we just loaded
model = PeftModel.from_pretrained(base_model, "./qwen_abstention_finetuned/hotpotqa-abstention-90-10_COT_lr0.0002_bs4_20251209_231306/final_adapter")

# 3. Merge
model = model.merge_and_unload()

# 4. Push
repo_id = f"{HF_USERNAME}/{NEW_MODEL_NAME}"
print(f"Pushing to {repo_id}...")
model.push_to_hub(repo_id, safe_serialization=True)
tokenizer.push_to_hub(repo_id)

print("Done! Model saved to Hugging Face.")