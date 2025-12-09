import json
import random
import math
from datasets import Dataset

# ==========================================
# Configuration
# ==========================================
INPUT_FILE = "abstention_dataset_train.jsonl"
OUTPUT_FILE = "abstention_dataset_70_30_10k.jsonl"
HF_REPO_ID = "fsiddiqui2/hotpotqa-abstention-70-30"

MAX_TOTAL_SAMPLES = 10000
TARGET_RATIO_REASONING = 0.70
TARGET_RATIO_REFUSAL = 0.30

# ==========================================
# Main Logic
# ==========================================

def balance_dataset():
    print(f"Loading dataset from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # 1. Separate into two pools
    refusals = [d for d in data if "I don't know" in d['target']]
    corrects = [d for d in data if "I don't know" not in d['target']]

    print(f"Pool Available - Reasoning: {len(corrects)}")
    print(f"Pool Available - Refusals:  {len(refusals)}")

    # 2. Calculate Caps based on MAX limit
    # We want 70% of 10k -> 7000
    # We want 30% of 10k -> 3000
    
    target_count_reasoning = int(MAX_TOTAL_SAMPLES * TARGET_RATIO_REASONING)
    target_count_refusal = int(MAX_TOTAL_SAMPLES * TARGET_RATIO_REFUSAL)

    # 3. Handle Scarcity (If we don't have enough data to fill the caps)
    # If we have fewer than 7000 corrects, we take all of them.
    # We then adjust the refusal count to maintain the ratio relative to the ACTUAL corrects.
    
    final_count_reasoning = min(len(corrects), target_count_reasoning)
    
    # Recalculate refusal target to maintain 70/30 ratio based on actual reasoning count
    # Formula: Refusals = (Reasoning / 0.70) * 0.30
    final_count_refusal = math.ceil(final_count_reasoning * (TARGET_RATIO_REFUSAL / TARGET_RATIO_REASONING))
    
    # Ensure we don't exceed available refusals or the hard cap
    final_count_refusal = min(final_count_refusal, len(refusals), target_count_refusal)

    print("-" * 30)
    print(f"Targeting Reasoning: {final_count_reasoning}")
    print(f"Targeting Refusals:  {final_count_refusal}")
    print("-" * 30)

    # 4. Shuffle and Slice
    random.shuffle(corrects)
    random.shuffle(refusals)

    selected_corrects = corrects[:final_count_reasoning]
    selected_refusals = refusals[:final_count_refusal]

    # 5. Combine
    final_dataset_list = selected_corrects + selected_refusals
    random.shuffle(final_dataset_list)

    final_total = len(final_dataset_list)
    if final_total == 0:
        print("Error: Dataset is empty.")
        return

    final_ratio_reasoning = (len(selected_corrects) / final_total) * 100
    final_ratio_refusal = (len(selected_refusals) / final_total) * 100

    print("FINAL DATASET STATISTICS")
    print("-" * 30)
    print(f"Total Size:         {final_total}")
    print(f"Reasoning Samples:  {len(selected_corrects)} ({final_ratio_reasoning:.2f}%)")
    print(f"Refusal Samples:    {len(selected_refusals)} ({final_ratio_refusal:.2f}%)")
    print("-" * 30)

    # 6. Save Locally
    print(f"Saving locally to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_dataset_list:
            f.write(json.dumps(item) + "\n")

    # 7. Push to Hub
    print(f"Pushing to Hugging Face Hub: {HF_REPO_ID}...")
    hf_dataset = Dataset.from_list(final_dataset_list)
    hf_dataset.push_to_hub(HF_REPO_ID)
    print("Done.")

if __name__ == "__main__":
    balance_dataset()