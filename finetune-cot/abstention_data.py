import json
from datasets import Dataset

# ==========================================
# 1. Configuration
# ==========================================

INPUT_FILE = "qwen_ftcot_closedbook_train_10k_results_vllm.json"
OUTPUT_FILE = "abstention_dataset_train.jsonl"
REFUSAL_RESPONSE = "I don't know."

# ==========================================
# 2. Main Logic
# ==========================================

def create_abstention_dataset():
    print(f"Loading results from {INPUT_FILE}...")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Make sure the previous script ran successfully.")
        return

    processed_samples = []
    stats = {"kept_original": 0, "converted_to_refusal": 0}

    print("Processing samples...")
    for item in data:
        question = item['question']
        em_score = item['metrics']['em']
        
        # Get the original gold answer for reference (ignoring model performance)
        original_gold = item['gold_answers'][0] if item['gold_answers'] else ""

        # Logic for targets
        if em_score == 1.0:
            # CASE 1: Model got it RIGHT
            # target:       The full Chain-of-Thought + Answer (Good for reasoning models)
            # short_target: Just the answer (Good for standard QA models)
            target_cot = item['full_generation']
            target_short = item['parsed_answer']
            
            stats["kept_original"] += 1
        else:
            # CASE 2: Model got it WRONG
            # Both targets become the refusal phrase
            target_cot = REFUSAL_RESPONSE
            target_short = REFUSAL_RESPONSE
            
            stats["converted_to_refusal"] += 1

        if target_cot:
            processed_samples.append({
                "question": question,
                "target": target_cot,           # Full CoT + Answer
                "short_target": target_short,   # Just Answer or IDK
                "original_answer": original_gold, # Gold Reference
                "original_id": item['id'],
                "was_correct": (em_score == 1.0)
            })

    # ==========================================
    # 3. Save & Verify
    # ==========================================
    
    hf_dataset = Dataset.from_list(processed_samples)
    
    print("\n" + "="*30)
    print("DATASET STATISTICS")
    print("="*30)
    print(f"Total Samples:        {len(hf_dataset)}")
    print(f"Correct (Reinforced): {stats['kept_original']}")
    print(f"Wrong (Refusals):     {stats['converted_to_refusal']}")
    print("="*30)
    
    # Print an example
    print("\nExample (Correct):")
    correct_sample = next(x for x in processed_samples if x['was_correct'])
    print(json.dumps(correct_sample, indent=2))
    
    print("\nExample (Refusal):")
    refusal_sample = next(x for x in processed_samples if not x['was_correct'])
    print(json.dumps(refusal_sample, indent=2))

    # Save to JSONL
    print(f"\nSaving to {OUTPUT_FILE}...")
    hf_dataset.to_json(OUTPUT_FILE, orient="records", lines=True)
    print("Done.")

if __name__ == "__main__":
    create_abstention_dataset()

    from datasets import load_dataset

    dataset = load_dataset("json", data_files=f"{OUTPUT_FILE}")
    dataset.push_to_hub("fsiddiqui2/hotpotqa-abstention-10k")