import os
import json
import re
import string
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

# vLLM Imports
from vllm import LLM, SamplingParams

# ==========================================
# 1. Configuration
# ==========================================

MODEL_ID = "fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-CoT-Finetuned-10000"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "train"  # CHANGED: Switched from 'validation' to 'train' per request

# Delimiters
DELIMITER_PROMPT = " ###\n"
DELIMITER_ANSWER = " --> "
DELIMITER_END = " END"

# Output Files
OUTPUT_FILE = "qwen_ftcot_closedbook_train_10k_results_vllm.json" # Updated filename
METRICS_FILE = "qwen_ftcot_closedbook_train_10k_metrics_vllm.json" # Updated filename

# ==========================================
# 2. Metric Utilities (Unchanged)
# ==========================================

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_metrics(gold_answers, pred_answers):
    em_total, f1_total = 0.0, 0.0
    for gold_list, pred in zip(gold_answers, pred_answers):
        if isinstance(gold_list, str): gold_list = [gold_list]
        best_em = max([float(exact_match_score(pred, gold)) for gold in gold_list])
        best_f1 = max([f1_score(pred, gold)[0] for gold in gold_list])
        em_total += best_em
        f1_total += best_f1
    return {
        "EM": (em_total / len(gold_answers)) * 100,
        "F1": (f1_total / len(gold_answers)) * 100,
    }

# ==========================================
# 3. Main Logic
# ==========================================

def main():
    print(f"Loading {DATASET_NAME} ({SUBSET_NAME}) split: {SPLIT}...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    
    # ---------------------------------------------------------
    # EDITED: Select first 10,000 samples
    # ---------------------------------------------------------
    print("Slicing dataset to the first 40,000 samples...")
    dataset = dataset.select(range(40000))
    # ---------------------------------------------------------
    
    print(f"Loaded {len(dataset)} samples.")

    # 1. Prepare Prompts
    # vLLM takes a list of strings directly
    prompts = [f"{item['question']}{DELIMITER_PROMPT}" for item in dataset]
    
    # Store references to gold answers for scoring later
    gold_answers_list = [
        item['answer'] if isinstance(item['answer'], list) else [item['answer']] 
        for item in dataset
    ]
    ids_list = [item['id'] for item in dataset]
    questions_list = [item['question'] for item in dataset]

    # 2. Initialize vLLM
    print(f"Initializing vLLM with model: {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        dtype="float16",             # Use float16 or bfloat16
        gpu_memory_utilization=0.90, # maximize VRAM usage for batching
        tensor_parallel_size=1,      # Use 1 GPU. Increase if multi-gpu
        max_model_len=2048,          # Ensure context fits
    )

    # 3. Set Sampling Parameters
    # We add the DELIMITER_END to 'stop' so the model stops generating immediately when done.
    sampling_params = SamplingParams(
        temperature=0,      # Greedy decoding
        max_tokens=512,     # Max CoT length
        stop=[DELIMITER_END, "<|im_end|>", "<|endoftext|>"] 
    )

    # 4. Generate
    print("🚀 Starting vLLM Generation (Continuous Batching)...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. Process Results
    results = []
    all_pred_answers = []
    format_errors = 0

    print("Processing outputs and calculating metrics...")
    
    # outputs matches the order of prompts
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold_answers = gold_answers_list[i]
        
        reasoning_trace = ""
        predicted_answer = ""
        parse_success = False

        # --- Parsing Logic ---
        # Note: vLLM usually strips the 'stop' token from the output.
        # So we look for the arrow " --> "
        if DELIMITER_ANSWER in generated_text:
            parts = generated_text.split(DELIMITER_ANSWER)
            reasoning_trace = parts[0].strip()
            
            if len(parts) > 1:
                predicted_answer = parts[1].strip()
                # Clean up any residual stop tokens if they leaked
                predicted_answer = predicted_answer.replace("END", "").strip()
                if predicted_answer.endswith("."):
                    predicted_answer = predicted_answer[:-1]
                parse_success = True
            else:
                # Case where "-->" exists but nothing follows
                predicted_answer = ""
        else:
            format_errors += 1
            reasoning_trace = generated_text
            predicted_answer = "" # Fallback

        # --- Scoring ---
        em = max([float(exact_match_score(predicted_answer, g)) for g in gold_answers])
        f1 = max([f1_score(predicted_answer, g)[0] for g in gold_answers])

        all_pred_answers.append(predicted_answer)

        results.append({
            "id": ids_list[i],
            "question": questions_list[i],
            "gold_answers": gold_answers,
            "full_generation": generated_text,
            "parsed_reasoning": reasoning_trace,
            "parsed_answer": predicted_answer,
            "metrics": {"em": em, "f1": f1},
            "format_compliant": parse_success
        })

    # 6. Final Metrics
    metrics = compute_metrics(gold_answers_list, all_pred_answers)
    compliance_rate = ((len(dataset) - format_errors) / len(dataset)) * 100

    final_metrics = {
        "model": MODEL_ID,
        "dataset": DATASET_NAME,
        "subset": SUBSET_NAME,
        "split": SPLIT,
        "samples": len(dataset),
        "format_compliance_rate": compliance_rate,
        "exact_match": metrics['EM'],
        "f1_score": metrics['F1']
    }

    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Format Compliance: {compliance_rate:.2f}%")
    print(f"Exact Match (EM):  {metrics['EM']:.2f}%")
    print(f"F1 Score:          {metrics['F1']:.2f}%")
    print("="*30)

    # Save
    print(f"Saving detailed logs to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Saving metrics to {METRICS_FILE}...")
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4)
        
    print("Done.")

if __name__ == "__main__":
    main()