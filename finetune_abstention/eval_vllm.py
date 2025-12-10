import os
import json
import re
import string
import numpy as np
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

# vLLM Imports
from vllm import LLM, SamplingParams

# ==========================================
# 1. Configuration
# ==========================================

# CHANGED: Updated Model ID
MODEL_ID = "fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-Abstention-CoT-10000"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation"

# Delimiters
DELIMITER_PROMPT = " ###\n"
DELIMITER_ANSWER = " --> "
DELIMITER_END = " END"
REFUSAL_PHRASE = "i dont know"  # Normalized check string

# Output Files
OUTPUT_FILE = "qwen_abstention_cot_eval_results.json"
METRICS_FILE = "qwen_abstention_cot_eval_metrics.json"

# ==========================================
# 2. Metric Utilities
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

def check_abstention(prediction):
    """Returns True if the prediction is a refusal."""
    norm_pred = normalize_answer(prediction)
    # Check if the refusal phrase is in the output (or is the output)
    return REFUSAL_PHRASE in norm_pred

# ==========================================
# 3. Main Logic
# ==========================================

def main():
    print(f"Loading {DATASET_NAME} ({SUBSET_NAME})...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    
    # Optional: Slice for testing
    # dataset = dataset.select(range(100))
    
    print(f"Loaded {len(dataset)} samples.")

    # 1. Prepare Prompts
    prompts = [f"{item['question']}{DELIMITER_PROMPT}" for item in dataset]
    
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
        dtype="float16",             
        gpu_memory_utilization=0.90, 
        tensor_parallel_size=1,      
        max_model_len=2048,          
    )

    # 3. Sampling Params
    sampling_params = SamplingParams(
        temperature=0,      
        max_tokens=512,     
        stop=[DELIMITER_END, "<|im_end|>", "<|endoftext|>"] 
    )

    # 4. Generate
    print("🚀 Starting vLLM Generation...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. Process Results
    results = []
    
    # Metric Accumulators
    total_samples = 0
    total_abstentions = 0
    
    # Selective Lists (only for non-abstained samples)
    selective_em_scores = []
    selective_f1_scores = []
    
    # Standard Lists (all samples)
    standard_em_scores = []
    standard_f1_scores = []

    print("Processing outputs and calculating metrics...")
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold_answers = gold_answers_list[i]
        
        reasoning_trace = ""
        predicted_answer = ""
        
        # --- Robust Parsing Logic ---
        # Case A: Model used CoT (Found " --> ")
        if DELIMITER_ANSWER in generated_text:
            parts = generated_text.split(DELIMITER_ANSWER)
            reasoning_trace = parts[0].strip()
            if len(parts) > 1:
                predicted_answer = parts[1].strip()
        
        # Case B: Model Refused Immediately (No " --> ")
        # or Model outputted malformed text. We treat the whole text as the answer.
        else:
            reasoning_trace = "" # No reasoning
            predicted_answer = generated_text.strip()

        # Clean up tokens
        predicted_answer = predicted_answer.replace("END", "").strip()
        if predicted_answer.endswith("."):
            predicted_answer = predicted_answer[:-1]

        # --- Check Abstention ---
        is_abstained = check_abstention(predicted_answer)

        # --- Calculate Individual Scores ---
        # If abstained, scores are 0.0 for standard metrics
        em = max([float(exact_match_score(predicted_answer, g)) for g in gold_answers])
        f1 = max([f1_score(predicted_answer, g)[0] for g in gold_answers])

        # --- Aggregate ---
        total_samples += 1
        standard_em_scores.append(em)
        standard_f1_scores.append(f1)

        if is_abstained:
            total_abstentions += 1
            # For selective metrics, we simply DO NOT append to the selective list
        else:
            # If the model attempted an answer, we track it for selective stats
            selective_em_scores.append(em)
            selective_f1_scores.append(f1)

        results.append({
            "id": ids_list[i],
            "question": questions_list[i],
            "gold_answers": gold_answers,
            "full_generation": generated_text,
            "parsed_answer": predicted_answer,
            "is_abstention": is_abstained,
            "metrics": {"em": em, "f1": f1}
        })

    # 6. Final Metrics Calculation
    abstention_rate = (total_abstentions / total_samples) * 100
    
    standard_em = np.mean(standard_em_scores) * 100
    standard_f1 = np.mean(standard_f1_scores) * 100
    
    # Selective metrics: Handle case where model abstains 100% of the time (avoid div by zero)
    if len(selective_em_scores) > 0:
        selective_em = np.mean(selective_em_scores) * 100
        selective_f1 = np.mean(selective_f1_scores) * 100
    else:
        selective_em = 0.0
        selective_f1 = 0.0

    final_metrics = {
        "model": MODEL_ID,
        "dataset": DATASET_NAME,
        "samples": total_samples,
        "abstention_rate": abstention_rate,
        "standard_metrics": {
            "em": standard_em,
            "f1": standard_f1
        },
        "selective_metrics": {
            "em": selective_em,
            "f1": selective_f1,
            "answered_count": len(selective_em_scores)
        }
    }

    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Abstention Rate: {abstention_rate:.2f}%")
    print("-" * 20)
    print(f"Standard EM:     {standard_em:.2f}%")
    print(f"Standard F1:     {standard_f1:.2f}%")
    print("-" * 20)
    print(f"Selective EM:    {selective_em:.2f}%  (Accuracy on Answered)")
    print(f"Selective F1:    {selective_f1:.2f}%")
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