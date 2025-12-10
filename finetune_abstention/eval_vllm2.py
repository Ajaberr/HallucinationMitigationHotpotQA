import os
import json
import re
import string
import numpy as np
import argparse
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

# vLLM Imports
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ==========================================
# 1. Configuration & Args
# ==========================================

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation"

# Delimiters
DELIMITER_PROMPT = " ###\n"
DELIMITER_ANSWER = " --> "
DELIMITER_END = " END"
REFUSAL_PHRASE = "i dont know" 

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter using vLLM")
    
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        required=True, 
        help="Path to the training run folder. Script will automatically look for 'final_adapter' inside."
    )
    
    return parser.parse_args()

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
    norm_pred = normalize_answer(prediction)
    return REFUSAL_PHRASE in norm_pred

# ==========================================
# 3. Main Logic
# ==========================================

def main():
    args = parse_args()
    
    # --- Intelligent Path Handling ---
    adapter_path = args.adapter_path.rstrip("/")
    
    # If the user points to the run folder, automatically append 'final_adapter'
    if not adapter_path.endswith("final_adapter"):
        potential_path = os.path.join(adapter_path, "final_adapter")
        if os.path.isdir(potential_path):
            print(f"Found 'final_adapter' inside run folder. Using: {potential_path}")
            adapter_path = potential_path
        else:
            # Fallback: maybe they pointed to a checkpoint folder or the raw files are there
            print(f"Warning: 'final_adapter' not found inside {adapter_path}. Trying to use path directly.")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Could not find adapter files at: {adapter_path}")

    # Determine Output Directory (Parent of the adapter folder)
    # e.g., if adapter_path is "./runs/run_name/final_adapter", output_dir is "./runs/run_name"
    output_dir = os.path.dirname(adapter_path)
    run_name = os.path.basename(output_dir) 

    # Construct full paths for results
    output_file_path = os.path.join(output_dir, f"eval_results_{run_name}.json")
    metrics_file_path = os.path.join(output_dir, f"eval_metrics_{run_name}.json")

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

    # 2. Initialize vLLM with LoRA Enabled
    print(f"Initializing vLLM with Base: {BASE_MODEL_ID} and Adapter: {adapter_path}...")
    llm = LLM(
        model=BASE_MODEL_ID,
        enable_lora=True,
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

    # 4. Generate with LoRA Request
    print("🚀 Starting vLLM Generation with Local Adapter...")
    
    lora_req = LoRARequest("adapter", 1, adapter_path)
    
    outputs = llm.generate(
        prompts, 
        sampling_params,
        lora_request=lora_req 
    )

    # 5. Process Results
    results = []
    
    total_samples = 0
    total_abstentions = 0
    
    selective_em_scores = []
    selective_f1_scores = []
    standard_em_scores = []
    standard_f1_scores = []

    print("Processing outputs and calculating metrics...")
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold_answers = gold_answers_list[i]
        
        reasoning_trace = ""
        predicted_answer = ""
        
        if DELIMITER_ANSWER in generated_text:
            parts = generated_text.split(DELIMITER_ANSWER)
            reasoning_trace = parts[0].strip()
            if len(parts) > 1:
                predicted_answer = parts[1].strip()
        else:
            reasoning_trace = "" 
            predicted_answer = generated_text.strip()

        predicted_answer = predicted_answer.replace("END", "").strip()
        if predicted_answer.endswith("."):
            predicted_answer = predicted_answer[:-1]

        is_abstained = check_abstention(predicted_answer)

        em = max([float(exact_match_score(predicted_answer, g)) for g in gold_answers])
        f1 = max([f1_score(predicted_answer, g)[0] for g in gold_answers])

        total_samples += 1
        standard_em_scores.append(em)
        standard_f1_scores.append(f1)

        if is_abstained:
            total_abstentions += 1
        else:
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
    
    if len(selective_em_scores) > 0:
        selective_em = np.mean(selective_em_scores) * 100
        selective_f1 = np.mean(selective_f1_scores) * 100
    else:
        selective_em = 0.0
        selective_f1 = 0.0

    final_metrics = {
        "base_model": BASE_MODEL_ID,
        "adapter_path": adapter_path,
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

    print(f"Saving detailed logs to: {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Saving metrics to: {metrics_file_path}")
    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4)
        
    print("Done.")

if __name__ == "__main__":
    main()