import os
import tinker
from tinker import types as ttypes
from dotenv import load_dotenv
import json
import re
import string
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

load_dotenv()

# ==========================================
# 1. Configuration
# ==========================================

# Load Registry to get URI
with open("tinker_models.json", "r") as f:
    MODEL_REGISTRY = json.load(f)

# Use the Abstention Model
ADAPTER_PATH = MODEL_REGISTRY["qwen_hotpot_abstention_v1"]["uri"]
BASE_MODEL = "Qwen/Qwen3-8B"

DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation"  

# Manual Format from Teammate's Script
DELIMITER_PROMPT = " ###\n"
DELIMITER_END = " END"
REFUSAL_PHRASE = "i dont know"

# Batching
BATCH_SIZE = 32
NUM_SAMPLES = 7500 # Align with other evals

OUTPUT_FILE = "tinker_abstention_results.json"
METRICS_FILE = "tinker_abstention_metrics.json"

# ==========================================
# 2. Metric Utils (Same as eval_vllm.py)
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
    print("🚀 Connecting to Tinker Service for Abstention Eval...")
    if not os.environ.get("TINKER_API_KEY"):
        print("⚠️ TINKER_API_KEY not found.")
        return

    service = tinker.ServiceClient()
    
    print(f"Initializing Sampling Client for {ADAPTER_PATH}...")
    sampling_client = service.create_sampling_client(model_path=ADAPTER_PATH)

    # Use Base Model Tokenizer
    training_client = service.create_lora_training_client(base_model=BASE_MODEL, rank=16)
    tokenizer = training_client.get_tokenizer()

    print(f"Loading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    # RESTORE FULL EVAL
    if NUM_SAMPLES:
        dataset = dataset.select(range(min(len(dataset), NUM_SAMPLES)))
    
    total_samples = len(dataset)
    print(f"Evaluating {total_samples} samples...")

    def chunked(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    # Helper for generation
    def process_single_sample(question):
        # Format Prompt: "Question ###\n"
        prompt_text = f"{question}{DELIMITER_PROMPT}"
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        model_input = ttypes.ModelInput.from_ints(input_ids)
        
        # Sampling Params
        sampling_params = ttypes.SamplingParams(
            max_tokens=512, 
            temperature=0.1,
        )
        
        try:
            result = sampling_client.sample(model_input, 1, sampling_params).result()
            output_ids = result.sequences[0].tokens
            output_text = tokenizer.decode(output_ids)
            
            # 1. Cleanup special tokens
            output_text = output_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
            if "assistant\n" in output_text:
                output_text = output_text.split("assistant\n")[-1]

            # 2. Check for " END" delimiter (from training)
            if " END" in output_text:
                output_text = output_text.split(" END")[0]
            
            output_text = output_text.strip()

            # DEBUG TRACING
            # print(f"RAW: {repr(output_text)}")

            # 3. Robust Parsing
            # Normalize dashes to standard hyphen to catch en-dash, em-dash, etc.
            normalized_text = output_text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
            
            parsed_answer = ""
            
            # Regex for arrow (handle variable spacing)
            arrow_match = re.search(r"\s*-{2,}>\s*", normalized_text)
            
            # PRIORITY 1: The Training Delimiter "-->"
            if arrow_match:
                # Split using the match
                parts = re.split(r"\s*-{2,}>\s*", normalized_text)
                parsed_answer = parts[-1].strip()
            
            # PRIORITY 2: Heuristic "Therefore..." (Only if arrow missing)
            elif "Therefore, the answer is" in normalized_text:
                parsed_answer = normalized_text.split("Therefore, the answer is")[-1].strip()
            
            elif "The answer is" in normalized_text:
                parsed_answer = normalized_text.split("The answer is")[-1].strip()
            
            else:
                # Fallback
                if "Step" in normalized_text: 
                    parsed_answer = "" 
                else:
                    parsed_answer = normalized_text

            # Final Cleanup
            if parsed_answer.endswith("."):
                parsed_answer = parsed_answer[:-1]
            
            return parsed_answer.strip()
            
        except Exception as e:
            return f"ERROR: {str(e)}"

    results = []
    
    # Store scores
    standard_em_scores = []
    standard_f1_scores = []
    selective_em_scores = [] # Only answered
    selective_f1_scores = [] # Only answered
    
    total_abstentions = 0

    print("Starting Parallel Generation...")
    
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        # Submit all jobs
        futures = {executor.submit(process_single_sample, item['question']): item for item in dataset}
        
        for future in tqdm(as_completed(futures), total=total_samples):
            item = futures[future]
            q = item['question']
            gold_answers = item['answer'] if isinstance(item['answer'], list) else [item['answer']]
            
            try:
                pred_text = future.result()
            except Exception:
                pred_text = ""
                
            # Check Abstention
            is_abstained = check_abstention(pred_text)
            
            # Metrics
            # If abstained, standard score = 0
            best_em = max([float(exact_match_score(pred_text, g)) for g in gold_answers])
            best_f1 = max([f1_score(pred_text, g)[0] for g in gold_answers])
            
            standard_em_scores.append(best_em)
            standard_f1_scores.append(best_f1)
            
            if is_abstained:
                total_abstentions += 1
            else:
                selective_em_scores.append(best_em)
                selective_f1_scores.append(best_f1)
            
            results.append({
                "question": q,
                "gold": gold_answers,
                "prediction": pred_text,
                "is_abstention": is_abstained,
                "metrics": {"em": best_em, "f1": best_f1}
            })

    # Final Stats
    abstention_rate = (total_abstentions / total_samples) * 100
    std_em = np.mean(standard_em_scores) * 100
    std_f1 = np.mean(standard_f1_scores) * 100
    
    sel_em = np.mean(selective_em_scores) * 100 if selective_em_scores else 0.0
    sel_f1 = np.mean(selective_f1_scores) * 100 if selective_f1_scores else 0.0
    
    print("\n" + "="*30)
    print("FINAL RESULTS (Abstention Model)")
    print("="*30)
    print(f"Abstention Rate: {abstention_rate:.2f}%")
    print("-" * 20)
    print(f"Standard EM:     {std_em:.2f}%")
    print(f"Standard F1:     {std_f1:.2f}%")
    print("-" * 20)
    print(f"Selective EM:    {sel_em:.2f}%  (Accuracy on Answered)")
    print(f"Selective F1:    {sel_f1:.2f}%")
    print("="*30)

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    final_metrics = {
        "model": ADAPTER_PATH,
        "abstention_rate": abstention_rate,
        "standard_em": std_em,
        "standard_f1": std_f1,
        "selective_em": sel_em,
        "selective_f1": sel_f1
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
