import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import string
from collections import Counter
import json
import os

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation"  # Validating on the full validation split (~7.4k samples)
BATCH_SIZE = 32
DETAILED_OUTPUT_FILE = "qwen_base_closedbook_full_results.json"
FINAL_METRICS_FILE = "qwen_based_closedbook_full_metrics.json"

def load_model_and_tokenizer(model_id):
    print(f"Loading model: {model_id}...")
    
    kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa" 
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.padding_side = "left" 
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# --- Metric Utils ---
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
        best_em = max([float(exact_match_score(pred, gold)) for gold in gold_list])
        best_f1 = max([f1_score(pred, gold)[0] for gold in gold_list])
        em_total += best_em
        f1_total += best_f1
    return {
        "EM": (em_total / len(gold_answers)) * 100,
        "F1": (f1_total / len(gold_answers)) * 100,
    }

# --- Main Logic ---
def main():
    tokenizer, model = load_model_and_tokenizer(MODEL_ID)
    if model is None: return
    device = model.device

    print(f"\nLoading HotpotQA dataset (split: {SPLIT})...")
    # UPDATED: Load the full split without slicing/selecting
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples.")
    
    # Helper to format prompts
    def format_batch_prompts(questions):
        prompts = []
        for q in questions:
            messages = [
                {"role": "system", "content": "You are a concise encyclopedia. Answer the question directly with a short phrase or entity name. Do not explain."},
                {"role": "user", "content": q}
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        return prompts

    # Prepare Data Loader
    def collate_fn(batch):
        questions = [item['question'] for item in batch]
        gold_answers = [item['answer'] if isinstance(item['answer'], list) else [item['answer']] for item in batch]
        ids = [item.get('id', i) for i, item in enumerate(batch)]
        return ids, questions, gold_answers

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    all_gold_answers = []
    all_pred_answers = []
    detailed_results = []

    print(f"\nStarting high-speed closed-book evaluation on FULL dataset (Batch Size: {BATCH_SIZE})...")

    for batch_ids, questions, gold_answers in tqdm(dataloader, total=len(dataloader)):
        
        # 1. Tokenize Batch
        prompts = format_batch_prompts(questions)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        # 2. Batch Generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False, 
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id
            )

        # 3. Process Batch Results
        input_len = inputs['input_ids'].shape[1]
        gen_sequences = outputs[:, input_len:]
        
        batch_decoded = tokenizer.batch_decode(gen_sequences, skip_special_tokens=True)

        for b in range(len(questions)):
            pred_text = batch_decoded[b].strip()
            current_gold = gold_answers[b]
            
            all_gold_answers.append(current_gold)
            all_pred_answers.append(pred_text)

            # Metrics
            sample_em = max([float(exact_match_score(pred_text, gold)) for gold in current_gold])
            sample_f1 = max([f1_score(pred_text, gold)[0] for gold in current_gold])

            detailed_results.append({
                "id": str(batch_ids[b]),
                "question": questions[b],
                "gold_answers": current_gold,
                "prediction": pred_text,
                "metrics": {
                    "exact_match": sample_em,
                    "f1_score": sample_f1
                }
            })

    # --- Final Computations ---
    metrics = compute_metrics(all_gold_answers, all_pred_answers)

    final_results = {
        "model": MODEL_ID,
        "dataset": DATASET_NAME,
        "setting": SUBSET_NAME,
        "split": SPLIT,
        "samples": total_samples, # Dynamic count
        "EM": metrics['EM'],
        "F1": metrics['F1']
    }

    print(f"\nResults for {MODEL_ID} on HotpotQA (Full {SPLIT}):")
    print(f"  Exact Match (EM): {metrics['EM']:.2f}%")
    print(f"  F1 Score (F1): {metrics['F1']:.2f}%")

    print(f"Saving detailed results to {DETAILED_OUTPUT_FILE}...")
    with open(DETAILED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
    print(f"Saving final metrics to {FINAL_METRICS_FILE}...")
    with open(FINAL_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    main()