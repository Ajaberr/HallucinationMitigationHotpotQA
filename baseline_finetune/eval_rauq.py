import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import string
from collections import Counter
import json
import os
import numpy as np

# --- Configuration ---
MODEL_ID = "fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-Finetuned-Distractor-10000"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation" 

# RAUQ Config
RAUQ_ALPHA = 0.2

# NOTE: BATCH_SIZE lowered to 4 because 'output_attentions=True' consumes significant VRAM.
# If you have >48GB VRAM, you can try increasing this to 8 or 16.
BATCH_SIZE = 4 

# Output Setup
RESULTS_DIR = "results"
DETAILED_OUTPUT_FILE = os.path.join(RESULTS_DIR, "qwen_sft_closedbook_full_results.json")
FINAL_METRICS_FILE = os.path.join(RESULTS_DIR, "qwen_sft_closedbook_full_metrics.json")

def load_model_and_tokenizer(model_id):
    print(f"Loading model: {model_id}...")
    
    kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        # MUST use 'eager' to get access to full attention weights for RAUQ
        "attn_implementation": "eager" 
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

# --- RAUQ Implementation ---
def get_rauq_score_batch(sequences, scores, attentions, batch_idx, alpha=0.2, tokenizer=None):
    """
    Calculates RAUQ for a single item within a batch, ignoring padding tokens.
    """
    if len(scores) == 0:
        return 0.0

    # 1. Identify the actual length of this specific sample
    # We stop at the first occurrence of EOS or PAD in the generated sequence
    gen_tokens = sequences[batch_idx]
    
    # Define stop tokens (EOS and PAD)
    stop_tokens = {tokenizer.eos_token_id, tokenizer.pad_token_id} if tokenizer else set()
    
    actual_length = len(gen_tokens)
    for i, token_id in enumerate(gen_tokens):
        if token_id.item() in stop_tokens:
            actual_length = i
            break
            
    # If the sequence is too short (less than 2 tokens), we can't compute RAUQ 
    # (needs current and previous). 
    if actual_length < 2:
        return 0.0

    # 2. Get Probabilities only up to actual_length
    probs = []
    for i in range(actual_length):
        step_logits = scores[i]
        step_probs = F.softmax(step_logits, dim=-1)
        token_id = gen_tokens[i] 
        token_prob = step_probs[batch_idx, token_id].item()
        probs.append(token_prob)
    
    # 3. Compute RAUQ
    # attentions structure: [step][layer][batch, heads, q_len, k_len]
    # We only look at attentions up to actual_length
    num_layers = len(attentions[0])
    layer_uncertainties = []

    for layer_idx in range(num_layers):
        # --- Step 1: Head Selection ---
        prev_token_attns = []
        
        # Start at t=1 (second generated token) 
        for t in range(1, actual_length):
            attn_map = attentions[t][layer_idx] 
            # Extract attention to the previous token (index -2 in key dimension)
            attn_to_prev = attn_map[batch_idx, :, 0, -2]
            prev_token_attns.append(attn_to_prev)
            
        if not prev_token_attns:
            layer_uncertainties.append(0.0)
            continue
            
        prev_token_attns = torch.stack(prev_token_attns) 
        mean_head_attn = torch.mean(prev_token_attns, dim=0)
        best_head_idx = torch.argmax(mean_head_attn).item()
        
        # --- Step 2: Recurrent Confidence ---
        confidences = []
        current_conf = probs[0] 
        confidences.append(current_conf)
        
        for t in range(1, actual_length):
            prob_curr = probs[t]
            # Use the sliced limit for attentions loop as well
            attn_val = attentions[t][layer_idx][batch_idx, best_head_idx, 0, -2].item()
            
            # The RAUQ recursion
            current_conf = alpha * prob_curr + (1 - alpha) * attn_val * current_conf
            confidences.append(current_conf)
            
        # --- Step 3: Sequence Aggregation ---
        # Add epsilon to prevent log(0) if confidence is extremely low
        log_confs = [torch.log(torch.tensor(c + 1e-9)) for c in confidences]
        layer_u = -torch.mean(torch.stack(log_confs)).item()
        layer_uncertainties.append(layer_u)

    if not layer_uncertainties:
        return 0.0
        
    return max(layer_uncertainties)

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
    # Ensure results directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"Creating output directory: {RESULTS_DIR}")
        os.makedirs(RESULTS_DIR, exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(MODEL_ID)
    if model is None: return
    device = model.device

    print(f"\nLoading HotpotQA dataset (split: {SPLIT})...")
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
    rauq_scores = []
    detailed_results = []

    print(f"\nStarting evaluation with RAUQ (Batch Size: {BATCH_SIZE})...")

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
                pad_token_id=tokenizer.pad_token_id,
                output_attentions=True,  # Required for RAUQ
                output_scores=True,      # Required for RAUQ
                return_dict_in_generate=True
            )

        # 3. Process Batch Results
        input_len = inputs['input_ids'].shape[1]
        
        # Slice the generated sequences (remove prompt)
        gen_sequences = outputs.sequences[:, input_len:]
        
        batch_decoded = tokenizer.batch_decode(gen_sequences, skip_special_tokens=True)

        for b in range(len(questions)):
            pred_text = batch_decoded[b].strip()
            current_gold = gold_answers[b]
            
            # Calculate RAUQ
            rauq = get_rauq_score_batch(
                sequences=gen_sequences,
                scores=outputs.scores,
                attentions=outputs.attentions,
                batch_idx=b,
                alpha=RAUQ_ALPHA,
                tokenizer=tokenizer
            )
            
            all_gold_answers.append(current_gold)
            all_pred_answers.append(pred_text)
            rauq_scores.append(rauq)

            # Metrics
            sample_em = max([float(exact_match_score(pred_text, gold)) for gold in current_gold])
            sample_f1 = max([f1_score(pred_text, gold)[0] for gold in current_gold])

            detailed_results.append({
                "id": str(batch_ids[b]),
                "question": questions[b],
                "gold_answers": current_gold,
                "prediction": pred_text,
                "metrics": {
                    "rauq_score": float(rauq),
                    "exact_match": sample_em,
                    "f1_score": sample_f1
                }
            })

        # Clear VRAM after every batch because attentions are huge
        del outputs
        del inputs
        torch.cuda.empty_cache()

    # --- Final Computations ---
    metrics = compute_metrics(all_gold_answers, all_pred_answers)
    avg_rauq = sum(rauq_scores) / len(rauq_scores) if rauq_scores else 0.0

    final_results = {
        "model": MODEL_ID,
        "dataset": DATASET_NAME,
        "setting": SUBSET_NAME,
        "split": SPLIT,
        "samples": total_samples,
        "EM": metrics['EM'],
        "F1": metrics['F1'],
        "Avg_RAUQ": avg_rauq,
        "RAUQ_alpha": RAUQ_ALPHA
    }

    print(f"\nResults for {MODEL_ID} on HotpotQA (Full {SPLIT}):")
    print(f"  Exact Match (EM): {metrics['EM']:.2f}%")
    print(f"  F1 Score (F1): {metrics['F1']:.2f}%")
    print(f"  Avg RAUQ: {avg_rauq:.4f}")

    print(f"Saving detailed results to {DETAILED_OUTPUT_FILE}...")
    with open(DETAILED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
    print(f"Saving final metrics to {FINAL_METRICS_FILE}...")
    with open(FINAL_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    main()