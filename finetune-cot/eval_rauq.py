import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
import string
from collections import Counter
import json
import os
import numpy as np

# ==========================================
# 1. Configuration
# ==========================================

MODEL_ID = "fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-CoT-Finetuned-10000"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation"

# Batching Configuration 
# (4-8 is usually safe for 24GB VRAM with output_attentions=True)
BATCH_SIZE = 4 
RAUQ_ALPHA = 0.2

# Parsing Constants
DELIMITER_PROMPT = " ###\n"
DELIMITER_ANSWER = " --> "
DELIMITER_END = " END"

# Output Directory & Files
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "qwen_ftcot_closedbook_rauq_results.json")
METRICS_FILE = os.path.join(RESULTS_DIR, "qwen_ftcot_closedbook_rauq_metrics.json")

# ==========================================
# 2. RAUQ Implementation (Adapted for Batching)
# ==========================================

def get_rauq_score_batch(sequences, scores, attentions, batch_idx, alpha=0.2):
    """
    Calculates RAUQ for a single item within a batch.
    """
    if len(scores) == 0:
        return 0.0

    # 1. Get Probabilities for the generated sequence
    probs = []
    # scores is a tuple (one per step) of tensors (Batch, Vocab)
    for i, step_logits in enumerate(scores):
        step_probs = F.softmax(step_logits, dim=-1)
        # Sequence is already sliced to just generated tokens
        token_id = sequences[batch_idx][i] 
        token_prob = step_probs[batch_idx, token_id].item()
        probs.append(token_prob)
    
    num_tokens = len(probs)
    
    # RAUQ requires at least 2 tokens (current + previous)
    if num_tokens < 2:
        return 0.0

    # attentions structure: [step][layer][batch, heads, q_len, k_len]
    # We assume 'eager' attention implementation where q_len=1
    num_layers = len(attentions[0])
    layer_uncertainties = []

    for layer_idx in range(num_layers):
        # --- Step 1: Head Selection ---
        prev_token_attns = []
        
        # Start at t=1 (second generated token) to look back at t=0 (first generated token)
        for t in range(1, num_tokens):
            attn_map = attentions[t][layer_idx] 
            
            # Extract attention to the previous token (index -2 in key dimension)
            # attn_map shape: [batch, heads, 1, keys]
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
        
        for t in range(1, num_tokens):
            prob_curr = probs[t]
            # Get attention value of the "best head" for this specific step
            attn_val = attentions[t][layer_idx][batch_idx, best_head_idx, 0, -2].item()
            
            # The RAUQ recursion
            current_conf = alpha * prob_curr + (1 - alpha) * attn_val * current_conf
            confidences.append(current_conf)
            
        # --- Step 3: Sequence Aggregation ---
        log_confs = [torch.log(torch.tensor(c + 1e-9)) for c in confidences]
        layer_u = -torch.mean(torch.stack(log_confs)).item()
        layer_uncertainties.append(layer_u)

    if not layer_uncertainties:
        return 0.0
        
    return max(layer_uncertainties)

# ==========================================
# 3. Metric Utilities
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

def compute_metrics(gold_answers_list, pred_answers):
    em_total, f1_total = 0.0, 0.0
    for gold_list, pred in zip(gold_answers_list, pred_answers):
        if isinstance(gold_list, str): gold_list = [gold_list]
        best_em = max([float(exact_match_score(pred, gold)) for gold in gold_list])
        best_f1 = max([f1_score(pred, gold)[0] for gold in gold_list])
        em_total += best_em
        f1_total += best_f1
    return {
        "EM": (em_total / len(gold_answers_list)) * 100,
        "F1": (f1_total / len(gold_answers_list)) * 100,
    }

# ==========================================
# 4. Main Execution
# ==========================================

def load_model(model_id):
    print(f"Loading model: {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Must use 'eager' attention to get full attention weights for RAUQ
    kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else torch.float16,
        "device_map": "auto",
        "attn_implementation": "eager" 
    }

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Important for batch generation
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model, device

def batch_data(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def main():
    # Ensure results directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"Creating output directory: {RESULTS_DIR}")
        os.makedirs(RESULTS_DIR, exist_ok=True)

    tokenizer, model, device = load_model(MODEL_ID)
    
    print(f"Loading {DATASET_NAME} ({SUBSET_NAME})...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    
    # OPTIONAL: Testing slice
    # dataset = dataset.select(range(50))
    
    print(f"Loaded {len(dataset)} samples.")

    detailed_results = []
    all_pred_answers = []
    all_gold_answers = []
    rauq_scores = []
    format_errors = 0

    print(f"Starting batched generation (Batch Size: {BATCH_SIZE})...")

    # Iterate through dataset in batches
    for batch in tqdm(batch_data(dataset, BATCH_SIZE), total=(len(dataset) // BATCH_SIZE) + 1):
        
        # 1. Prepare Batch Prompts
        questions = batch['question']
        ids = batch['id']
        golds = [a if isinstance(a, list) else [a] for a in batch['answer']]
        
        prompts = [f"{q}{DELIMITER_PROMPT}" for q in questions]
        
        # 2. Tokenize
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # 3. Generate with Attentions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, # Greedy
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
                stop_strings=[DELIMITER_END, "<|im_end|>"], 
                tokenizer=tokenizer
            )

        # 4. Process Batch Results
        # Slice out the prompt to get just generated tokens
        gen_sequences = outputs.sequences[:, inputs['input_ids'].shape[1]:]
        
        for i, sequence in enumerate(gen_sequences):
            # A. Decode Text
            full_text = tokenizer.decode(sequence, skip_special_tokens=True).strip()
            
            # B. Calculate RAUQ
            # Pass the batch index 'i' so RAUQ extracts the correct row from attentions
            rauq = get_rauq_score_batch(
                sequences=gen_sequences,
                scores=outputs.scores,
                attentions=outputs.attentions,
                batch_idx=i,
                alpha=RAUQ_ALPHA
            )
            rauq_scores.append(rauq)
            
            # C. Parse Output (CoT Logic)
            reasoning_trace = ""
            predicted_answer = ""
            parse_success = False
            
            if DELIMITER_ANSWER in full_text:
                parts = full_text.split(DELIMITER_ANSWER)
                reasoning_trace = parts[0].strip()
                if len(parts) > 1:
                    predicted_answer = parts[1].replace("END", "").strip()
                    if predicted_answer.endswith("."):
                        predicted_answer = predicted_answer[:-1]
                    parse_success = True
            else:
                format_errors += 1
                reasoning_trace = full_text
            
            # D. Scoring
            current_golds = golds[i]
            em = max([float(exact_match_score(predicted_answer, g)) for g in current_golds])
            f1 = max([f1_score(predicted_answer, g)[0] for g in current_golds])
            
            all_pred_answers.append(predicted_answer)
            all_gold_answers.append(current_golds)

            detailed_results.append({
                "id": ids[i],
                "question": questions[i],
                "gold_answers": current_golds,
                "full_generation": full_text,
                "parsed_reasoning": reasoning_trace,
                "parsed_answer": predicted_answer,
                "metrics": {
                    "rauq_score": float(rauq),
                    "em": em, 
                    "f1": f1
                },
                "format_compliant": parse_success
            })

        # Memory Cleanup (Critical when returning attentions)
        del outputs
        del inputs
        torch.cuda.empty_cache()

    # ==========================================
    # 5. Final Metrics & Save
    # ==========================================
    
    metrics = compute_metrics(all_gold_answers, all_pred_answers)
    avg_rauq = sum(rauq_scores) / len(rauq_scores) if rauq_scores else 0.0
    compliance_rate = ((len(dataset) - format_errors) / len(dataset)) * 100

    final_results = {
        "model": MODEL_ID,
        "dataset": DATASET_NAME,
        "subset": SUBSET_NAME,
        "samples": len(dataset),
        "format_compliance_rate": compliance_rate,
        "EM": metrics['EM'],
        "F1": metrics['F1'],
        "Avg_RAUQ": avg_rauq,
        "RAUQ_alpha": RAUQ_ALPHA
    }

    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Exact Match (EM):  {metrics['EM']:.2f}%")
    print(f"F1 Score:          {metrics['F1']:.2f}%")
    print(f"Avg RAUQ:          {avg_rauq:.4f}")
    print(f"Compliance Rate:   {compliance_rate:.2f}%")
    print("="*30)

    print(f"Saving detailed results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
    print(f"Saving metrics to {METRICS_FILE}...")
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    main()