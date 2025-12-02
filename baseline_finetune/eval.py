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

# --- Configuration ---
MODEL_ID = "fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-Finetuned-Distractor-10000"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation"
NUM_SAMPLES = 1000 # Adjusted for testing
RAUQ_ALPHA = 0.2
DETAILED_OUTPUT_FILE = "detailed_results.json"
FINAL_METRICS_FILE = "final_metrics.json"

def load_model_and_tokenizer(model_id):
    print(f"Loading model: {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Qwen 2.5 handles Flash Attention 2 if available, but "eager" is safer for extracting attention weights
    kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager" # Mandatory for output_attentions=True stability
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Ensure pad token exists for batching/generation logic
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# --- RAUQ Implementation ---
def get_rauq_score(sequences, scores, attentions, alpha=0.2):
    if len(scores) == 0:
        return 0.0

    # Convert logits to probabilities
    probs = []
    for i, step_logits in enumerate(scores):
        step_probs = F.softmax(step_logits, dim=-1)
        token_id = sequences[0][i] 
        token_prob = step_probs[0, token_id].item()
        probs.append(token_prob)
    
    num_tokens = len(probs)
    
    # SAFETY CHECK: RAUQ requires at least 2 tokens (current + previous)
    # If answer is "Yes", only 1 token exists. No transition to analyze.
    if num_tokens < 2:
        return 0.0

    num_layers = len(attentions[0])
    layer_uncertainties = []

    for layer_idx in range(num_layers):
        # --- Step 1: Head Selection ---
        prev_token_attns = []
        
        # We start at t=1 because t=0 has no "previous generated token" (only prompt)
        for t in range(1, num_tokens):
            # attentions shape: [step][layer][batch, heads, query_len, key_len]
            # With caching, query_len=1. key_len grows.
            # Index -1 is self (current token). Index -2 is immediately previous token.
            attn_map = attentions[t][layer_idx] 
            
            # Extract attention to the previous token (index -2)
            attn_to_prev = attn_map[0, :, 0, -2]
            prev_token_attns.append(attn_to_prev)
            
        if not prev_token_attns:
            layer_uncertainties.append(0.0)
            continue
            
        prev_token_attns = torch.stack(prev_token_attns) 
        mean_head_attn = torch.mean(prev_token_attns, dim=0)
        best_head_idx = torch.argmax(mean_head_attn).item()
        
        # --- Step 2: Recurrent Confidence ---
        confidences = []
        current_conf = probs[0] # Initial confidence is just the prob of first token
        confidences.append(current_conf)
        
        for t in range(1, num_tokens):
            prob_curr = probs[t]
            attn_val = attentions[t][layer_idx][0, best_head_idx, 0, -2].item()
            
            # The RAUQ recursion
            current_conf = alpha * prob_curr + (1 - alpha) * attn_val * current_conf
            confidences.append(current_conf)
            
        # --- Step 3: Sequence Aggregation ---
        log_confs = [torch.log(torch.tensor(c + 1e-9)) for c in confidences]
        layer_u = -torch.mean(torch.stack(log_confs)).item()
        layer_uncertainties.append(layer_u)

    # --- Step 4: Final Score ---
    # If generation was short/failed, handle empty list
    if not layer_uncertainties:
        return 0.0
        
    final_rauq_score = max(layer_uncertainties)
    return final_rauq_score

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

def main():
    tokenizer, model, device = load_model_and_tokenizer(MODEL_ID)
    if model is None: return

    print(f"\nLoading HotpotQA dataset (split: {SPLIT}, samples: {NUM_SAMPLES})...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    # Shuffle to get a variety of questions, then select
    dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
    
    gold_answers = []
    pred_answers = []
    rauq_scores = []
    detailed_results = []

    print("\nStarting closed-book evaluation with RAUQ...")

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        question = example['question']
        gold_answer_list = example['answer'] if isinstance(example['answer'], list) else [example['answer']]
        gold_answers.append(gold_answer_list)

        # --- IMPROVED PROMPT ---
        # Using system role for constraint and User role for question.
        # This prevents "Answer: Answer: " repetitions.
        messages = [
            {"role": "system", "content": "You are a concise encyclopedia. Answer the question directly with a short phrase or entity name. Do not explain."},
            {"role": "user", "content": question}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False, # Greedy decoding preferred for RAUQ baselines
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Extract generated sequence
        gen_sequence = outputs.sequences[:, inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(gen_sequence[0], skip_special_tokens=True).strip()
        pred_answers.append(generated_text)

        # Calculate RAUQ Score
        rauq = get_rauq_score(
            sequences=gen_sequence,
            scores=outputs.scores,
            attentions=outputs.attentions,
            alpha=RAUQ_ALPHA
        )
        rauq_scores.append(rauq)
        
        # Cleanup VRAM - Critical for output_attentions
        del outputs
        # Only empty cache occasionally to save speed, or every time if OOM occurs
        if i % 10 == 0: 
            torch.cuda.empty_cache()

        # Metrics for this sample
        sample_em = max([float(exact_match_score(generated_text, gold)) for gold in gold_answer_list])
        sample_f1 = max([f1_score(generated_text, gold)[0] for gold in gold_answer_list])

        result_entry = {
            "id": i,
            "question": question,
            "gold_answers": gold_answer_list,
            "prediction": generated_text,
            "metrics": {
                "rauq_score": float(rauq),
                "exact_match": sample_em,
                "f1_score": sample_f1
            }
        }
        detailed_results.append(result_entry)

        if i % (NUM_SAMPLES // 5 or 1) == 0 and i > 0:
            print(f"\nSample {i}: {question[:50]}... -> {generated_text} | RAUQ: {rauq:.4f}")

    # --- Final Computations ---
    metrics = compute_metrics(gold_answers, pred_answers)
    avg_rauq = sum(rauq_scores) / len(rauq_scores) if rauq_scores else 0.0

    final_results = {
        "model": MODEL_ID,
        "dataset": DATASET_NAME,
        "setting": SUBSET_NAME,
        "split": SPLIT,
        "samples": NUM_SAMPLES,
        "EM": metrics['EM'],
        "F1": metrics['F1'],
        "Avg_RAUQ": avg_rauq,
        "RAUQ_alpha": RAUQ_ALPHA
    }

    print(f"\nResults for {MODEL_ID} on HotpotQA:")
    print(f"  Exact Match (EM): {metrics['EM']:.2f}%")
    print(f"  F1 Score (F1): {metrics['F1']:.2f}%")
    print(f"  Avg RAUQ Uncertainty: {avg_rauq:.4f}")

    print(f"Saving detailed results to {DETAILED_OUTPUT_FILE}...")
    with open(DETAILED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
    print(f"Saving final metrics to {FINAL_METRICS_FILE}...")
    with open(FINAL_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    main()