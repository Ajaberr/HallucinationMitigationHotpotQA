import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import string
from collections import Counter
import json
import os

# ==========================================
# 1. Configuration
# ==========================================

# Use the ID where you uploaded your fine-tuned model
MODEL_ID = "fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-CoT-Finetuned-2-10000" 
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation"

# BATCH SIZE NOTE: CoT generation (512 tokens) uses much more VRAM than short answers.
# We use 8 here to be safe. If you have an A100/H100, you can try increasing to 16 or 32.
BATCH_SIZE = 16

# Delimiters defined in your fine-tuning script
DELIMITER_PROMPT = " ###\n"
DELIMITER_ANSWER = " --> "
DELIMITER_END = " END"

OUTPUT_FILE = "qwen_ftcot_closedbook_full_results_2.json"
METRICS_FILE = "qwen_ftcot_closedbook_full_metrics_2.json"

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

def compute_metrics(gold_answers, pred_answers):
    # gold_answers is a list of lists (HotpotQA has multiple valid answers per q)
    # pred_answers is a list of strings
    em_total, f1_total = 0.0, 0.0
    for gold_list, pred in zip(gold_answers, pred_answers):
        # Handle case where gold_list might be a single string wrapped in list or just string
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
# 3. Model Loading
# ==========================================

def load_model():
    print(f"Loading model: {MODEL_ID}...")

    # # 1. Use 4-bit quantization (Same as training) to save bandwidth
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    # )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Critical for Batching: Left Padding
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use flash_attention_2 if available for speed, otherwise sdpa
    attn_impl = "sdpa"#"flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else "sdpa"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else torch.float16,
        attn_implementation=attn_impl,
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

# ==========================================
# 4. Main Evaluation Logic
# ==========================================

def main():
    tokenizer, model = load_model()
    if model is None: return
    device = model.device

    print(f"Loading {DATASET_NAME} ({SUBSET_NAME})...")
    # Load FULL validation set
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples.")

    # --- Data Loader Setup ---
    def collate_fn(batch):
        ids = [item.get('id', i) for i, item in enumerate(batch)]
        questions = [item['question'] for item in batch]
        # Normalize answer to list for metric calculation later
        answers = [item['answer'] if isinstance(item['answer'], list) else [item['answer']] for item in batch]
        return ids, questions, answers

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Storage
    results = []
    all_gold_answers = []
    all_pred_answers = []
    format_errors = 0

    print(f"🚀 Starting Batched CoT Evaluation (Batch Size: {BATCH_SIZE})...")
    
    for batch_ids, questions, batch_gold_answers in tqdm(dataloader, total=len(dataloader)):
        
        # 1. Format Prompts (Using your specific delimiters)
        prompts = [f"{q}{DELIMITER_PROMPT}" for q in questions]
        
        # 2. Tokenize
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        # 3. Generate (Batched)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512, # Long context for reasoning
                do_sample=False,    # Greedy
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 4. Decode
        # Slice off the input prompt part to get just the generated text
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 5. Parse & Compute Metrics for each item in batch
        for b in range(len(questions)):
            full_output = generated_texts[b]
            gold_answer_list = batch_gold_answers[b]
            
            reasoning_trace = ""
            predicted_answer = ""
            parse_success = False

            # --- Custom Parsing Logic ---
            if DELIMITER_ANSWER in full_output:
                parts = full_output.split(DELIMITER_ANSWER)
                reasoning_trace = parts[0].strip()
                
                # The answer is the part after the arrow
                raw_answer_part = parts[1]
                # Remove END token if present
                predicted_answer = raw_answer_part.replace(DELIMITER_END.strip(), "").strip()
                
                # Clean up trailing dots
                if predicted_answer.endswith("."):
                    predicted_answer = predicted_answer[:-1]
                    
                parse_success = True
            else:
                format_errors += 1
                reasoning_trace = full_output
                predicted_answer = "" # Fallback
            
            # --- Per-Sample Metrics ---
            # gold_answer_list is a list of valid answers. We take the max score against any valid answer.
            em = max([float(exact_match_score(predicted_answer, g)) for g in gold_answer_list])
            f1 = max([f1_score(predicted_answer, g)[0] for g in gold_answer_list])
            
            # Store for final aggregation
            all_gold_answers.append(gold_answer_list)
            all_pred_answers.append(predicted_answer)

            # Store detailed result
            results.append({
                "id": str(batch_ids[b]),
                "question": questions[b],
                "gold_answers": gold_answer_list,
                "full_generation": full_output,
                "parsed_reasoning": reasoning_trace,
                "parsed_answer": predicted_answer,
                "metrics": {"em": em, "f1": f1},
                "format_compliant": parse_success
            })

    # ==========================================
    # 5. Final Calculation & Saving
    # ==========================================
    
    metrics = compute_metrics(all_gold_answers, all_pred_answers)
    compliance_rate = ((total_samples - format_errors) / total_samples) * 100

    final_metrics = {
        "model": MODEL_ID,
        "dataset": DATASET_NAME,
        "subset": SUBSET_NAME,
        "samples": total_samples,
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

    # Save Detailed Results
    print(f"Saving detailed logs to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Saving as a proper JSON list (like Script 2) instead of JSONL for easier reading
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save Metrics
    print(f"Saving metrics to {METRICS_FILE}...")
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4)
        
    print("Done.")

if __name__ == "__main__":
    main()