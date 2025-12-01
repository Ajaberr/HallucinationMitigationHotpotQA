import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
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
MODEL_ID = "fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-CoT-Finetuned-1000" 
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor" # Using distractor to match typical Hotpot eval
SPLIT = "validation"
NUM_SAMPLES = 1000 # Adjust as needed

# NEW: How often to save intermediate results
SAVE_INTERVAL = 100

# Delimiters defined in your fine-tuning script
DELIMITER_PROMPT = " ###\n"
DELIMITER_ANSWER = " --> "
DELIMITER_END = " END"

OUTPUT_FILE = "cot_eval_results.json"
METRICS_FILE = "cot_eval_metrics.json"

# ==========================================
# 2. Metric Utilities (Standard HotpotQA)
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

# ==========================================
# 3. Model Loading
# ==========================================

def load_model():
    print(f"Loading model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Ensure padding side is correct for generation
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

# ==========================================
# 4. Main Evaluation Logic
# ==========================================

def main():
    tokenizer, model = load_model()
    device = model.device

    print(f"Loading {DATASET_NAME} ({SUBSET_NAME})...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    
    # If using streaming or large dataset, slice it
    if NUM_SAMPLES > 0:
        dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))

    results = []
    
    # Metrics Accumulators
    total_em = 0
    total_f1 = 0
    format_errors = 0

    print("🚀 Starting CoT Evaluation...")
    
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        question = example['question']
        gold_answer = example['answer'] 
        
        # 1. Format Prompt (Matches Fine-Tuning Logic)
        prompt_text = f"{question}{DELIMITER_PROMPT}"
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        # 2. Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512, # Allow enough space for CoT
                do_sample=False,    # Greedy decoding for reproducible eval
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 3. Decode
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 4. Parse (Separate Reasoning from Answer)
        reasoning_trace = ""
        predicted_answer = ""
        parse_success = False

        if DELIMITER_ANSWER in full_output:
            parts = full_output.split(DELIMITER_ANSWER)
            reasoning_trace = parts[0].strip()
            
            # The answer is the part after the arrow, before the END delimiter
            raw_answer_part = parts[1]
            predicted_answer = raw_answer_part.replace(DELIMITER_END.strip(), "").strip()
            
            # Clean up any trailing dots if the model added them before END
            if predicted_answer.endswith("."):
                predicted_answer = predicted_answer[:-1]
                
            parse_success = True
        else:
            format_errors += 1
            reasoning_trace = full_output
            predicted_answer = "" 
        
        # 5. Calculate Metrics
        em = exact_match_score(predicted_answer, gold_answer)
        f1, _, _ = f1_score(predicted_answer, gold_answer)
        
        total_em += em
        total_f1 += f1

        # 6. Store Result
        results.append({
            "id": example.get('id', i),
            "question": question,
            "gold_answer": gold_answer,
            "full_generation": full_output,
            "parsed_reasoning": reasoning_trace,
            "parsed_answer": predicted_answer,
            "metrics": {"em": em, "f1": f1},
            "format_compliant": parse_success
        })

        if i % 50 == 0:
            print(f"\n[Sample {i}] Q: {question[:50]}...")
            print(f"   Pred: {predicted_answer} | Gold: {gold_answer}")
            print(f"   Format OK: {parse_success} | EM: {em}")

        # === NEW: Checkpoint Saving ===
        # Save every SAVE_INTERVAL examples
        if (i + 1) % SAVE_INTERVAL == 0:
            print(f" 💾 [Checkpoint] Saving {len(results)} examples to {OUTPUT_FILE}...")
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")

    # ==========================================
    # 5. Final Report
    # ==========================================
    
    avg_em = (total_em / len(dataset)) * 100
    avg_f1 = (total_f1 / len(dataset)) * 100
    compliance_rate = ((len(dataset) - format_errors) / len(dataset)) * 100

    final_metrics = {
        "model": MODEL_ID,
        "samples": len(dataset),
        "exact_match": avg_em,
        "f1_score": avg_f1,
        "format_compliance_rate": compliance_rate
    }

    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Format Compliance: {compliance_rate:.2f}%")
    print(f"Exact Match (EM):  {avg_em:.2f}%")
    print(f"F1 Score:          {avg_f1:.2f}%")
    print("="*30)

    # Save to files (Final Save)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"Detailed logs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()