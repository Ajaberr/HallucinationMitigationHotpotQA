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

# ==========================================
# 1. Configuration
# ==========================================

MODEL_ID = "fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-CoT-Finetuned-1000" 
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor" 
SPLIT = "validation"
NUM_SAMPLES = 1000 

# RAUQ Config
RAUQ_ALPHA = 0.2

# BATCH SIZE WARNING: 
# RAUQ requires storing attention weights for every token. This is VRAM heavy.
# If you get OOM (Out of Memory), reduce this to 2 or 1.
BATCH_SIZE = 4

SAVE_INTERVAL = 50

DELIMITER_PROMPT = " ###\n"
DELIMITER_ANSWER = " --> "
DELIMITER_END = " END"

OUTPUT_FILE = "cot_rauq_eval_results.json"
METRICS_FILE = "cot_rauq_eval_metrics.json"

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

# ==========================================
# 3. RAUQ Implementation (Batch Adapted)
# ==========================================

def get_rauq_score(batch_idx, sequences, scores, attentions, alpha=0.2, pad_token_id=None):
    """
    Calculates RAUQ score for a specific item (batch_idx) within a batch.
    """
    if len(scores) == 0:
        return 0.0

    # 1. Identify the generated tokens for this specific batch index
    # sequences shape: [batch_size, full_seq_len]
    # We need to isolate the *new* tokens to align with scores/attentions
    
    # Scores is a tuple of length `gen_len`. Each element is tensor [batch, vocab]
    probs = []
    
    # Iterate through generation steps
    for t, step_logits in enumerate(scores):
        # step_logits: [batch, vocab]
        step_probs = F.softmax(step_logits, dim=-1)
        
        # Get the token ID that was actually generated at this step for this batch index
        # Note: sequences includes prompt. 
        # But `scores` only aligns with new tokens.
        # We need to find where the new tokens start in `sequences`.
        # However, it's easier to just look at the max of the logits or trust the decoder loop structure.
        # Actually, let's use the sequence directly, but we must offset by prompt length.
        # To simplify: The `scores` index `t` corresponds to the generation step.
        
        # Let's extract the token ID from the logits (greedy assumption) or use the passed sequence if available.
        # Since we ran greedy decoding, argmax of logits is the token.
        token_id = torch.argmax(step_logits[batch_idx]).item()
        
        # Stop if we hit a pad token (which means this specific sequence finished early)
        if pad_token_id is not None and token_id == pad_token_id:
            break
            
        token_prob = step_probs[batch_idx, token_id].item()
        probs.append(token_prob)
    
    num_tokens = len(probs)
    
    # RAUQ requires at least 2 tokens (current + previous)
    if num_tokens < 2:
        return 0.0

    # `attentions` structure: tuple(steps) -> tuple(layers) -> tensor(batch, heads, 1, past_len)
    # We grab the first step's first layer to count layers
    num_layers = len(attentions[0])
    layer_uncertainties = []

    for layer_idx in range(num_layers):
        # --- Step 1: Head Selection ---
        prev_token_attns = []
        
        # We start at t=1 because t=0 has no "previous generated token" (only prompt)
        for t in range(1, num_tokens):
            # attentions[step][layer] -> [batch, heads, q_len, k_len]
            attn_map = attentions[t][layer_idx] 
            
            # Extract attention for specific batch_idx, specific layer
            # Index -1 is self (current token). Index -2 is immediately previous token.
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
            # Get attention value for the best head, specific batch index
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
        
    final_rauq_score = max(layer_uncertainties)
    return final_rauq_score

# ==========================================
# 4. Model Loading
# ==========================================

def load_model():
    print(f"Loading model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # RAUQ Requirement: attn_implementation="eager"
    # Flash Attention often does not return the full attention weights needed for RAUQ
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        trust_remote_code=True,
        attn_implementation="eager" 
    )
    model.eval()
    return tokenizer, model

# ==========================================
# 5. Main Evaluation Logic
# ==========================================

def main():
    tokenizer, model = load_model()
    device = model.device

    print(f"Loading {DATASET_NAME} ({SUBSET_NAME})...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    
    if NUM_SAMPLES > 0:
        dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))

    results = []
    total_em = 0
    total_f1 = 0
    total_rauq = 0
    format_errors = 0

    print(f"🚀 Starting CoT + RAUQ Eval (Batch Size: {BATCH_SIZE})...")
    
    # Process in Batches
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Processing Batches"):
        
        # 1. Batch Slice
        batch_indices = list(range(i, min(i + BATCH_SIZE, len(dataset))))
        batch_data = dataset.select(batch_indices)
        
        batch_questions = batch_data['question']
        batch_answers = batch_data['answer']
        
        if 'id' in batch_data.column_names:
            batch_ids = batch_data['id']
        else:
            batch_ids = batch_indices

        # 2. Prepare Prompts
        prompts = [f"{q}{DELIMITER_PROMPT}" for q in batch_questions]

        # 3. Tokenize
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048 
        ).to(device)

        # 4. Generate with Attentions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_attentions=True, # Required for RAUQ
                output_scores=True,     # Required for RAUQ
                return_dict_in_generate=True
            )

        # 5. Decode
        # outputs.sequences contains [prompt + generated]
        generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
        decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 6. Process Batch Results
        for j, full_output in enumerate(decoded_outputs):
            
            # --- Calculate RAUQ for this sample ---
            # We pass the batch index 'j' to extract the specific attention slice
            rauq_score = get_rauq_score(
                batch_idx=j,
                sequences=outputs.sequences,
                scores=outputs.scores,
                attentions=outputs.attentions,
                alpha=RAUQ_ALPHA,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # --- Parsing & Standard Metrics ---
            question = batch_questions[j]
            gold_answer = batch_answers[j]
            example_id = batch_ids[j]

            reasoning_trace = ""
            predicted_answer = ""
            parse_success = False

            if DELIMITER_ANSWER in full_output:
                parts = full_output.split(DELIMITER_ANSWER)
                reasoning_trace = parts[0].strip()
                raw_answer_part = parts[1]
                predicted_answer = raw_answer_part.replace(DELIMITER_END.strip(), "").strip()
                if predicted_answer.endswith("."):
                    predicted_answer = predicted_answer[:-1]
                parse_success = True
            else:
                format_errors += 1
                reasoning_trace = full_output
                predicted_answer = "" 
            
            em = exact_match_score(predicted_answer, gold_answer)
            f1, _, _ = f1_score(predicted_answer, gold_answer)
            
            total_em += em
            total_f1 += f1
            total_rauq += rauq_score

            results.append({
                "id": example_id,
                "question": question,
                "gold_answer": gold_answer,
                "full_generation": full_output,
                "parsed_reasoning": reasoning_trace,
                "parsed_answer": predicted_answer,
                "metrics": {
                    "em": em, 
                    "f1": f1,
                    "rauq": rauq_score
                },
                "format_compliant": parse_success
            })
        
        # 7. Memory Cleanup (Critical for RAUQ)
        del outputs
        del generated_ids
        torch.cuda.empty_cache()

        # 8. Checkpoint Saving
        if len(results) % SAVE_INTERVAL < BATCH_SIZE and len(results) >= SAVE_INTERVAL:
            print(f" 💾 [Checkpoint] Saving {len(results)} examples...")
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")

    # ==========================================
    # 6. Final Report
    # ==========================================
    
    n = len(dataset)
    avg_em = (total_em / n) * 100
    avg_f1 = (total_f1 / n) * 100
    avg_rauq = (total_rauq / n)
    compliance_rate = ((n - format_errors) / n) * 100

    final_metrics = {
        "model": MODEL_ID,
        "samples": n,
        "exact_match": avg_em,
        "f1_score": avg_f1,
        "avg_rauq": avg_rauq,
        "format_compliance_rate": compliance_rate
    }

    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Format Compliance: {compliance_rate:.2f}%")
    print(f"Exact Match (EM):  {avg_em:.2f}%")
    print(f"F1 Score:          {avg_f1:.2f}%")
    print(f"Avg RAUQ:          {avg_rauq:.4f}")
    print("="*30)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"Detailed logs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()