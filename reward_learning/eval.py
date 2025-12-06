import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
import string
from collections import Counter
import numpy as np
import json # Added for JSON saving
from reward_and_loss import FactualityVerifier  # Import NLI verifier for abstention detection

# --- Configuration ---
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID = "jxrma/Qwen2.5-7B-RLHF-HotpotQA"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "fullwiki"
SPLIT = "validation"
NUM_SAMPLES = 1000
RAUQ_ALPHA = 0.2
# Output filenames
DETAILED_OUTPUT_FILE = "detailed_results.json"
FINAL_METRICS_FILE = "final_metrics.json"

def load_model_and_tokenizer(model_id):
    """Loads the model and tokenizer, optimizing for available hardware."""
    print(f"Loading model: {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager"
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        
        model.eval()
        print(f"Model loaded successfully on device: {device}")
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# --- RAUQ Implementation ---
def get_rauq_score(sequences, scores, attentions, alpha=0.2):
    if len(scores) == 0:
        return 0.0

    probs = []
    for i, step_logits in enumerate(scores):
        step_probs = F.softmax(step_logits, dim=-1)
        token_id = sequences[0][i] 
        token_prob = step_probs[0, token_id].item()
        probs.append(token_prob)
    
    num_tokens = len(probs)
    num_layers = len(attentions[0])
    
    layer_uncertainties = []

    for layer_idx in range(num_layers):
        # --- Step 1: Head Selection ---
        prev_token_attns = []
        for t in range(1, num_tokens):
            attn_map = attentions[t][layer_idx] # [1, H, 1, K]
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
        current_conf = probs[0]
        confidences.append(current_conf)
        
        for t in range(1, num_tokens):
            prob_curr = probs[t]
            attn_val = attentions[t][layer_idx][0, best_head_idx, 0, -2].item()
            current_conf = alpha * prob_curr + (1 - alpha) * attn_val * current_conf
            confidences.append(current_conf)
            
        # --- Step 3: Sequence Aggregation ---
        log_confs = [torch.log(torch.tensor(c + 1e-9)) for c in confidences]
        layer_u = -torch.mean(torch.stack(log_confs)).item()
        layer_uncertainties.append(layer_u)

    # --- Step 4: Final Score ---
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

def check_abstention_nli(verifier, answers):
    """
    Use NLI verifier to detect if answers are abstentions.
    Returns a tensor of booleans indicating which answers are abstentions.
    """
    abstention_premise = ["I don't know the answer to this question." for _ in answers]
    p_ent_abstain, p_cont_abstain, f_abstain = verifier.compute_factuality_score(
        evidences=abstention_premise,
        answers=answers
    )
    # Model is abstaining if it entails "I don't know" (high p_entailment)
    return (p_ent_abstain > 0.5).cpu().numpy()

def compute_metrics(gold_answers, pred_answers, verifier=None):
    """
    Aggregated metrics calculation with abstention support.
    If verifier is provided and prediction is detected as abstention, count as correct.
    """
    em_total, f1_total = 0.0, 0.0

    # Check for abstentions if verifier provided
    abstained = None
    if verifier is not None:
        abstained = check_abstention_nli(verifier, pred_answers)

    for idx, (gold_list, pred) in enumerate(zip(gold_answers, pred_answers)):
        # If this is an abstention, count as correct
        if abstained is not None and abstained[idx]:
            em_total += 1.0
            f1_total += 1.0
        else:
            # Normal scoring
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

    # Initialize NLI verifier for abstention detection
    print("\nLoading NLI verifier for abstention detection...")
    verifier = FactualityVerifier(device=device)

    print(f"\nLoading HotpotQA dataset (split: {SPLIT}, samples: {NUM_SAMPLES})...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT).select(range(NUM_SAMPLES))

    gold_answers = []
    pred_answers = []
    rauq_scores = []

    # List to store individual sample details
    detailed_results = []

    print("\nStarting closed-book evaluation with RAUQ...")

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        question = example['question']
        gold_answer_list = example['answer'] if isinstance(example['answer'], list) else [example['answer']]
        gold_answers.append(gold_answer_list)

        prompt = f"""You are an expert at giving concise answers. Do not give any explanations, only a short answer.

IMPORTANT: If you are not confident or do not know the answer, simply respond with "I don't know" rather than guessing.

Question: {question}
Answer: """

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
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
        
        # --- Calculate Individual Metrics for this sample ---
        # Check if this is an abstention using NLI verifier
        is_abstention = check_abstention_nli(verifier, [generated_text])[0]

        if is_abstention:
            # Count abstention as correct
            sample_em = 1.0
            sample_f1 = 1.0
        else:
            # Normal scoring: take max score across possible valid answers (HotpotQA format)
            sample_em = max([float(exact_match_score(generated_text, gold)) for gold in gold_answer_list])
            sample_f1 = max([f1_score(generated_text, gold)[0] for gold in gold_answer_list])

        # Prepare data entry
        result_entry = {
            "id": i,
            "question": question,
            "gold_answers": gold_answer_list,
            "prediction": generated_text,
            "is_abstention": bool(is_abstention),
            "metrics": {
                "rauq_score": float(rauq), # cast to native float for JSON serialization
                "exact_match": sample_em,
                "f1_score": sample_f1
            }
        }
        detailed_results.append(result_entry)

        if i % (NUM_SAMPLES // 5 or 1) == 0 and i > 0:
            print(f"\nSample {i}: {question} -> {generated_text} | RAUQ: {rauq:.4f}")

    # --- Final Computations ---
    metrics = compute_metrics(gold_answers, pred_answers, verifier=verifier)
    avg_rauq = sum(rauq_scores) / len(rauq_scores)

    # Calculate abstention rate
    num_abstentions = sum(1 for result in detailed_results if result["is_abstention"])
    abstention_rate = (num_abstentions / len(detailed_results)) * 100

    final_results = {
        "model": MODEL_ID,
        "dataset": DATASET_NAME,
        "samples": NUM_SAMPLES,
        "EM": metrics['EM'],
        "F1": metrics['F1'],
        "Avg_RAUQ": avg_rauq,
        "Abstention_Rate": abstention_rate
    }

    print(f"\nResults for {MODEL_ID} on HotpotQA:")
    print(f"  Exact Match (EM): {metrics['EM']:.2f}%")
    print(f"  F1 Score (F1): {metrics['F1']:.2f}%")
    print(f"  Avg RAUQ Uncertainty: {avg_rauq:.4f}")
    print(f"  Abstention Rate: {abstention_rate:.2f}% ({num_abstentions}/{len(detailed_results)} samples)")

    # --- Save Results to JSON ---
    print(f"Saving detailed results to {DETAILED_OUTPUT_FILE}...")
    with open(DETAILED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
    print(f"Saving final metrics to {FINAL_METRICS_FILE}...")
    with open(FINAL_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    main()