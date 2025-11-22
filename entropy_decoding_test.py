import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
import string
from collections import Counter
import numpy as np

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "fullwiki"
SPLIT = "validation"
NUM_SAMPLES = 100
RAUQ_ALPHA = 0.2  # Recommended alpha for QA tasks from the paper

def load_model_and_tokenizer(model_id):
    """Loads the model and tokenizer, optimizing for available hardware."""
    print(f"Loading model: {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    kwargs = {
        # Fix deprecation warning
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        # Force eager attention so output_attentions=True works
        "attn_implementation": "eager"
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        
        # We removed the print statement that caused the crash here.
        
        model.eval()
        print(f"Model loaded successfully on device: {device}")
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# --- RAUQ Implementation ---
def get_rauq_score(sequences, scores, attentions, alpha=0.2):
    """
    Computes the RAUQ uncertainty score for a generated sequence.
    
    Args:
        sequences: Tensor of generated token IDs (shape: [1, seq_len])
        scores: Tuple of tensors (logits) for each generation step
        attentions: Tuple (per step) of Tuple (per layer) of tensors (batch, head, 1, past_key_len)
        alpha: Hyperparameter balancing probability and attention (default 0.2)
    
    Returns:
        float: The RAUQ uncertainty score (higher = more uncertain)
    """
    # We assume batch_size=1 for this implementation loop
    if len(scores) == 0:
        return 0.0

    # 1. Extract Probabilities for the generated tokens
    probs = []
    for i, step_logits in enumerate(scores):
        step_probs = F.softmax(step_logits, dim=-1)
        # Get prob of the actual token generated at this step
        token_id = sequences[0][i] 
        token_prob = step_probs[0, token_id].item()
        probs.append(token_prob)
    
    num_tokens = len(probs)
    num_layers = len(attentions[0])
    num_heads = attentions[0][0].shape[1]
    
    layer_uncertainties = []

    # Iterate through each layer to calculate layer-wise uncertainty
    for layer_idx in range(num_layers):
        # --- Step 1: Head Selection (Eq. 1) ---
        # Find the "uncertainty-aware" head: the one with max average attention to the previous token
        head_avg_attentions = []
        
        # Collect attention to previous token (a_{i, i-1}) for all heads
        # Note: We start from 2nd token (index 1) because 1st token has no generated predecessor
        prev_token_attns = []
        
        for t in range(1, num_tokens):
            # attentions[step][layer] shape: (batch, heads, query_len=1, key_len)
            # The previous token is at index -2 in the key dimension
            attn_map = attentions[t][layer_idx] # [1, H, 1, K]
            attn_to_prev = attn_map[0, :, 0, -2] # [H] - Get attention to previous token for all heads
            prev_token_attns.append(attn_to_prev)
            
        if not prev_token_attns:
            layer_uncertainties.append(0.0)
            continue
            
        # Stack to [num_tokens-1, num_heads]
        prev_token_attns = torch.stack(prev_token_attns) 
        
        # Calculate mean over the sequence for each head
        mean_head_attn = torch.mean(prev_token_attns, dim=0) # [H]
        
        # Select the best head
        best_head_idx = torch.argmax(mean_head_attn).item()
        
        # --- Step 2: Recurrent Confidence (Eq. 2) ---
        confidences = []
        
        # Initialize c_l(y_1) = P(y_1|x)
        current_conf = probs[0]
        confidences.append(current_conf)
        
        for t in range(1, num_tokens):
            prob_curr = probs[t]
            # Get attention from selected head to previous token
            attn_val = attentions[t][layer_idx][0, best_head_idx, 0, -2].item()
            
            # Apply Recurrence: alpha * P(curr) + (1-alpha) * attn * prev_conf
            current_conf = alpha * prob_curr + (1 - alpha) * attn_val * current_conf
            confidences.append(current_conf)
            
        # --- Step 3: Sequence Aggregation (Eq. 3) ---
        # u_l(y) = -1/N * sum(log(c_l(y_i)))
        # Add epsilon to avoid log(0)
        log_confs = [torch.log(torch.tensor(c + 1e-9)) for c in confidences]
        layer_u = -torch.mean(torch.stack(log_confs)).item()
        layer_uncertainties.append(layer_u)

    # --- Step 4: Final Score (Eq. 4) ---
    # u(y) = max_{l} u_l(y)
    # The paper suggests taking the max over layers
    final_rauq_score = max(layer_uncertainties)
    
    return final_rauq_score

# --- Metric Utils (Unchanged) ---
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
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT).select(range(NUM_SAMPLES))
    
    gold_answers = []
    pred_answers = []
    rauq_scores = []

    print("\nStarting closed-book evaluation with RAUQ...")

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        question = example['question']
        gold_answer_list = example['answer'] if isinstance(example['answer'], list) else [example['answer']]
        gold_answers.append(gold_answer_list)

        prompt = f"""You are an expert at giving concise answers. Do not give any explanations, only a short answer.
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
                # --- REQUIRED FOR RAUQ ---
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Extract generated sequence (remove prompt)
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

        if i % (NUM_SAMPLES // 5 or 1) == 0 and i > 0:
            print(f"\nSample {i}: {question} -> {generated_text} | RAUQ: {rauq:.4f}")

    metrics = compute_metrics(gold_answers, pred_answers)
    avg_rauq = sum(rauq_scores) / len(rauq_scores)

    print(f"\nResults for {MODEL_ID} on HotpotQA:")
    print(f"  Exact Match (EM): {metrics['EM']:.2f}%")
    print(f"  F1 Score (F1): {metrics['F1']:.2f}%")
    print(f"  Avg RAUQ Uncertainty: {avg_rauq:.4f}")

if __name__ == "__main__":
    main()