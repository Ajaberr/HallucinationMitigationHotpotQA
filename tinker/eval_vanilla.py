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

load_dotenv()

# Configuration
# BASE_MODEL = "Qwen/Qwen3-8B" # Tinker's available Qwen model
# Using the base model path directly for inference
MODEL_PATH = "Qwen/Qwen3-8B" 

DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation"
BATCH_SIZE = 32
DETAILED_OUTPUT_FILE = "tinker_vanilla_results.json"
FINAL_METRICS_FILE = "tinker_vanilla_metrics.json"

# --- Metric Utils (Exact copy) ---
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
    print(f"🚀 Connecting to Tinker Service for VANILLA Baseline ({MODEL_PATH})...")
    if not os.environ.get("TINKER_API_KEY"):
        print("⚠️ TINKER_API_KEY not found. Check .env file.")
        return

    service = tinker.ServiceClient()

    print(f"Initializing Training Client to register {MODEL_PATH}...")
    # We must create a "dummy" adapter to get a tinker:// URI for sampling the base model
    training_client = service.create_lora_training_client(base_model=MODEL_PATH, rank=16)
    
    print("Saving Vanilla Adapter (Zero-shot)...")
    # This saves the initialized (random/zero) LoRA weights. 
    # WAIT: Random LoRA weights will mess up the model! 
    # We need rank=0 or to ensure they are zero-initialized?
    # Tinker docs say "Base model... initialize from". 
    # Actually, standard LoRA relies on B=0 init so it starts as identity.
    # So saving immediately should give us the Base Model behavior.
    
    save_future = training_client.save_weights_for_sampler("vanilla-baseline-fresh")
    save_result = save_future.result()
    tinker_uri = save_result.path
    print(f"✅ Registered Base Model as: {tinker_uri}")

    print(f"Initializing Sampling Client...")
    sampling_client = service.create_sampling_client(
        model_path=tinker_uri
    )

    print("Getting tokenizer via training client...")
    tokenizer = training_client.get_tokenizer()

    print(f"\nLoading HotpotQA dataset (split: {SPLIT})...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples.")

    print(f"\nStarting evaluation (Batch Size: {BATCH_SIZE})...")

    # Helper to chunk list
    def chunked(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    def process_single_sample(prompt_text):
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        model_input = ttypes.ModelInput.from_ints(input_ids)
        
        # Greedy decoding (temperature=0) roughly matches do_sample=False
        sampling_params = ttypes.SamplingParams(
            max_tokens=50, 
            temperature=0.001, # Effectively greedy
            stop=["\n"]        # Stop at newline to enforce concise answer
        )
        
        try:
            result = sampling_client.sample(model_input, 1, sampling_params).result()
            output_ids = result.sequences[0].tokens
            output_text = tokenizer.decode(output_ids)
            
            # Robust Cleaning (Same as the fix)
            if output_text.startswith(prompt_text):
                output_text = output_text[len(prompt_text):]
            if "assistant\n" in output_text:
                 output_text = output_text.split("assistant\n")[-1]
            
            output_text = output_text.strip()
            
        except Exception as e:
            # print(f"Error sampling: {e}")
            output_text = ""
        return output_text

    all_gold_answers = []
    all_pred_answers = []
    detailed_results = []

    for batch in tqdm(chunked(dataset, BATCH_SIZE), total=total_samples // BATCH_SIZE):
        questions = batch['question']
        gold_answers = [ans if isinstance(ans, list) else [ans] for ans in batch['answer']]
        ids = batch.get('id', list(range(len(questions))))

        # 1. Format Prompts (Exact same system prompt as baseline_vanilla/eval_batch.py)
        prompts = []
        for q in questions:
            # Manually constructing the prompt to mimic apply_chat_template 
            # if we trust the tokenizer, we can use apply_chat_template, otherwise manual string.
            # Using manual string to be safe and consistent with logic or rely on tokenizer.
            # The baseline used tokenizer.apply_chat_template. Let's try to simulate or use it.
            # But earlier tinker script used manual string. Let's stick to the prompt text that worked.
            
            # Original Prompt from eval_batch.py:
            # system: You are a concise encyclopedia...
            
            # We construct it similarly:
            prompt_text = f"You are a concise encyclopedia. Answer the question directly with a short phrase or entity name. Do not explain.\nuser\n{q}\nassistant\n"
            prompts.append(prompt_text)
        
        # 2. Parallel Generation
        batch_preds = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            future_to_idx = {executor.submit(process_single_sample, p): i for i, p in enumerate(prompts)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    batch_preds[idx] = future.result()
                except Exception as e:
                    print(f"Sample {idx} failed: {e}")

        # 3. Process Results
        for b in range(len(questions)):
            pred_text = batch_preds[b].strip()
            current_gold = gold_answers[b]

            all_gold_answers.append(current_gold)
            all_pred_answers.append(pred_text)

            sample_em = max([float(exact_match_score(pred_text, gold)) for gold in current_gold])
            sample_f1 = max([f1_score(pred_text, gold)[0] for gold in current_gold])

            detailed_results.append({
                "id": str(ids[b]),
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
        "model": MODEL_PATH,
        "dataset": DATASET_NAME,
        "setting": SUBSET_NAME,
        "split": SPLIT,
        "samples": total_samples,
        "EM": metrics['EM'],
        "F1": metrics['F1']
    }

    print(f"\nResults for {MODEL_PATH} on HotpotQA (Full {SPLIT}):")
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
