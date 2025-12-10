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

# --- Configuration ---
# Newly trained CoT Adapter
ADAPTER_PATH = "tinker://a13a33af-9a99-51bf-8447-722ab7aef0d4:train:0/sampler_weights/qwen-cot-distilled"
BASE_MODEL = "Qwen/Qwen3-8B"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation"  
BATCH_SIZE = 32
# Short Eval for quick feedback
NUM_SAMPLES = 100 

DETAILED_OUTPUT_FILE = "tinker_cot_results.json"
FINAL_METRICS_FILE = "tinker_cot_metrics.json"

# CoT Prompts (MUST MATCH distill_cot.py)
SYSTEM_PROMPT = "You are an expert Professor of Knowledge Graph Reasoning."
USER_PROMPT_TEMPLATE = "Answer the following question with a step-by-step reasoning trace.\n\nQuestion: {question}"

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
    # Avoid division by zero if empty
    if not gold_answers: return {"EM": 0, "F1": 0}
    return {
        "EM": (em_total / len(gold_answers)) * 100,
        "F1": (f1_total / len(gold_answers)) * 100,
    }

def main():
    print("🚀 Connecting to Tinker Service for CoT Evaluation...")
    if not os.environ.get("TINKER_API_KEY"):
        print("⚠️ TINKER_API_KEY not found. Check .env file.")
        return

    service = tinker.ServiceClient()

    print(f"Initializing Sampling Client for {ADAPTER_PATH}...")
    sampling_client = service.create_sampling_client(
        model_path=ADAPTER_PATH
    )

    print("Getting tokenizer...")
    training_client = service.create_lora_training_client(base_model=BASE_MODEL, rank=16)
    tokenizer = training_client.get_tokenizer()

    print(f"\nLoading HotpotQA dataset (split: {SPLIT})...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    
    # LIMIT FOR SHORT EVAL
    print(f"Selecting first {NUM_SAMPLES} samples for quick verification...")
    dataset = dataset#.select(range(NUM_SAMPLES))
    total_samples = len(dataset)

    all_gold_answers = []
    all_pred_answers = []
    detailed_results = []

    print(f"\nStarting evaluation (Batch Size: {BATCH_SIZE})...")

    def chunked(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    def process_single_sample(prompt_text):
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        model_input = ttypes.ModelInput.from_ints(input_ids)
        # CoT needs more tokens for reasoning
        sampling_params = ttypes.SamplingParams(max_tokens=256, temperature=0.1)
        
        try:
            result = sampling_client.sample(model_input, 1, sampling_params).result()
            output_ids = result.sequences[0].tokens
            output_text = tokenizer.decode(output_ids)
            
            # 1. Strip prompt if present
            if output_text.startswith(prompt_text):
                output_text = output_text[len(prompt_text):]
            if "assistant\n" in output_text:
                 output_text = output_text.split("assistant\n")[-1]

            # 2. Strip <|im_end|>
            output_text = output_text.replace("<|im_end|>", "")
            output_text = output_text.strip()
            
            # 3. PARSE ANSWER FROM TRACE
            # Observed format: "Step ... Therefore, the answer is <Answer>."
            # Also support the "-->" format as a fallback.
            
            parsed_answer = ""
            
            # Pattern 1: "Therefore, the answer is"
            if "Therefore, the answer is" in output_text:
                parsed_answer = output_text.split("Therefore, the answer is")[-1].strip()
            
            # Pattern 2: VM Delimiter "-->"
            elif " --> " in output_text:
                parsed_answer = output_text.split(" --> ")[1].strip()
            
            # Cleanup trailing period if present (common in sentence-ending answers)
            if parsed_answer and parsed_answer.endswith("."):
                parsed_answer = parsed_answer[:-1]
                
            if parsed_answer:
                return parsed_answer
            
            # If parsing failed, we return empty string to avoid false positives on the whole trace
            # or return the whole trace (though that guarantees 0 EM usually)
            return ""
            
        except Exception as e:
            # print(f"Error sampling: {e}")
            output_text = ""
        return output_text

    for batch in tqdm(chunked(dataset, BATCH_SIZE), total=total_samples // BATCH_SIZE if total_samples >= BATCH_SIZE else 1):
        questions = batch['question']
        gold_answers = [ans if isinstance(ans, list) else [ans] for ans in batch['answer']]
        ids = batch.get('id', list(range(len(questions))))

        # 1. Format Outputs using CoT Prompt
        prompts = []
        for q in questions:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=q)}
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
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

    metrics = compute_metrics(all_gold_answers, all_pred_answers)

    print(f"\nResults for {ADAPTER_PATH} on HotpotQA (First {NUM_SAMPLES} samples):")
    print(f"  Exact Match (EM): {metrics['EM']:.2f}% (Expect low due to reasoning trace)")
    print(f"  F1 Score (F1): {metrics['F1']:.2f}%")

    print(f"Saving detailed results to {DETAILED_OUTPUT_FILE}...")
    with open(DETAILED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    print(f"Saving final metrics to {FINAL_METRICS_FILE}...")
    with open(FINAL_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "model": ADAPTER_PATH}, f, indent=2)

if __name__ == "__main__":
    main()
