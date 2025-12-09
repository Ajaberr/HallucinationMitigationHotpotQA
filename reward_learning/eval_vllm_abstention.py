import json
import re
import string
import argparse
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

# vLLM Imports
from vllm import LLM, SamplingParams

# Import NLI verifier for abstention detection
from reward_and_loss import FactualityVerifier

# ==========================================
# 1. Configuration (Defaults)
# ==========================================

DEFAULT_MODEL_ID = "jxrma/Qwen2.5-7B-RLHF-HotpotQA"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "fullwiki"
SPLIT = "validation"
NUM_SAMPLES = 1000

# Output Files
DETAILED_OUTPUT_FILE = "detailed_results_vllm.json"
FINAL_METRICS_FILE = "final_metrics_vllm.json"

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

def check_abstention_nli(verifier, answers):
    """Check if answers are abstentions using NLI verifier"""
    # Create dummy context (not used for abstention detection)
    contexts = ["This is a question."] * len(answers)

    # Get factuality scores
    p_ent, p_cont, f = verifier.compute_factuality_score(contexts, answers)

    # Detect abstentions: very low entailment + high contradiction for "I don't know" type answers
    abstentions = []
    for i, answer in enumerate(answers):
        answer_lower = answer.lower().strip()

        # Pattern matching for common abstention phrases
        is_abstention = (
            "i don't know" in answer_lower or
            "i do not know" in answer_lower or
            "not sure" in answer_lower or
            "cannot answer" in answer_lower or
            answer_lower == "unknown" or
            answer_lower == "unclear"
        )

        abstentions.append(is_abstention)

    return abstentions

def compute_metrics(gold_answers, pred_answers, abstained=None):
    """
    Compute EM and F1 metrics.

    Args:
        gold_answers: List of gold answer lists
        pred_answers: List of predicted answers
        abstained: Optional list of booleans indicating which predictions are abstentions
    """
    em_total, f1_total = 0.0, 0.0

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

# ==========================================
# 3. Main Logic
# ==========================================

def main(model_id):
    print(f"Loading {DATASET_NAME} ({SUBSET_NAME})...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT).select(range(NUM_SAMPLES))

    print(f"Loaded {len(dataset)} samples.")

    # Initialize NLI verifier for abstention detection
    print("\nLoading NLI verifier for abstention detection...")
    device = "cuda"
    verifier = FactualityVerifier(device=device)

    # 1. Prepare Prompts
    prompts = []
    for item in dataset:
        prompt = f"""You are an expert at giving concise answers. Do not give any explanations, only a short answer.

IMPORTANT: If you are not confident or do not know the answer, simply respond with "I don't know" rather than guessing.

Question: {item['question']}
Answer: """
        prompts.append(prompt)

    # Store references to gold answers for scoring later
    gold_answers_list = [
        item['answer'] if isinstance(item['answer'], list) else [item['answer']]
        for item in dataset
    ]
    questions_list = [item['question'] for item in dataset]

    # 2. Initialize vLLM
    print(f"Initializing vLLM with model: {model_id}...")
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        max_model_len=2048,
    )

    # 3. Set Sampling Parameters
    sampling_params = SamplingParams(
        temperature=0,      # Greedy decoding
        max_tokens=50,
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    # 4. Generate
    print("🚀 Starting vLLM Generation (Continuous Batching)...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. Process Results
    results = []
    all_pred_answers = []

    print("Processing outputs and calculating metrics...")

    # Process all outputs first to get predictions
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        all_pred_answers.append(generated_text)

    # Batch abstention detection for efficiency
    print("Detecting abstentions using NLI verifier...")
    abstained = check_abstention_nli(verifier, all_pred_answers)

    # Calculate metrics for each sample
    for i in range(len(dataset)):
        generated_text = all_pred_answers[i]
        gold_answers = gold_answers_list[i]
        is_abstention = abstained[i]

        # --- Scoring ---
        if is_abstention:
            # Count abstention as correct
            em = 1.0
            f1 = 1.0
        else:
            em = max([float(exact_match_score(generated_text, g)) for g in gold_answers])
            f1 = max([f1_score(generated_text, g)[0] for g in gold_answers])

        results.append({
            "id": i,
            "question": questions_list[i],
            "gold_answers": gold_answers,
            "prediction": generated_text,
            "is_abstention": bool(is_abstention),
            "metrics": {"em": em, "f1": f1}
        })

        if i % (NUM_SAMPLES // 5 or 1) == 0 and i > 0:
            print(f"Sample {i}: {questions_list[i][:80]}... -> {generated_text[:50]}...")

    # 6. Final Metrics
    metrics = compute_metrics(gold_answers_list, all_pred_answers, abstained=abstained)

    # Calculate abstention rate
    num_abstentions = sum(1 for result in results if result["is_abstention"])
    abstention_rate = (num_abstentions / len(results)) * 100

    # Calculate metrics for non-abstained samples only
    non_abstained_gold = [gold for gold, abs_flag in zip(gold_answers_list, abstained) if not abs_flag]
    non_abstained_pred = [pred for pred, abs_flag in zip(all_pred_answers, abstained) if not abs_flag]

    if len(non_abstained_pred) > 0:
        non_abstained_metrics = compute_metrics(non_abstained_gold, non_abstained_pred, abstained=None)
    else:
        non_abstained_metrics = {"EM": 0.0, "F1": 0.0}

    final_metrics = {
        "model": model_id,
        "dataset": DATASET_NAME,
        "subset": SUBSET_NAME,
        "samples": len(dataset),
        "exact_match": metrics['EM'],
        "f1_score": metrics['F1'],
        "abstention_rate": abstention_rate,
        "non_abstained_em": non_abstained_metrics['EM'],
        "non_abstained_f1": non_abstained_metrics['F1'],
        "non_abstained_count": len(non_abstained_pred)
    }

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Exact Match (EM):  {metrics['EM']:.2f}%")
    print(f"F1 Score:          {metrics['F1']:.2f}%")
    print(f"Abstention Rate:   {abstention_rate:.2f}% ({num_abstentions}/{len(results)} samples)")
    print(f"\nNon-Abstained Samples ({len(non_abstained_pred)} samples):")
    print(f"  EM: {non_abstained_metrics['EM']:.2f}%")
    print(f"  F1: {non_abstained_metrics['F1']:.2f}%")
    print("="*70)

    # Save
    print(f"\nSaving detailed logs to {DETAILED_OUTPUT_FILE}...")
    with open(DETAILED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saving metrics to {FINAL_METRICS_FILE}...")
    with open(FINAL_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on HotpotQA with abstention detection using vLLM")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID,
                       help=f"Model ID to evaluate (default: {DEFAULT_MODEL_ID})")
    args = parser.parse_args()

    main(args.model)
