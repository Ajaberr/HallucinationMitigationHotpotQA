"""
Unified HotpotQA Evaluation Script
===================================
Evaluates both baseline (vanilla) models and RLHF-trained models on HotpotQA.

Supports:
- Baseline evaluation: Pre-trained models without RLHF
- RLHF evaluation: Base RLHF and custom anti-hallucination models
- Comparison mode: Side-by-side comparison

Usage Examples:
    # Baseline evaluation (vanilla model)
    python evaluate_models.py --mode baseline --model Qwen/Qwen2.5-7B-Instruct --num_samples 100

    # RLHF evaluation (base)
    python evaluate_models.py --mode rlhf --model_type base --policy_path ./rlhf_policy --reward_path ./reward.pt

    # RLHF evaluation (custom anti-hallucination)
    python evaluate_models.py --mode rlhf --model_type custom --policy_path ./rlhf_policy --reward_path ./reward.pt

    # Compare base vs custom RLHF
    python evaluate_models.py --mode compare --policy_path ./base_policy --custom_policy_path ./custom_policy

    # Compare baseline vs RLHF
    python evaluate_models.py --mode compare_baseline --baseline_model Qwen/Qwen2.5-7B-Instruct --policy_path ./rlhf_policy
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
import string
from collections import Counter
from typing import List, Dict, Tuple, Optional
import json

# Try importing RLHF modules
try:
    from base_simple_reward import (
        RewardModelConfig,
        SimpleRLConfig,
        RewardModelTrainer,
        SimpleRLTrainer,
        RewardModel,
    )
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False

try:
    from custom_simple_reward import (
        RewardModelConfig as CustomRewardModelConfig,
        SimpleRLConfig as CustomSimpleRLConfig,
        RewardModelTrainer as CustomRewardModelTrainer,
        SimpleRLTrainer as CustomSimpleRLTrainer,
        RewardModel as CustomRewardModel,
    )
    CUSTOM_AVAILABLE = True
except ImportError:
    CUSTOM_AVAILABLE = False


# ============================================================================
# Official HotpotQA Metrics (shared by all evaluation modes)
# ============================================================================

def normalize_answer(s):
    """
    Official HotpotQA normalization for answers.
    Lowercases, removes punctuation, articles (a, an, the), and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Computes F1 score based on token overlap (official HotpotQA version)."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0.0, 0.0, 0.0)  # F1, Precision, Recall

    # Handling 'yes', 'no', 'noanswer' as non-overlapping tokenization causes issues.
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    # Token matching using Counter
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    """Computes Exact Match score (official HotpotQA version)."""
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_metrics(gold_answers, pred_answers):
    """Calculates Exact Match (EM), F1 Score, Precision, and Recall for a list of answers."""
    em_total = 0.0
    f1_total = 0.0
    prec_total = 0.0
    recall_total = 0.0

    for gold_list, pred in zip(gold_answers, pred_answers):
        # HotpotQA official metric uses the MAX score across all gold answers
        best_em = 0.0
        best_f1 = 0.0
        best_prec = 0.0
        best_recall = 0.0

        for gold_text in gold_list:
            # 1. EM Score
            em = exact_match_score(pred, gold_text)
            best_em = max(best_em, float(em))

            # 2. F1 Score (Returns F1, Precision, Recall tuple)
            f1, prec, recall = f1_score(pred, gold_text)

            if f1 > best_f1:
                best_f1 = f1
                best_prec = prec
                best_recall = recall

        em_total += best_em
        f1_total += best_f1
        prec_total += best_prec
        recall_total += best_recall

    num_samples = len(gold_answers)
    return {
        "EM": (em_total / num_samples) * 100,
        "F1": (f1_total / num_samples) * 100,
        "Precision": (prec_total / num_samples) * 100,
        "Recall": (recall_total / num_samples) * 100,
    }


# ============================================================================
# Dataset Loading (shared)
# ============================================================================

def load_hotpotqa_data(split: str = "validation", num_samples: int = 100) -> Tuple[List[str], List[List[str]]]:
    """
    Load HotpotQA dataset and extract questions and answers.

    Args:
        split: Dataset split to use ('train', 'validation', 'test')
        num_samples: Number of samples to load

    Returns:
        Tuple of (questions, gold_answers) where gold_answers is a list of lists
    """
    print(f"\nLoading HotpotQA dataset (split: {split}, samples: {num_samples})...")

    try:
        dataset = load_dataset("hotpot_qa", "fullwiki", split=split)
        dataset = dataset.select(range(min(num_samples, len(dataset))))

        questions = []
        gold_answers = []

        for example in dataset:
            question = example['question']
            questions.append(question)

            # Ensure answer is in list format
            answer = example['answer']
            if isinstance(answer, list):
                gold_answers.append(answer)
            else:
                gold_answers.append([answer])

        print(f"Loaded {len(questions)} questions from HotpotQA")
        return questions, gold_answers

    except Exception as e:
        print(f"Error loading HotpotQA dataset: {e}")
        print("Make sure you have the datasets library installed: pip install datasets")
        return [], []


# ============================================================================
# Baseline Evaluation (Vanilla Pre-trained Models)
# ============================================================================

def load_baseline_model(model_id: str):
    """Loads the baseline model and tokenizer, optimizing for available hardware."""
    print(f"\nLoading baseline model: {model_id}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True
    }

    # Try using 4-bit quantization if GPU is available to save VRAM
    if device == "cuda" and torch.cuda.get_device_properties(0).total_memory / (1024**3) < 24:
        print("Warning: Low VRAM detected. Attempting to load with 4-bit quantization.")
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["quantization_config"] = bnb_config
        except ImportError:
            print("Install 'bitsandbytes' for 4-bit quantization to work on low VRAM.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.eval()
        print(f"Model loaded successfully on device: {device}")
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have sufficient hardware (GPU/RAM) and required libraries installed.")
        return None, None, None


def evaluate_baseline(
    model_id: str,
    questions: List[str] = None,
    gold_answers: List[List[str]] = None,
    num_samples: int = 100,
    split: str = "validation",
    verbose: bool = True
) -> Dict:
    """
    Evaluate a baseline (vanilla) model on HotpotQA.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B-Instruct")
        questions: Pre-loaded questions (optional)
        gold_answers: Pre-loaded gold answers (optional)
        num_samples: Number of samples to evaluate
        split: Dataset split to use
        verbose: Print sample predictions

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*70)
    print("BASELINE EVALUATION (Vanilla Pre-trained Model)")
    print("="*70)

    # Load model
    tokenizer, model, device = load_baseline_model(model_id)
    if model is None:
        return {}

    # Load data if not provided
    if questions is None or gold_answers is None:
        questions, gold_answers = load_hotpotqa_data(split=split, num_samples=num_samples)
        if not questions:
            return {}

    pred_answers = []

    print("\nStarting closed-book evaluation...")

    for i, (question, gold_list) in tqdm(enumerate(zip(questions, gold_answers)), total=len(questions)):
        # Closed-book prompting (no context provided)
        prompt = f"""You are an expert at giving concise answers. Do not give any explanations, only a short answer. For example:
Question: Which magazine was started first Arthur's Magazine or First for Women?
Answer: Arthur's Magazine

Question: Is Children's National Medical Center or MedStar Washington Hospital Center the largest private hospital in Washington, D.C.?
Answer: MedStar Washington Hospital Center

Now answer the question:

Question: {question}
Answer: """

        # Apply chat template if model supports it
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            # Fallback if chat template not available
            text = prompt

        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the generated text, skipping the input prompt
        generated_text = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        pred_answers.append(generated_text)

        # Print samples
        if verbose and i % (len(questions) // 5 or 1) == 0 and i > 0:
            print(f"\n--- Sample {i+1}/{len(questions)} ---")
            print(f"Q: {question}")
            print(f"Gold: {gold_list[0]}")
            print(f"Pred: {generated_text}")

    # Compute metrics
    print("\n--- Evaluation Complete ---")
    metrics = compute_metrics(gold_answers, pred_answers)

    print(f"\nResults for {model_id} on HotpotQA ({len(questions)} samples):")
    print(f"  Exact Match (EM): {metrics['EM']:.2f}%")
    print(f"  F1 Score (F1): {metrics['F1']:.2f}%")
    print(f"  Precision: {metrics['Precision']:.2f}%")
    print(f"  Recall: {metrics['Recall']:.2f}%")

    return metrics


# ============================================================================
# RLHF Evaluation (Base and Custom Models)
# ============================================================================

def evaluate_rlhf_base(
    policy_path: str,
    reward_path: str = None,
    questions: List[str] = None,
    gold_answers: List[List[str]] = None,
    num_samples: int = 100
) -> Dict:
    """
    Evaluate the base RLHF model on HotpotQA.

    Args:
        policy_path: Path to trained policy model
        reward_path: Path to trained reward model (optional)
        questions: Pre-loaded questions (optional)
        gold_answers: Pre-loaded gold answers (optional)
        num_samples: Number of samples to evaluate if loading from scratch

    Returns:
        Dictionary with evaluation metrics
    """
    if not BASE_AVAILABLE:
        print("Error: base_simple_reward.py is not available")
        return {}

    print("\n" + "="*70)
    print("RLHF EVALUATION - Base Model")
    print("="*70)

    # Load data if not provided
    if questions is None or gold_answers is None:
        questions, gold_answers = load_hotpotqa_data(num_samples=num_samples)
        if not questions:
            return {}

    # Load policy model
    print(f"\nLoading policy from {policy_path}...")
    rl_config = SimpleRLConfig(
        policy_model_name=policy_path,
        batch_size=1
    )

    dummy_reward = RewardModel("Qwen/Qwen2-7B")
    rl_trainer = SimpleRLTrainer(rl_config, dummy_reward)

    # Load reward model if provided
    reward_trainer = None
    if reward_path:
        print(f"Loading reward model from {reward_path}...")
        rm_config = RewardModelConfig()
        reward_trainer = RewardModelTrainer(rm_config)
        reward_trainer.load_model(reward_path)

    # Evaluate
    metrics = rl_trainer.evaluate_on_hotpotqa(
        questions=questions,
        gold_answers=gold_answers,
        reward_model_trainer=reward_trainer
    )

    return metrics


def evaluate_rlhf_custom(
    policy_path: str,
    reward_path: str = None,
    questions: List[str] = None,
    gold_answers: List[List[str]] = None,
    num_samples: int = 100
) -> Dict:
    """
    Evaluate the custom anti-hallucination RLHF model on HotpotQA.

    Args:
        policy_path: Path to trained policy model
        reward_path: Path to trained reward model (optional)
        questions: Pre-loaded questions (optional)
        gold_answers: Pre-loaded gold answers (optional)
        num_samples: Number of samples to evaluate if loading from scratch

    Returns:
        Dictionary with evaluation metrics including hallucination scores
    """
    if not CUSTOM_AVAILABLE:
        print("Error: custom_simple_reward.py is not available")
        return {}

    print("\n" + "="*70)
    print("RLHF EVALUATION - Custom Anti-Hallucination Model")
    print("="*70)

    # Load data if not provided
    if questions is None or gold_answers is None:
        questions, gold_answers = load_hotpotqa_data(num_samples=num_samples)
        if not questions:
            return {}

    # Load policy model
    print(f"\nLoading policy from {policy_path}...")
    rl_config = CustomSimpleRLConfig(
        policy_model_name=policy_path,
        batch_size=1
    )

    dummy_reward = CustomRewardModel("Qwen/Qwen2-7B")
    rl_trainer = CustomSimpleRLTrainer(rl_config, dummy_reward)

    # Load reward model if provided
    reward_trainer = None
    if reward_path:
        print(f"Loading reward model from {reward_path}...")
        rm_config = CustomRewardModelConfig()
        reward_trainer = CustomRewardModelTrainer(rm_config)
        reward_trainer.load_model(reward_path)

    # Evaluate
    metrics = rl_trainer.evaluate_on_hotpotqa(
        questions=questions,
        gold_answers=gold_answers,
        reward_model_trainer=reward_trainer,
        return_detailed_metrics=False
    )

    return metrics


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_rlhf_models(
    base_policy_path: str,
    custom_policy_path: str,
    base_reward_path: str = None,
    custom_reward_path: str = None,
    num_samples: int = 100
) -> Dict:
    """
    Compare base and custom RLHF models side-by-side on HotpotQA.

    Args:
        base_policy_path: Path to base model policy
        custom_policy_path: Path to custom model policy
        base_reward_path: Path to base reward model (optional)
        custom_reward_path: Path to custom reward model (optional)
        num_samples: Number of samples to evaluate

    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*70)
    print("COMPARING BASE vs CUSTOM RLHF MODELS")
    print("="*70)

    # Load data once
    questions, gold_answers = load_hotpotqa_data(num_samples=num_samples)
    if not questions:
        return {}

    # Evaluate both models
    base_metrics = evaluate_rlhf_base(
        base_policy_path,
        base_reward_path,
        questions,
        gold_answers
    )

    custom_metrics = evaluate_rlhf_custom(
        custom_policy_path,
        custom_reward_path,
        questions,
        gold_answers
    )

    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    comparison = {
        'base_rlhf': base_metrics,
        'custom_rlhf': custom_metrics
    }

    print("\nMetric                     | Base RLHF  | Custom RLHF | Improvement")
    print("-" * 70)

    for metric in ['EM', 'F1', 'Precision', 'Recall']:
        base_val = base_metrics.get(metric, 0.0)
        custom_val = custom_metrics.get(metric, 0.0)
        diff = custom_val - base_val
        print(f"{metric:25} | {base_val:10.2f} | {custom_val:12.2f} | {diff:+11.2f}")

    if 'Avg_Reward' in base_metrics or 'Avg_Base_Reward' in custom_metrics:
        print("\nReward Metrics:")
        base_reward = base_metrics.get('Avg_Reward', 0.0)
        custom_reward = custom_metrics.get('Avg_Base_Reward', 0.0)
        print(f"{'Reward':25} | {base_reward:10.4f} | {custom_reward:12.4f}")

    if 'Avg_Halluc_Score' in custom_metrics:
        print(f"\nCustom Model Hallucination Score: {custom_metrics['Avg_Halluc_Score']:.4f}")
        print(f"Custom Model Abstention Rate: {custom_metrics.get('Abstention_Rate', 0.0):.2f}%")

    return comparison


def compare_baseline_vs_rlhf(
    baseline_model: str,
    policy_path: str,
    reward_path: str = None,
    model_type: str = "base",
    num_samples: int = 100
) -> Dict:
    """
    Compare baseline vanilla model vs RLHF-trained model.

    Args:
        baseline_model: HuggingFace model ID for baseline
        policy_path: Path to RLHF policy
        reward_path: Path to reward model (optional)
        model_type: "base" or "custom"
        num_samples: Number of samples

    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*70)
    print("COMPARING BASELINE vs RLHF")
    print("="*70)

    # Load data once
    questions, gold_answers = load_hotpotqa_data(num_samples=num_samples)
    if not questions:
        return {}

    # Evaluate baseline
    baseline_metrics = evaluate_baseline(
        baseline_model,
        questions,
        gold_answers,
        verbose=False
    )

    # Evaluate RLHF
    if model_type == "base":
        rlhf_metrics = evaluate_rlhf_base(
            policy_path,
            reward_path,
            questions,
            gold_answers
        )
    else:
        rlhf_metrics = evaluate_rlhf_custom(
            policy_path,
            reward_path,
            questions,
            gold_answers
        )

    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    comparison = {
        'baseline': baseline_metrics,
        'rlhf': rlhf_metrics
    }

    print("\nMetric                     | Baseline   | RLHF        | Improvement")
    print("-" * 70)

    for metric in ['EM', 'F1', 'Precision', 'Recall']:
        base_val = baseline_metrics.get(metric, 0.0)
        rlhf_val = rlhf_metrics.get(metric, 0.0)
        diff = rlhf_val - base_val
        print(f"{metric:25} | {base_val:10.2f} | {rlhf_val:12.2f} | {diff:+11.2f}")

    return comparison


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for baseline and RLHF models on HotpotQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection (defaults to compare_baseline if policy_path is provided, otherwise baseline)
    parser.add_argument(
        '--mode',
        type=str,
        choices=['baseline', 'rlhf', 'compare', 'compare_baseline', 'all'],
        default=None,
        help='Evaluation mode: baseline (vanilla), rlhf (trained), compare (RLHF models), compare_baseline (baseline vs RLHF), all (evaluate all 3: baseline + base RL + anti-halluc RL). Default: auto-detect based on arguments'
    )

    # Baseline model
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Model ID for baseline evaluation'
    )

    # RLHF paths
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['base', 'custom'],
        default='base',
        help='Type of RLHF model (base or custom anti-hallucination)'
    )
    parser.add_argument(
        '--policy_path',
        type=str,
        help='Path to trained policy model (for RLHF modes)'
    )
    parser.add_argument(
        '--reward_path',
        type=str,
        default=None,
        help='Path to trained reward model (optional)'
    )
    parser.add_argument(
        '--custom_policy_path',
        type=str,
        default=None,
        help='Path to custom policy model (for compare mode)'
    )
    parser.add_argument(
        '--custom_reward_path',
        type=str,
        default=None,
        help='Path to custom reward model (for compare mode)'
    )

    # Dataset options
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to evaluate'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='validation',
        choices=['train', 'validation', 'test'],
        help='Dataset split to use'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results JSON (optional)'
    )

    args = parser.parse_args()

    # Route to appropriate evaluation function
    results = {}

    if args.mode == 'baseline':
        results = evaluate_baseline(
            args.model,
            num_samples=args.num_samples,
            split=args.split
        )

    elif args.mode == 'rlhf':
        if not args.policy_path:
            print("Error: --policy_path required for RLHF mode")
            return

        if args.model_type == 'base':
            results = evaluate_rlhf_base(
                args.policy_path,
                args.reward_path,
                num_samples=args.num_samples
            )
        else:
            results = evaluate_rlhf_custom(
                args.policy_path,
                args.reward_path,
                num_samples=args.num_samples
            )

    elif args.mode == 'compare':
        if not args.policy_path or not args.custom_policy_path:
            print("Error: --policy_path and --custom_policy_path required for compare mode")
            return

        results = compare_rlhf_models(
            args.policy_path,
            args.custom_policy_path,
            args.reward_path,
            args.custom_reward_path,
            num_samples=args.num_samples
        )

    elif args.mode == 'compare_baseline':
        if not args.policy_path:
            print("Error: --policy_path required for compare_baseline mode")
            return

        results = compare_baseline_vs_rlhf(
            args.model,
            args.policy_path,
            args.reward_path,
            args.model_type,
            num_samples=args.num_samples
        )

    elif args.mode == 'all':
        # Evaluate all 3 models: baseline, base RL, anti-hallucination RL
        print("\n" + "="*70)
        print("EVALUATING ALL MODELS: Baseline + Base RL + Anti-Hallucination RL")
        print("="*70)

        # Load data once
        questions, gold_answers = load_hotpotqa_data(split=args.split, num_samples=args.num_samples)
        if not questions:
            return

        # 1. Baseline (no RL)
        print("\n" + "="*70)
        print("1/3: BASELINE MODEL (No RLHF)")
        print("="*70)
        baseline_results = evaluate_baseline(
            args.model,
            questions=questions,
            gold_answers=gold_answers,
            verbose=False
        )

        # 2. Base RL
        base_rl_results = {}
        if args.policy_path and BASE_AVAILABLE:
            print("\n" + "="*70)
            print("2/3: BASE RLHF MODEL")
            print("="*70)
            base_rl_results = evaluate_rlhf_base(
                args.policy_path,
                args.reward_path,
                questions=questions,
                gold_answers=gold_answers
            )
        else:
            print("\n[Skipping Base RL - no policy_path provided or base_simple_reward.py unavailable]")

        # 3. Anti-Hallucination RL
        custom_rl_results = {}
        if args.custom_policy_path and CUSTOM_AVAILABLE:
            print("\n" + "="*70)
            print("3/3: ANTI-HALLUCINATION RLHF MODEL")
            print("="*70)
            custom_rl_results = evaluate_rlhf_custom(
                args.custom_policy_path,
                args.custom_reward_path,
                questions=questions,
                gold_answers=gold_answers
            )
        else:
            print("\n[Skipping Anti-Hallucination RL - no custom_policy_path provided or custom_simple_reward.py unavailable]")

        # Print comprehensive comparison
        print("\n" + "="*70)
        print("COMPREHENSIVE COMPARISON - ALL MODELS")
        print("="*70)

        results = {
            'baseline': baseline_results,
            'base_rl': base_rl_results,
            'custom_rl': custom_rl_results
        }

        print("\nMetric          | Baseline   | Base RL    | Anti-Halluc RL | RL vs Base | Custom vs Base")
        print("-" * 95)

        for metric in ['EM', 'F1', 'Precision', 'Recall']:
            baseline_val = baseline_results.get(metric, 0.0)
            base_rl_val = base_rl_results.get(metric, 0.0) if base_rl_results else 0.0
            custom_rl_val = custom_rl_results.get(metric, 0.0) if custom_rl_results else 0.0

            rl_diff = base_rl_val - baseline_val
            custom_diff = custom_rl_val - baseline_val

            print(f"{metric:15} | {baseline_val:10.2f} | {base_rl_val:10.2f} | {custom_rl_val:14.2f} | {rl_diff:+10.2f} | {custom_diff:+14.2f}")

        if base_rl_results and 'Avg_Reward' in base_rl_results:
            print(f"\nBase RL Reward: {base_rl_results['Avg_Reward']:.4f}")
        if custom_rl_results and 'Avg_Base_Reward' in custom_rl_results:
            print(f"Anti-Halluc RL Reward: {custom_rl_results['Avg_Base_Reward']:.4f}")
            if 'Avg_Halluc_Score' in custom_rl_results:
                print(f"Hallucination Score: {custom_rl_results['Avg_Halluc_Score']:.4f}")
                print(f"Abstention Rate: {custom_rl_results.get('Abstention_Rate', 0.0):.2f}%")

    # Save results if output path provided
    if args.output and results:
        print(f"\nSaving results to {args.output}...")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved successfully!")


if __name__ == "__main__":
    # Ensure all required libraries are installed:
    # pip install transformers datasets accelerate torch bitsandbytes tqdm
    main()
