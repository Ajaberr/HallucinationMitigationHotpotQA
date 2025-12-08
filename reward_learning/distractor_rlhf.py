"""
Verifier-Based Reward Learning RLHF with DISTRACTOR setting (Qwen2-7B)

DISTRACTOR SETTING:
- Policy receives: Question + ALL context paragraphs (including distractors)
- Verifier receives: Question + ONLY supporting facts (correct context subset)

This makes the task harder for the policy (must filter distractors) while
the verifier can properly assess factuality against ground truth context.

REWARD FUNCTION:
- Factuality Score: f = p_ent - p_cont ∈ [-1, 1] from NLI verifier
- Confidence: conf = 1 - H_normalized ∈ [0, 1] from policy entropy
- Final Reward: R(x,y) = f * (λ_base + λ_conf * conf)

REINFORCEMENT LEARNING (REINFORCE):
- Sample responses from policy: ŷ ~ π_θ(·|x)
- Compute reward: R(x, ŷ) using verifier + confidence
- Optimize: E[-(R(x, ŷ) - b) log π_θ(ŷ|x)] where b is baseline (mean reward)

Components:
- FactualityVerifier: NLI-based verifier (DeBERTa-FEVER) for computing f = p_ent - p_cont
- EntropyConfidenceCalculator: Computes confidence from policy entropy
- RewardFunction: Combines factuality + confidence into scalar reward
- REINFORCELoss: Policy gradient loss with baseline
- SimpleRLTrainer: REINFORCE training using verifier-based rewards
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import re
import string
from collections import Counter
import signal
import sys
import os

# Import reward components from reward_and_loss.py
from reward_and_loss import (
    FactualityVerifier,
    AbstentionClassifier,
    EntropyConfidenceCalculator,
    RewardFunction,
    REINFORCELoss
)


################################################################################
# HotpotQA Evaluation Metrics (from baselines.py)
################################################################################

def normalize_answer(s: str) -> str:
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


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
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


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Computes Exact Match score (official HotpotQA version)."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_metrics(gold_answers: List[List[str]], pred_answers: List[str]) -> Dict[str, float]:
    """
    Calculates Exact Match (EM), F1 Score, Precision, and Recall for a list of answers.

    Args:
        gold_answers: List of lists of gold answers (multiple possible answers per question)
        pred_answers: List of predicted answers

    Returns:
        Dictionary with EM, F1, Precision, and Recall scores
    """
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


################################################################################
# Reward Configuration (Verifier-based)
################################################################################

@dataclass
class RewardModelConfig:
    """Configuration for verifier-based reward function"""
    verifier_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    lambda_base: float = 0.5
    lambda_conf: float = 0.5
    abstention_reward: float = 0.3  # Positive reward for abstentions (between 0 and correct answer)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RewardModel:
    """
    Verifier-based reward model for factuality + confidence.
    Uses NLI verifier (DeBERTa-FEVER) + entropy-based confidence.

    This is NOT a trainable neural network - it's a combination of:
    1. Pre-trained verifier for factuality
    2. Entropy calculator for confidence
    """

    def __init__(
        self,
        verifier_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        lambda_base: float = 0.5,
        lambda_conf: float = 0.5,
        abstention_reward: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Initialize verifier
        self.verifier = FactualityVerifier(
            model_name=verifier_model,
            device=device
        )

        # Note: vocab_size will be set when we have access to the policy tokenizer
        self.confidence_calc = None

        # Reward function
        self.reward_fn = RewardFunction(
            verifier=self.verifier,
            confidence_calc=None,  # Will be set later
            lambda_base_init=lambda_base,
            lambda_conf_init=lambda_conf
        )

        self.device = device
        self.lambda_base = lambda_base
        self.lambda_conf = lambda_conf
        self.abstention_reward = abstention_reward

    def set_vocab_size(self, vocab_size: int):
        """Set vocab size for entropy calculation (call this after policy is loaded)"""
        self.confidence_calc = EntropyConfidenceCalculator(vocab_size)
        self.reward_fn.confidence_calc = self.confidence_calc

    def compute_rewards(
        self,
        evidences: List[str],
        answers: List[str],
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rewards using verifier + confidence.

        Args:
            evidences: Evidence texts (questions in QA context)
            answers: Generated answers
            logits: Policy logits [B, T, V]
            mask: Attention mask [B, T]

        Returns:
            Dictionary with rewards, factuality, confidence scores
        """
        return self.reward_fn.compute_rewards(evidences, answers, logits, mask)

    def to(self, device):
        """Move verifier to device"""
        self.verifier.model.to(device)
        self.device = device
        return self

    def parameters(self):
        """Return empty iterator (no trainable parameters)"""
        return iter([])

    def eval(self):
        """Set verifier to eval mode"""
        self.verifier.model.eval()
        return self


class RewardModelTrainer:
    """
    Trainer wrapper for verifier-based reward model.

    NOTE: This model doesn't require training - it uses a pre-trained verifier.
    This class exists for API compatibility with the old Bradley-Terry trainer.
    """

    def __init__(self, config: RewardModelConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize verifier-based reward model
        self.model = RewardModel(
            verifier_model=config.verifier_model,
            lambda_base=config.lambda_base,
            lambda_conf=config.lambda_conf,
            abstention_reward=config.abstention_reward,
            device=config.device
        )

    def score_text(
        self,
        question: str,
        answer: str,
        logits: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Score a question-answer pair using the verifier.

        Args:
            question: The question (acts as evidence)
            answer: The generated answer
            logits: Optional logits for confidence (if None, conf=0.5)
            mask: Optional attention mask

        Returns:
            reward: Scalar reward score
        """
        # If no logits provided, use factuality only
        if logits is None:
            p_ent, p_cont, f = self.model.verifier.compute_factuality_score([question], [answer])
            # Use simple reward without confidence
            return (f[0] * self.config.lambda_base).item()

        # Full reward with confidence
        reward_info = self.model.compute_rewards(
            evidences=[question],
            answers=[answer],
            logits=logits.unsqueeze(0) if logits.dim() == 2 else logits,
            mask=mask.unsqueeze(0) if mask is not None and mask.dim() == 1 else mask
        )

        return reward_info["rewards"][0].item()

    def compare_responses(
        self,
        question: str,
        response_a: str,
        response_b: str
    ) -> Dict[str, float]:
        """
        Compare two responses using factuality scores.

        Args:
            question: The question
            response_a: First response
            response_b: Second response

        Returns:
            Dictionary with scores and preference probability
        """
        score_a = self.score_text(question, response_a)
        score_b = self.score_text(question, response_b)

        # Probability that A is preferred over B (using sigmoid)
        prob_a_over_b = 1 / (1 + np.exp(score_b - score_a))

        return {
            'score_a': score_a,
            'score_b': score_b,
            'prob_a_preferred': prob_a_over_b,
            'prob_b_preferred': 1 - prob_a_over_b
        }

    def save_model(self, path: str):
        """Save the reward model configuration"""
        torch.save({
            'config': self.config,
            'lambda_base': self.model.lambda_base,
            'lambda_conf': self.model.lambda_conf,
            'abstention_reward': self.model.abstention_reward
        }, path)
        print(f"Reward config saved to {path}")

    def load_model(self, path: str):
        """Load saved reward model configuration"""
        checkpoint = torch.load(path, map_location=self.device)
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        if 'lambda_base' in checkpoint:
            self.model.lambda_base = checkpoint['lambda_base']
        if 'lambda_conf' in checkpoint:
            self.model.lambda_conf = checkpoint['lambda_conf']
        if 'abstention_reward' in checkpoint:
            self.model.abstention_reward = checkpoint['abstention_reward']
        print(f"Reward config loaded from {path}")


################################################################################
# PHASE B: Simple Reward Learning (REINFORCE) with DISTRACTOR setting
################################################################################

@dataclass
class SimpleRLConfig:
    """Configuration for REINFORCE policy training"""
    policy_model_name: str = "Qwen/Qwen2-7B"
    learning_rate: float = 1e-5
    batch_size: int = 2
    num_epochs: int = 1
    max_length: int = 2048  # Increased for distractor context
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.9
    kl_penalty: float = 0.1  # KL divergence penalty coefficient
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PromptDataset(Dataset):
    """Dataset containing prompts for policy training"""

    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]


class SimpleRLTrainer:
    """
    Simple reward learning using REINFORCE with verifier-based rewards.
    DISTRACTOR SETTING: Policy sees all context, verifier sees only supporting facts.
    """

    def __init__(self, config: SimpleRLConfig, reward_model: RewardModel = None):
        self.config = config
        self.device = torch.device(config.device)

        # Policy we are optimizing: π_θ
        self.tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # IMPORTANT: Decoder-only models require left-padding for generation
        self.tokenizer.padding_side = 'left'

        # Load model with 4-bit quantization + LoRA
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
        import os

        # Check if this is a saved LoRA checkpoint
        is_lora_checkpoint = os.path.exists(os.path.join(config.policy_model_name, "adapter_config.json"))

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        if is_lora_checkpoint:
            print(f"Loading LoRA checkpoint from {config.policy_model_name}...")
            self.policy = AutoPeftModelForCausalLM.from_pretrained(
                config.policy_model_name,
                device_map="auto",
                quantization_config=bnb_config,
                is_trainable=True
            )
            print("LoRA checkpoint loaded successfully!")

            self.policy.train()
            for name, param in self.policy.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True

        else:
            print(f"Loading base model {config.policy_model_name} with LoRA adapters...")
            self.policy = AutoModelForCausalLM.from_pretrained(
                config.policy_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="sdpa" if torch.cuda.is_available() else "eager"
            )

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )

            self.policy = get_peft_model(self.policy, lora_config)
            self.policy.print_trainable_parameters()

        # Verifier-based reward model
        self.reward_model = None
        self.abstention_classifier = None
        if reward_model is not None:
            self.reward_model = reward_model.to(self.device)
            self.reward_model.eval()
            self.reward_model.set_vocab_size(self.tokenizer.vocab_size)
            # Initialize abstention classifier
            self.abstention_classifier = AbstentionClassifier(self.reward_model.verifier)

        # Store reference policy for KL divergence
        self.ref_policy = None
        if reward_model is not None and config.kl_penalty > 0.0:
            print(f"Creating reference policy for KL divergence penalty (β={config.kl_penalty})...")
            from copy import deepcopy
            self.ref_policy = deepcopy(self.policy)

            for param in self.ref_policy.parameters():
                param.requires_grad = False
            self.ref_policy.eval()
            print(f"Reference policy created and frozen")

        # REINFORCE loss function
        self.loss_fn = REINFORCELoss(
            baseline_type="batch_mean",
            kl_penalty=config.kl_penalty
        ) if reward_model is not None else None

        # Optimizer for policy
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate
        ) if reward_model is not None else None

        # Checkpoint management
        self.checkpoint_dir = "checkpoints_distractor"
        self.current_step = 0
        self.current_epoch = 0
        self.should_exit = False

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, _):
        """Handle SIGTERM and SIGINT by saving checkpoint"""
        print(f"\n{'='*70}")
        print(f"Received signal {signum} - saving checkpoint before exit...")
        print(f"{'='*70}")
        self.save_checkpoint(f"checkpoint_interrupted_step_{self.current_step}")
        print(f"Checkpoint saved. Exiting gracefully.")
        self.should_exit = True
        sys.exit(0)

    def save_checkpoint(self, checkpoint_name: str = None):
        """Save training checkpoint"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{self.current_epoch}_step_{self.current_step}"

        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        print(f"\nSaving checkpoint to {checkpoint_path}...")

        # Save LoRA adapters
        self.policy.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config.__dict__
        }
        torch.save(state, os.path.join(checkpoint_path, 'training_state.pt'))

        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def compute_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int
    ) -> torch.Tensor:
        """Compute log π_θ(y | x) for the response tokens only."""
        outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]

        # NUMERICAL SAFETY
        logits = torch.clamp(logits, -50, 50)
        log_probs = torch.log_softmax(logits, dim=-1)

        next_ids = input_ids[:, 1:]
        token_logprobs = log_probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)

        # Mask to only keep response tokens
        mask = torch.zeros_like(token_logprobs)
        mask[:, prompt_len-1:] = 1.0

        response_logprobs = (token_logprobs * mask).sum(dim=1)
        return response_logprobs

    def train_step(self, batch_data: List[Tuple[str, str, str]]) -> Tuple[float, float, float, Dict[str, float]]:
        """
        Single REINFORCE training step with distractor setting.

        Args:
            batch_data: List of (prompt_with_all_context, gold_answer, supporting_evidence_only) tuples

        Returns:
            Tuple of (loss, mean_reward, mean_logprob, extra_metrics)
        """
        # Unpack batch
        prompts_with_distractors = [p for p, _, _ in batch_data]  # Policy sees ALL context
        gold_answers = [a for _, a, _ in batch_data]
        supporting_evidences = [e for _, _, e in batch_data]  # Verifier sees ONLY supporting facts

        # 1) Encode prompts with distractors
        encoded = self.tokenizer(
            prompts_with_distractors,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_len = input_ids.size(1)

        # 2) Sample responses from policy
        self.policy.eval()
        with torch.no_grad():
            gen_ids = self.policy.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,  # Enable sampling for exploration (allows discovering abstention rewards)
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False
            )

        generated_sequences = gen_ids[:, prompt_len:]
        generated_answers = self.tokenizer.batch_decode(
            generated_sequences,
            skip_special_tokens=True
        )

        # 3) Recompute logits
        gen_attention = (gen_ids != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = self.policy(
                input_ids=gen_ids,
                attention_mask=gen_attention
            )
            logits = outputs.logits[:, :-1, :]
            logits = torch.clamp(logits, -50, 50)
            log_probs_all = torch.log_softmax(logits, dim=-1)

        # 4) Compute entropy-based confidence
        import math
        with torch.no_grad():
            probs = log_probs_all.exp()
            entropy = -(probs * log_probs_all).sum(dim=-1)

            resp_mask = torch.zeros_like(entropy)
            resp_mask[:, prompt_len-1:] = 1.0

            sum_entropy = (entropy * resp_mask).sum(dim=1)
            num_resp_tokens = resp_mask.sum(dim=1).clamp(min=1)
            avg_entropy = sum_entropy / num_resp_tokens

            max_entropy = math.log(self.tokenizer.vocab_size)
            conf = 1.0 - (avg_entropy / max_entropy)
            conf = torch.clamp(conf, 0.0, 1.0)
            conf = torch.nan_to_num(conf, nan=0.0)

        # 5) Compute factuality using SUPPORTING EVIDENCE ONLY (not distractors!)
        with torch.no_grad():
            p_ent, p_cont, f = self.reward_model.verifier.compute_factuality_score(
                evidences=supporting_evidences,  # ← Only supporting facts!
                answers=generated_answers
            )

            # --- ABSTENTION DETECTION using classifier ---
            abstention_scores = self.abstention_classifier.predict_proba(generated_answers).to(f.device)
            abstention_threshold = 0.5
            abstained = (abstention_scores >= abstention_threshold)  # bool [B]

            # --- BASE REWARD: factuality * (λ_base + λ_conf * conf) ---
            lambda_base = self.reward_model.lambda_base
            lambda_conf = self.reward_model.lambda_conf
            base_reward = f * (lambda_base + lambda_conf * conf)  # [B]

            # --- CONFIDENCE-WEIGHTED PENALTY for confident negatives ---
            # If f < 0, multiply by (1 + alpha * conf^gamma) to make confident errors more negative
            alpha_penalty = 2.0   # tuneable hyperparam
            gamma_penalty = 2.0   # quadratic in confidence

            negative_mask = (f < 0) & (~abstained)
            penalty_multiplier = torch.ones_like(base_reward)
            # Ensure dtype matches for assignment
            penalty_values = (1.0 + alpha_penalty * (conf[negative_mask] ** gamma_penalty)).to(penalty_multiplier.dtype)
            penalty_multiplier[negative_mask] = penalty_values

            reward_answer = base_reward * penalty_multiplier  # [B]

            # --- APPLY ABSTENTION: R = abstention_reward (positive, between wrong and correct) ---
            # This creates the preference ordering: correct > abstention > wrong
            abstention_reward_val = self.reward_model.abstention_reward
            rewards = torch.where(
                abstained,
                torch.full_like(reward_answer, abstention_reward_val),
                reward_answer
            )

            # REWARD NORMALIZATION: Normalize to mean=0, std=1 for variance reduction
            # Abstentions now maintain their positive reward relative to wrong answers
            rewards_mean = rewards.mean()
            rewards_std = rewards.std()
            if rewards_std > 1e-8:  # Avoid division by zero
                rewards = (rewards - rewards_mean) / (rewards_std + 1e-8)

            # Store original stats for logging
            original_reward_mean = rewards_mean.item()
            original_reward_std = rewards_std.item()
            abstention_rate = abstained.float().mean().item()

        torch.cuda.empty_cache()

        # 6) Compute log π_θ(ŷ | x) for REINFORCE
        self.policy.train()
        gen_attention = (gen_ids != self.tokenizer.pad_token_id).long()
        log_probs = self.compute_logprobs(gen_ids, gen_attention, prompt_len)

        # 6b) Compute reference log probs for KL penalty
        ref_log_probs = None
        if self.ref_policy is not None:
            with torch.no_grad():
                ref_outputs = self.ref_policy(input_ids=gen_ids, attention_mask=gen_attention)
                ref_logits = ref_outputs.logits[:, :-1, :]
                ref_logits = torch.clamp(ref_logits, -50, 50)
                ref_log_probs_all = torch.log_softmax(ref_logits, dim=-1)

                next_ids = gen_ids[:, 1:]
                ref_token_logprobs = ref_log_probs_all.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)

                mask = torch.zeros_like(ref_token_logprobs)
                mask[:, prompt_len-1:] = 1.0

                ref_log_probs = (ref_token_logprobs * mask).sum(dim=1)

        # 7) REINFORCE loss with KL penalty
        loss, loss_metrics = self.loss_fn.compute_loss(rewards, log_probs, ref_log_probs)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Extra metrics
        extra_metrics = {
            "mean_factuality": f.mean().item(),
            "mean_confidence": conf.mean().item(),
            "abstention_rate": abstention_rate,            # % of answers that are abstentions
            "original_reward_mean": original_reward_mean,  # Before normalization
            "original_reward_std": original_reward_std,    # Before normalization
            "normalized_reward_mean": rewards.mean().item(),  # After normalization
            **loss_metrics
        }

        return loss.item(), original_reward_mean, log_probs.mean().item(), extra_metrics

    def train(self, training_data: List[Tuple[str, str, str]]) -> Dict[str, List[float]]:
        """Train the policy using REINFORCE with distractor setting"""
        losses = []
        rewards = []
        factuality_scores = []
        confidence_scores = []

        print("\n" + "="*70)
        print("REINFORCE Training with DISTRACTOR SETTING")
        print("="*70)
        print(f"Policy: Sees ALL context (including distractors)")
        print(f"Verifier: Sees ONLY supporting facts")
        print(f"Checkpoints: {self.checkpoint_dir}/ every 1000 steps")
        print("="*70)

        checkpoint_interval = 1000

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_losses = []
            epoch_rewards = []
            epoch_factuality = []
            epoch_confidence = []

            import random
            indices = list(range(len(training_data)))
            random.seed(42 + epoch)
            random.shuffle(indices)

            start_step = self.current_step if self.current_step > 0 else 0
            if start_step > 0:
                print(f"Resuming from step {start_step}...")

            for step in range(start_step, len(training_data), self.config.batch_size):
                if self.should_exit:
                    print("Graceful shutdown requested. Exiting training loop...")
                    break

                batch_indices = indices[step:step + self.config.batch_size]
                batch = [training_data[i] for i in batch_indices]

                loss, mean_reward, mean_logprob, extra_metrics = self.train_step(batch)

                epoch_losses.append(loss)
                epoch_rewards.append(mean_reward)
                epoch_factuality.append(extra_metrics["mean_factuality"])
                epoch_confidence.append(extra_metrics["mean_confidence"])

                self.current_step = step

                if step % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.config.num_epochs}, Step {step}")
                    print(f"  Loss: {loss:.4f}, Reward: {mean_reward:.4f} (std: {extra_metrics['original_reward_std']:.4f})")
                    print(f"  Factuality: {extra_metrics['mean_factuality']:.4f}, "
                          f"Confidence: {extra_metrics['mean_confidence']:.4f}, "
                          f"Abstention Rate: {extra_metrics['abstention_rate']:.2%}")
                    if self.config.kl_penalty > 0.0:
                        print(f"  KL Loss: {extra_metrics.get('kl_loss', 0.0):.4f}, "
                              f"KL Div: {extra_metrics.get('mean_kl_div', 0.0):.4f}")

                # Periodic checkpoint
                if step > 0 and step % checkpoint_interval == 0:
                    print(f"\n{'='*70}")
                    print(f"Saving periodic checkpoint at step {step}...")
                    self.save_checkpoint()
                    print(f"{'='*70}\n")

            if self.should_exit:
                break

            avg_loss = np.mean(epoch_losses)
            avg_reward = np.mean(epoch_rewards)
            avg_factuality = np.mean(epoch_factuality)
            avg_confidence = np.mean(epoch_confidence)

            losses.append(avg_loss)
            rewards.append(avg_reward)
            factuality_scores.append(avg_factuality)
            confidence_scores.append(avg_confidence)

            print(f"\nEpoch {epoch+1} completed:")
            print(f"  Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Factuality: {avg_factuality:.4f}, Avg Confidence: {avg_confidence:.4f}")

            print(f"\nSaving end-of-epoch checkpoint...")
            self.save_checkpoint()

        return {
            'losses': losses,
            'rewards': rewards,
            'factuality': factuality_scores,
            'confidence': confidence_scores
        }

    def generate_response(self, prompt: str) -> str:
        """Generate a response for a given prompt"""
        self.policy.eval()

        with torch.no_grad():
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            gen_ids = self.policy.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            response = self.tokenizer.decode(
                gen_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )

        return response

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint to resume training"""
        print(f"\nLoading checkpoint from {checkpoint_path}...")

        state_path = os.path.join(checkpoint_path, 'training_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.current_step = state.get('current_step', 0)
            self.current_epoch = state.get('current_epoch', 0)

            if self.optimizer and state.get('optimizer_state'):
                self.optimizer.load_state_dict(state['optimizer_state'])

            self.policy.train()
            print(f"Checkpoint loaded: epoch {self.current_epoch}, step {self.current_step}")
        else:
            print(f"Warning: training_state.pt not found in {checkpoint_path}")

    def save_policy(self, path: str):
        """Save the trained policy model (LoRA adapters only)"""
        print(f"Saving LoRA adapters to {path}...")
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"LoRA adapters saved to {path}")


################################################################################
# Main Training Loop with DISTRACTOR setting
################################################################################

def main(resume_from_checkpoint=None):
    """
    DISTRACTOR SETTING RLHF:
    - Policy receives: Question + ALL paragraphs (including distractors)
    - Verifier receives: Question + ONLY supporting facts

    Args:
        resume_from_checkpoint: Path to checkpoint directory to resume from
    """
    print("="*70)
    print("Distractor-Setting RLHF (Qwen2-7B)")
    if resume_from_checkpoint:
        print(f"RESUMING FROM CHECKPOINT: {resume_from_checkpoint}")
    print("="*70)

    # ========================================================================
    # Initialize Reward Model
    # ========================================================================
    print("\n" + "="*70)
    print("Initializing Verifier-Based Reward Model")
    print("="*70)

    rm_config = RewardModelConfig(
        verifier_model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        lambda_base=0.5,
        lambda_conf=0.5
    )

    print(f"\nDevice: {rm_config.device}")
    print(f"Verifier: {rm_config.verifier_model}")
    rm_trainer = RewardModelTrainer(rm_config)

    rm_trainer.save_model("verifier_reward_config_distractor.pt")

    # ========================================================================
    # Initialize RL Trainer
    # ========================================================================
    print("\n" + "="*70)
    print("REINFORCE Training with Distractor Setting")
    print("="*70)

    reward_model = rm_trainer.model
    reward_model.eval()

    USE_FINETUNED_MODEL = True

    if resume_from_checkpoint:
        policy_model = resume_from_checkpoint
        print(f"\nResuming from checkpoint: {policy_model}")
    elif USE_FINETUNED_MODEL:
        policy_model = "fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-Finetuned-10000"
    else:
        policy_model = "Qwen/Qwen2-7B"

    rl_config = SimpleRLConfig(
        policy_model_name=policy_model,
        learning_rate=5e-7,
        batch_size=2,
        num_epochs=1,
        max_new_tokens=20,
        max_length=2048,  # Larger for distractor context
        kl_penalty=0.0
    )

    print(f"\nPolicy Model: {rl_config.policy_model_name}")
    print(f"KL Penalty: {rl_config.kl_penalty}")
    rl_trainer = SimpleRLTrainer(rl_config, reward_model)

    if resume_from_checkpoint:
        rl_trainer.load_checkpoint(resume_from_checkpoint)

    # ========================================================================
    # Load HotpotQA with DISTRACTOR setting
    # ========================================================================
    print("\nLoading HotpotQA DISTRACTOR data (all samples from position 10000)...")
    from datasets import load_dataset

    dataset = load_dataset("hotpot_qa", "distractor", split="train")  # ← DISTRACTOR split!
    total_samples = len(dataset)
    print(f"Total training samples: {total_samples}")
    dataset = dataset.select(range(10000, total_samples))

    training_data = []
    for example in dataset:
        question = example['question']
        gold_answer = example['answer']

        # Get ALL context (including distractors) - for POLICY
        context_titles = example['context']['title']
        context_sentences = example['context']['sentences']

        all_context_parts = []
        for title, sentences in zip(context_titles, context_sentences):
            all_context_parts.append(f"{title}: {' '.join(sentences)}")
        all_context = "\n".join(all_context_parts)

        # Get ONLY supporting facts - for VERIFIER
        supporting_titles = example['supporting_facts']['title']
        supporting_sent_ids = example['supporting_facts']['sent_id']

        # Build supporting evidence
        supporting_parts = []
        for supp_title, supp_sent_id in zip(supporting_titles, supporting_sent_ids):
            # Find the paragraph with this title
            for i, title in enumerate(context_titles):
                if title == supp_title:
                    sentences = context_sentences[i]
                    if supp_sent_id < len(sentences):
                        supporting_parts.append(f"{supp_title}: {sentences[supp_sent_id]}")
                    break

        supporting_evidence = "\n".join(supporting_parts) if supporting_parts else question

        # Format prompt WITH all context (distractors) for policy
        prompt_with_all_context = f"""You are an expert at answering questions. Read the context and answer the question concisely.

Context:
{all_context}

Question: {question}
Answer: """

        training_data.append((prompt_with_all_context, gold_answer, supporting_evidence))

    print(f"Loaded {len(training_data)} training samples")
    print(f"Policy sees: Question + ALL context (with distractors)")
    print(f"Verifier sees: Question + ONLY supporting facts")

    # ========================================================================
    # Train
    # ========================================================================
    print(f"\nTraining with DISTRACTOR setting ({len(training_data)} samples)...")
    rl_metrics = rl_trainer.train(training_data)

    print(f"\nTraining Results:")
    print(f"Final RL Loss: {rl_metrics['losses'][-1]:.4f}")
    print(f"Final Avg Reward: {rl_metrics['rewards'][-1]:.4f}")
    print(f"Final Avg Factuality: {rl_metrics['factuality'][-1]:.4f}")
    print(f"Final Avg Confidence: {rl_metrics['confidence'][-1]:.4f}")

    # ========================================================================
    # Save
    # ========================================================================
    print("\n" + "="*70)
    print("Saving trained policy...")
    rl_trainer.save_policy("distractor_rlhf_policy")

    print("\n" + "="*70)
    print("Distractor-Setting RLHF Training Complete!")
    print("="*70)
    print(f"- Policy trained on: Question + ALL context (distractors)")
    print(f"- Verifier used: Question + ONLY supporting facts")
    print(f"- Final Reward: {rl_metrics['rewards'][-1]:.4f}")
    print(f"- Model saved to: distractor_rlhf_policy/")
    print("="*70)


if __name__ == "__main__":
    import glob

    # Auto-detect latest checkpoint for resuming
    checkpoint_dir = "checkpoints_distractor"
    resume_checkpoint = None

    if os.path.exists(checkpoint_dir):
        # Find all checkpoint directories
        checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*_step_*"))

        if checkpoint_paths:
            # Sort by step number (extract step from path)
            def get_step_number(path):
                try:
                    step_str = path.split("_step_")[-1]
                    return int(step_str)
                except:
                    return 0

            checkpoint_paths.sort(key=get_step_number, reverse=True)
            latest_checkpoint = checkpoint_paths[0]

            print("\n" + "="*70)
            print("CHECKPOINT DETECTED (DISTRACTOR)")
            print("="*70)
            print(f"Latest checkpoint found: {latest_checkpoint}")
            print(f"Available checkpoints: {len(checkpoint_paths)}")
            print("="*70)

            # Auto-resume from latest checkpoint
            resume_checkpoint = latest_checkpoint
            print(f"Auto-resuming from: {resume_checkpoint}")
            print("="*70)

    # Start or resume training
    main(resume_from_checkpoint=resume_checkpoint)
