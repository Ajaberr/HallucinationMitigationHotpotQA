"""
Verifier-Based Reward Learning RLHF (Qwen2-7B)

This module implements reward learning RLHF using factuality verification:

REWARD FUNCTION:
- Factuality Score: f = p_ent - p_cont ∈ [-1, 1] from NLI verifier
- Confidence: conf = 1 - H_normalized ∈ [0, 1] from policy entropy
- Final Reward: R(x,y) = f * (λ_base + λ_conf * conf)

REINFORCEMENT LEARNING (REINFORCE):
- Sample responses from policy: ŷ ~ π_θ(·|x)
- Compute reward: R(x, ŷ) using verifier + confidence
- Optimize: E[-(R(x, ŷ) - b) log π_θ(ŷ|x)] where b is baseline (mean reward)
- This is reward-weighted log-likelihood (REINFORCE algorithm)

Components:
- FactualityVerifier: NLI-based verifier (DeBERTa-FEVER) for computing f = p_ent - p_cont
- EntropyConfidenceCalculator: Computes confidence from policy entropy
- RewardFunction: Combines factuality + confidence into scalar reward
- REINFORCELoss: Policy gradient loss with baseline
- SimpleRLTrainer: REINFORCE training using verifier-based rewards

Reference: This uses reward learning (not DPO) with verifier-based factuality signals.
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
import json

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
# PHASE B: Simple Reward Learning (REINFORCE)
################################################################################

@dataclass
class SimpleRLConfig:
    """Configuration for REINFORCE policy training"""
    policy_model_name: str = "Qwen/Qwen2.5-7B"  # Use base model for supervised abstention learning
    learning_rate: float = 1e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients
    num_epochs: int = 1
    max_length: int = 512
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.9
    kl_penalty: float = 0.1  # KL divergence penalty coefficient (0.0 = no penalty, 0.01-0.1 = recommended)
    exploration_epsilon: float = 0.0  # No random forcing - using supervised labels
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

    This trainer optimizes a policy π_θ(y|x) using rewards from a verifier-based
    reward function R(x,y) = f * (λ_base + λ_conf * conf).
    The REINFORCE algorithm samples responses and uses reward-weighted log-likelihoods
    to update the policy.
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

        # Load model with 4-bit quantization + LoRA (matching qwen-finetune.py approach)
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
        import os

        # Check if this is a saved LoRA checkpoint (has adapter_config.json)
        is_lora_checkpoint = os.path.exists(os.path.join(config.policy_model_name, "adapter_config.json"))

        # Quantization config (matching qwen-finetune.py)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        if is_lora_checkpoint:
            # Load saved LoRA adapters from checkpoint
            print(f"Loading LoRA checkpoint from {config.policy_model_name}...")
            self.policy = AutoPeftModelForCausalLM.from_pretrained(
                config.policy_model_name,
                device_map="auto",
                quantization_config=bnb_config,
                is_trainable=True  # Ensure model is trainable
            )
            print("LoRA checkpoint loaded successfully!")

            # Enable gradient checkpointing for memory efficiency
            self.policy.enable_input_require_grads()
            self.policy.gradient_checkpointing_enable()

            # Ensure model is in training mode with gradients enabled
            self.policy.train()
            for name, param in self.policy.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True

        else:
            # Load base model and add new LoRA adapters (matching qwen-finetune.py)
            print(f"Loading base model {config.policy_model_name} with LoRA adapters...")
            self.policy = AutoModelForCausalLM.from_pretrained(
                config.policy_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="sdpa" if torch.cuda.is_available() else "eager"
            )

            # Prepare LoRA configuration (matching qwen-finetune.py)
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )

            # Add LoRA adapters to the quantized model
            self.policy = get_peft_model(self.policy, lora_config)

            # Enable gradient checkpointing for memory efficiency
            self.policy.enable_input_require_grads()
            self.policy.gradient_checkpointing_enable()

            # Print trainable parameters
            self.policy.print_trainable_parameters()

        # Verifier-based reward model (optional for evaluation-only mode)
        self.reward_model = None
        self.abstention_classifier = None
        if reward_model is not None:
            self.reward_model = reward_model.to(self.device)
            self.reward_model.eval()
            # Set vocab size for entropy calculation
            self.reward_model.set_vocab_size(self.tokenizer.vocab_size)
            # Initialize abstention classifier
            self.abstention_classifier = AbstentionClassifier(self.reward_model.verifier)

        # Store reference policy for KL divergence (frozen copy of initial policy)
        # Only create if training (reward_model is not None) and KL penalty > 0
        self.ref_policy = None
        if reward_model is not None and config.kl_penalty > 0.0:
            print(f"Creating reference policy for KL divergence penalty (β={config.kl_penalty})...")
            # For LoRA models: reference is the base model (before adapter training)
            # We keep the initial state by cloning the model at initialization
            if is_lora_checkpoint:
                # If loading from checkpoint, reference should be the base model
                print("WARNING: Loading from LoRA checkpoint - reference policy will also include checkpoint adapters")
                print("For true KL penalty against base model, train from scratch (not from checkpoint)")

            from copy import deepcopy
            self.ref_policy = deepcopy(self.policy)

            # Freeze all parameters
            for param in self.ref_policy.parameters():
                param.requires_grad = False
            self.ref_policy.eval()
            print(f"Reference policy created and frozen (memory: ~{torch.cuda.max_memory_allocated() / 1e9:.2f}GB)")

            # Warning about memory usage
            if torch.cuda.is_available():
                print(f"NOTE: Reference policy uses extra GPU memory. Consider lower kl_penalty if OOM occurs.")

        # REINFORCE loss function (only needed for training)
        self.loss_fn = REINFORCELoss(
            baseline_type="batch_mean",
            kl_penalty=config.kl_penalty
        ) if reward_model is not None else None

        # Optimizer for policy (only needed for training)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate
        ) if reward_model is not None else None

        # Mixed precision training scaler
        self.scaler = torch.cuda.amp.GradScaler() if reward_model is not None else None

        # Checkpoint management
        self.checkpoint_dir = "checkpoints"
        self.current_step = 0
        self.current_epoch = 0
        self.should_exit = False

        # Register signal handlers for graceful shutdown with checkpoint saving
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, _):
        """Handle SIGTERM and SIGINT by saving checkpoint and exiting gracefully"""
        print(f"\n{'='*70}")
        print(f"Received signal {signum} - saving checkpoint before exit...")
        print(f"{'='*70}")
        self.save_checkpoint(f"checkpoint_interrupted_step_{self.current_step}")
        print(f"Checkpoint saved. Exiting gracefully.")
        self.should_exit = True
        sys.exit(0)

    def save_checkpoint(self, checkpoint_name: str = None):
        """Save training checkpoint including model state and training progress"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{self.current_epoch}_step_{self.current_step}"

        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        print(f"\nSaving checkpoint to {checkpoint_path}...")

        # Save LoRA adapters (policy)
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
        """
        Compute log π_θ(y | x) for the response tokens only.

        Args:
            input_ids: [batch_size, seq_len] (prompt + response)
            attention_mask: [batch_size, seq_len]
            prompt_len: Length of the prompt (excluding response)

        Returns:
            log_probs: [batch_size] sum of log probabilities for response tokens
        """
        # Get model predictions
        outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # [B, T-1, V] - predict next token

        # NUMERICAL SAFETY: clamp logits to avoid NaN in softmax
        logits = torch.clamp(logits, -50, 50)

        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)  # [B, T-1, V]

        # Get the actual next tokens
        next_ids = input_ids[:, 1:]  # [B, T-1]

        # Gather log probs of actual tokens
        token_logprobs = log_probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        # Mask to only keep response tokens (positions >= prompt_len)
        mask = torch.zeros_like(token_logprobs)
        mask[:, prompt_len-1:] = 1.0  # Only response tokens

        # Sum log probs over response tokens
        response_logprobs = (token_logprobs * mask).sum(dim=1)  # [B]

        return response_logprobs

    def train_step(self, batch_data: List[Tuple[str, str, str, bool]], should_step: bool = True) -> Tuple[float, float, float, Dict[str, float]]:
        """
        Single REINFORCE training step with supervised abstention rewards

        Args:
            batch_data: List of (prompt, gold_answer, evidence, is_abstention_label) tuples
            should_step: Whether to step the optimizer (False for gradient accumulation)

        Returns:
            Tuple of (loss, mean_reward, mean_logprob, extra_metrics)
        """
        # Unpack batch
        prompts = [p for p, _, _, _ in batch_data]
        gold_answers = [a for _, a, _, _ in batch_data]
        evidences = [e for _, _, e, _ in batch_data]
        gold_is_abstention = [label for _, _, _, label in batch_data]

        # 1) Encode prompts (questions)
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_len = input_ids.size(1)

        # 2) Sample responses from policy π_θ(y|x) - sequence-only generation (memory-safe)
        # IMPORTANT: Set to eval mode for generation to avoid NaN with LoRA + quantization
        self.policy.eval()
        with torch.no_grad():
            # Use sampling for exploration (allows model to discover abstentions)
            gen_ids = self.policy.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,              # Use sampling for exploration
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False  # Returns sequences only, not dict
            )  # [B, T_total]

        # Extract only generated part
        generated_sequences = gen_ids[:, prompt_len:]  # [B, T_gen]

        # Decode generated answers
        generated_answers = self.tokenizer.batch_decode(
            generated_sequences,
            skip_special_tokens=True
        )

        # NO FORCING: Let model learn naturally from reward signal
        # Model will learn: Abstaining (0) > Being Wrong (-10)

        # 3) Recompute logits safely via separate forward pass (NUMERICAL SAFETY)
        # This is needed for: (a) REINFORCE log_probs, (b) entropy-based confidence
        gen_attention = (gen_ids != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = self.policy(
                input_ids=gen_ids,
                attention_mask=gen_attention
            )
            logits = outputs.logits[:, :-1, :]  # [B, T_total-1, V] - predict next token

            # NUMERICAL SAFETY: clamp logits to avoid NaN in softmax
            logits = torch.clamp(logits, -50, 50)

            # Compute log probabilities for all positions (stable)
            log_probs_all = torch.log_softmax(logits, dim=-1)  # [B, T_total-1, V]

        # 4) Compute entropy-based confidence from recomputed logits (response tokens only)
        import math
        with torch.no_grad():
            # Compute probabilities and entropy per token
            probs = log_probs_all.exp()  # [B, T_total-1, V]
            entropy = -(probs * log_probs_all).sum(dim=-1)  # [B, T_total-1]

            # Mask: only response positions (>= prompt_len-1 due to shift)
            resp_mask = torch.zeros_like(entropy)
            resp_mask[:, prompt_len-1:] = 1.0

            # Average entropy over response tokens
            sum_entropy = (entropy * resp_mask).sum(dim=1)
            num_resp_tokens = resp_mask.sum(dim=1).clamp(min=1)
            avg_entropy = sum_entropy / num_resp_tokens  # [B]

            # Normalize by log(vocab_size) and compute confidence
            max_entropy = math.log(self.tokenizer.vocab_size)
            conf = 1.0 - (avg_entropy / max_entropy)
            conf = torch.clamp(conf, 0.0, 1.0)
            conf = torch.nan_to_num(conf, nan=0.0)  # Safety for NaNs

        # 5) Compute factuality scores using verifier (no logits needed)
        with torch.no_grad():
            p_ent, p_cont, f = self.reward_model.verifier.compute_factuality_score(
                evidences=evidences,  # Use HotpotQA context as evidence/premise
                answers=generated_answers
            )

            # --- ABSTENTION DETECTION using classifier ---
            abstention_scores = self.abstention_classifier.predict_proba(generated_answers).to(f.device)
            abstention_threshold = 0.5
            abstained = (abstention_scores >= abstention_threshold)  # bool [B]

            # DEBUG: Print sample answers and their abstention scores every 100 steps
            if self.current_step % 100 == 0:
                print(f"\n[DEBUG] Sample answers at step {self.current_step}:")
                for i, (answer, score, is_abstain) in enumerate(zip(generated_answers[:2], abstention_scores[:2], abstained[:2])):
                    print(f"  Answer {i}: '{answer}'")
                    print(f"    Abstention score: {score.item():.4f}, Classified as abstention: {is_abstain.item()}")

            # --- NEW REWARD STRUCTURE: Learn to abstain when uncertain ---
            # Abstention = 0 (neutral), Wrong = amplified negative, Correct = factuality-based

            # Initialize rewards
            rewards = torch.zeros_like(f)

            # Penalty multiplier for wrong answers
            WRONG_ANSWER_MULTIPLIER = 50.0  # Strong penalty to encourage exploration of abstention

            for i in range(len(rewards)):
                model_abstained = abstained[i].item()

                if model_abstained:
                    # Abstention: neutral reward (0)
                    rewards[i] = 0.0
                else:
                    # Model answered: reward based on factuality
                    lambda_base = self.reward_model.lambda_base
                    lambda_conf = self.reward_model.lambda_conf
                    factuality_reward = f[i] * (lambda_base + lambda_conf * conf[i])

                    if factuality_reward < 0:
                        # Wrong answer: amplify the penalty
                        rewards[i] = factuality_reward * WRONG_ANSWER_MULTIPLIER
                    else:
                        # Correct answer: positive reward
                        rewards[i] = factuality_reward

            # NO NORMALIZATION - keep raw rewards to preserve penalty magnitude
            # Store stats for logging
            original_reward_mean = rewards.mean().item()
            original_reward_std = rewards.std().item()
            abstention_rate = abstained.float().mean().item()

        # Clear CUDA cache to free memory before gradient-requiring forward pass
        torch.cuda.empty_cache()

        # 6) Compute log π_θ(ŷ | x) for sampled responses (REINFORCE)
        # Need to re-run forward pass for gradients - set back to train mode
        self.policy.train()
        gen_attention = (gen_ids != self.tokenizer.pad_token_id).long()

        # Use mixed precision for forward pass
        with torch.cuda.amp.autocast():
            log_probs = self.compute_logprobs(gen_ids, gen_attention, prompt_len)  # [B]

            # 6b) Compute reference log probs for KL divergence penalty (if enabled)
            ref_log_probs = None
            if self.ref_policy is not None:
                with torch.no_grad():
                    # Compute log probs using frozen reference policy
                    ref_outputs = self.ref_policy(input_ids=gen_ids, attention_mask=gen_attention)
                    ref_logits = ref_outputs.logits[:, :-1, :]  # [B, T-1, V]

                    # NUMERICAL SAFETY: clamp logits to avoid NaN
                    ref_logits = torch.clamp(ref_logits, -50, 50)

                    ref_log_probs_all = torch.log_softmax(ref_logits, dim=-1)  # [B, T-1, V]

                    # Get log probs for actual tokens
                    next_ids = gen_ids[:, 1:]  # [B, T-1]
                    ref_token_logprobs = ref_log_probs_all.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

                    # Mask to only keep response tokens
                    mask = torch.zeros_like(ref_token_logprobs)
                    mask[:, prompt_len-1:] = 1.0

                    # Sum log probs over response tokens
                    ref_log_probs = (ref_token_logprobs * mask).sum(dim=1)  # [B]

            # 7) REINFORCE loss with baseline and KL penalty
            loss, loss_metrics = self.loss_fn.compute_loss(rewards, log_probs, ref_log_probs)

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.config.gradient_accumulation_steps

        # Backward pass with gradient scaling (accumulate gradients)
        self.scaler.scale(scaled_loss).backward()

        # Only step optimizer when we've accumulated enough gradients
        if should_step:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            # Clear CUDA cache after optimizer step for better memory management
            torch.cuda.empty_cache()

        # Extra metrics
        extra_metrics = {
            "mean_factuality": f.mean().item(),
            "mean_confidence": conf.mean().item(),
            "abstention_rate": abstention_rate,            # % of answers that are abstentions
            "original_reward_mean": original_reward_mean,  # Raw reward mean
            "original_reward_std": original_reward_std,    # Raw reward std
            **loss_metrics
        }

        # Save prediction details for analysis
        prediction_details = []
        for i in range(len(generated_answers)):
            prediction_details.append({
                "question": prompts[i],
                "generated_answer": generated_answers[i],
                "gold_answer": gold_answers[i],
                "abstention_score": abstention_scores[i].item(),
                "is_abstention": abstained[i].item(),
                "factuality_score": f[i].item(),
                "confidence": conf[i].item(),
                "reward": rewards[i].item()  # Raw reward (no normalization)
            })
        extra_metrics["predictions"] = prediction_details

        return loss.item(), original_reward_mean, log_probs.mean().item(), extra_metrics

    def train(self, training_data: List[Tuple[str, str, str, bool]]) -> Dict[str, List[float]]:
        """
        Train the policy using REINFORCE with supervised abstention rewards

        Args:
            training_data: List of (prompt, gold_answer, evidence, is_abstention_label) tuples for training

        Returns:
            Dictionary with training metrics
        """
        losses = []
        rewards = []
        factuality_scores = []
        confidence_scores = []

        print("\n" + "="*70)
        print("REINFORCE Policy Training with Verifier-Based Rewards")
        print("="*70)
        print(f"Checkpoints will be saved every 1000 steps to {self.checkpoint_dir}/")
        print(f"Use Ctrl+C or kill signal to save checkpoint and exit gracefully")
        print("="*70)

        checkpoint_interval = 1000  # Save checkpoint every 1000 steps

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_losses = []
            epoch_rewards = []
            epoch_factuality = []
            epoch_confidence = []

            # Manual batching instead of DataLoader (which doesn't handle string tuples well)
            import random
            indices = list(range(len(training_data)))
            # Use fixed seed for reproducibility when resuming
            random.seed(42 + epoch)
            random.shuffle(indices)

            # Resume from checkpoint step if available
            start_step = self.current_step if self.current_step > 0 else 0
            if start_step > 0:
                print(f"Resuming from step {start_step}...")

            # Track gradient accumulation iterations
            accumulation_iter = 0

            for step in range(start_step, len(training_data), self.config.batch_size):
                # Check if we should exit gracefully
                if self.should_exit:
                    print("Graceful shutdown requested. Exiting training loop...")
                    break

                # Get batch indices
                batch_indices = indices[step:step + self.config.batch_size]
                batch = [training_data[i] for i in batch_indices]

                # Determine if we should step the optimizer this iteration
                accumulation_iter += 1
                should_step = (accumulation_iter % self.config.gradient_accumulation_steps == 0)

                loss, mean_reward, mean_logprob, extra_metrics = self.train_step(batch, should_step)

                epoch_losses.append(loss)
                epoch_rewards.append(mean_reward)
                epoch_factuality.append(extra_metrics["mean_factuality"])
                epoch_confidence.append(extra_metrics["mean_confidence"])

                # Update global step counter
                self.current_step = step

                if step % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.config.num_epochs}, Step {step}")
                    print(f"  Loss: {loss:.4f}, Reward: {mean_reward:.4f} (std: {extra_metrics['original_reward_std']:.4f})")
                    print(f"  Factuality: {extra_metrics['mean_factuality']:.4f}, "
                          f"Confidence: {extra_metrics['mean_confidence']:.4f}, "
                          f"Abstention Rate: {extra_metrics['abstention_rate']:.2%}")
                    # Print KL divergence metrics if KL penalty is enabled
                    if self.config.kl_penalty > 0.0:
                        print(f"  KL Loss: {extra_metrics.get('kl_loss', 0.0):.4f}, "
                              f"KL Div: {extra_metrics.get('mean_kl_div', 0.0):.4f}")

                # Periodic checkpoint saving
                if step > 0 and step % checkpoint_interval == 0:
                    print(f"\n{'='*70}")
                    print(f"Saving periodic checkpoint at step {step}...")
                    self.save_checkpoint()

                    # Save prediction details to JSON
                    predictions_file = f"predictions_step_{step}.json"
                    if "predictions" in extra_metrics:
                        with open(predictions_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                "step": step,
                                "epoch": epoch,
                                "predictions": extra_metrics["predictions"],
                                "summary": {
                                    "mean_factuality": extra_metrics["mean_factuality"],
                                    "mean_confidence": extra_metrics["mean_confidence"],
                                    "abstention_rate": extra_metrics["abstention_rate"],
                                    "mean_reward": mean_reward
                                }
                            }, f, indent=2, ensure_ascii=False)
                        print(f"Saved predictions to {predictions_file}")

                    print(f"{'='*70}\n")

            # Check if we should exit after epoch
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

            # Save checkpoint at end of epoch
            print(f"\nSaving end-of-epoch checkpoint...")
            self.save_checkpoint()

        return {
            'losses': losses,
            'rewards': rewards,
            'factuality': factuality_scores,
            'confidence': confidence_scores
        }

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response for a given prompt using the trained policy

        Args:
            prompt: Input prompt

        Returns:
            Generated response text
        """
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
                max_new_tokens=64,  # Use reasonable length for evaluation
                do_sample=True,  # Use sampling (same as training)
                top_p=0.9,  # Nucleus sampling (same as training)
                temperature=0.8,  # Same as training
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id  # Tell model when to stop
            )

            # Decode only the generated part (skip the prompt)
            response = self.tokenizer.decode(
                gen_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )

        return response

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint to resume training"""
        print(f"\nLoading checkpoint from {checkpoint_path}...")

        # Load training state
        state_path = os.path.join(checkpoint_path, 'training_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.current_step = state.get('current_step', 0)
            self.current_epoch = state.get('current_epoch', 0)

            # Load optimizer state if it exists
            if self.optimizer and state.get('optimizer_state'):
                self.optimizer.load_state_dict(state['optimizer_state'])

            # Ensure model is in training mode with gradients enabled
            self.policy.train()
            print(f"Model set to training mode")

            # Verify LoRA parameters have gradients
            lora_params_with_grad = sum(1 for n, p in self.policy.named_parameters()
                                       if 'lora' in n.lower() and p.requires_grad)
            print(f"LoRA parameters with gradients: {lora_params_with_grad}")

            print(f"Checkpoint loaded: epoch {self.current_epoch}, step {self.current_step}")
        else:
            print(f"Warning: training_state.pt not found in {checkpoint_path}")

    def save_policy(self, path: str):
        """
        Save the trained policy model (LoRA adapters only).

        This saves only the LoRA adapter weights (~80MB), not the full model.
        To merge and save the full model, use merge_and_save_full_model().
        """
        print(f"Saving LoRA adapters to {path}...")
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"LoRA adapters saved to {path}")

    def merge_and_save_full_model(
        self,
        adapter_path: str,
        output_path: str,
        push_to_hub: bool = False,
        repo_id: str = None
    ):
        """
        Merge LoRA adapters with base model and save/push the full model.

        This follows the approach from save_model.py:
        1. Reload base model in FP16
        2. Load LoRA adapters
        3. Merge adapters into base model
        4. Optionally push to HuggingFace Hub

        Args:
            adapter_path: Path to saved LoRA adapters
            output_path: Path to save the merged full model
            push_to_hub: Whether to push to HuggingFace Hub
            repo_id: HuggingFace repo ID (required if push_to_hub=True)
        """
        from peft import PeftModel

        print("\n" + "="*70)
        print("Merging LoRA adapters with base model...")
        print("="*70)

        # Clear GPU memory
        torch.cuda.empty_cache()

        # 1. Reload base model in FP16 (cannot merge 4bit directly)
        print(f"Loading base model {self.config.policy_model_name} in FP16...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.policy_model_name,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # 2. Load the adapter using the base_model object
        print(f"Loading LoRA adapters from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)

        # 3. Merge
        print("Merging adapters into base model...")
        model = model.merge_and_unload()

        # 4. Save locally
        print(f"Saving merged model to {output_path}...")
        model.save_pretrained(output_path, safe_serialization=True)
        self.tokenizer.save_pretrained(output_path)
        print(f"Merged model saved to {output_path}")

        # 5. Push to hub if requested
        if push_to_hub:
            if repo_id is None:
                raise ValueError("repo_id must be provided when push_to_hub=True")
            print(f"Pushing to HuggingFace Hub: {repo_id}...")
            model.push_to_hub(repo_id, safe_serialization=True)
            self.tokenizer.push_to_hub(repo_id)
            print(f"Model pushed to {repo_id}")

        print("="*70)
        print("Merge complete!")
        print("="*70)

    def evaluate_on_hotpotqa(
        self,
        questions: List[str],
        gold_answers: List[List[str]],
        reward_model_trainer: 'RewardModelTrainer' = None
    ) -> Dict[str, float]:
        """
        Evaluate the trained policy on HotpotQA questions using official metrics.

        Args:
            questions: List of HotpotQA questions
            gold_answers: List of lists of gold answers (multiple possible answers per question)
            reward_model_trainer: Optional reward model trainer to compute reward scores

        Returns:
            Dictionary with EM, F1, Precision, Recall, and reward metrics
        """
        self.policy.eval()

        print("\n" + "="*70)
        print(f"Evaluating policy on {len(questions)} HotpotQA questions")
        print("="*70)

        pred_answers = []
        rewards = []
        factuality_scores = []
        confidence_scores = []

        for i, question in enumerate(questions):
            # Generate response
            response = self.generate_response(question)
            pred_answers.append(response)

            # Compute reward if reward model is provided
            if reward_model_trainer is not None:
                reward = reward_model_trainer.score_text(question, response)
                rewards.append(reward)

                # Get detailed factuality score
                _, _, f = reward_model_trainer.model.verifier.compute_factuality_score(
                    [question], [response]
                )
                factuality_scores.append(f[0].item())

            if (i + 1) % 20 == 0:
                print(f"Processed {i+1}/{len(questions)} questions...")

        # Compute HotpotQA metrics
        metrics = compute_metrics(gold_answers, pred_answers)

        # Add reward metrics if available
        if rewards:
            metrics['Avg_Reward'] = np.mean(rewards)
        if factuality_scores:
            metrics['Avg_Factuality'] = np.mean(factuality_scores)

        print("\n" + "="*70)
        print("HotpotQA Evaluation Results:")
        print("="*70)
        print(f"  Exact Match (EM): {metrics['EM']:.2f}%")
        print(f"  F1 Score: {metrics['F1']:.2f}%")
        print(f"  Precision: {metrics['Precision']:.2f}%")
        print(f"  Recall: {metrics['Recall']:.2f}%")
        if 'Avg_Reward' in metrics:
            print(f"  Average Reward: {metrics['Avg_Reward']:.4f}")
        if 'Avg_Factuality' in metrics:
            print(f"  Average Factuality: {metrics['Avg_Factuality']:.4f}")
        print("="*70)

        return metrics


################################################################################
# Example usage and demonstration
################################################################################

def create_example_prompts() -> List[str]:
    """Create example questions for RL training"""
    return [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is the speed of light?",
        "Who wrote Romeo and Juliet?",
        "What is photosynthesis?",
    ]


def main(resume_from_checkpoint=None):
    """
    Verifier-Based Reward Learning RLHF demonstration:

    Uses NLI verifier (DeBERTa-FEVER) + entropy confidence for factuality-based rewards.
    Trains policy using REINFORCE with R(x,y) = f * (λ_base + λ_conf * conf).

    Args:
        resume_from_checkpoint: Path to checkpoint directory to resume training from
    """
    print("="*70)
    print("Verifier-Based Reward Learning RLHF (Qwen2-7B)")
    if resume_from_checkpoint:
        print(f"RESUMING FROM CHECKPOINT: {resume_from_checkpoint}")
    print("="*70)

    # ========================================================================
    # Initialize Verifier-Based Reward Model (No Training Required)
    # ========================================================================
    print("\n" + "="*70)
    print("Initializing Verifier-Based Reward Model")
    print("="*70)

    # Configuration
    rm_config = RewardModelConfig(
        verifier_model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        lambda_base=0.5,
        lambda_conf=0.5
    )

    # Initialize reward model (uses pre-trained verifier, no training needed)
    print("\nLoading pre-trained NLI verifier...")
    print(f"Device: {rm_config.device}")
    print(f"Verifier: {rm_config.verifier_model}")
    rm_trainer = RewardModelTrainer(rm_config)

    # Test reward scoring
    print("\n" + "="*70)
    print("Testing Verifier-Based Reward:")
    print("="*70)

    test_qa_pairs = [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of France?", "London"),  # Incorrect
        ("Who invented the telephone?", "Alexander Graham Bell"),
        ("Who invented the telephone?", "Thomas Edison"),  # Incorrect
    ]

    for question, answer in test_qa_pairs:
        score = rm_trainer.score_text(question, answer)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"Reward: {score:.4f}")

    # Test comparison
    print("\n" + "="*70)
    print("Testing Response Comparison:")
    print("="*70)

    question = "What is the capital of France?"
    response_correct = "Paris"
    response_wrong = "London"

    comparison = rm_trainer.compare_responses(question, response_correct, response_wrong)
    print(f"\nQuestion: {question}")
    print(f"\nResponse A: {response_correct}")
    print(f"Score A: {comparison['score_a']:.4f}")
    print(f"\nResponse B: {response_wrong}")
    print(f"Score B: {comparison['score_b']:.4f}")
    print(f"\nP(A preferred over B): {comparison['prob_a_preferred']:.4f}")

    # Save reward config
    print("\nSaving reward model configuration...")
    rm_trainer.save_model("verifier_reward_config.pt")

    # ========================================================================
    # Policy Training with REINFORCE + Verifier Rewards
    # ========================================================================
    print("\n" + "="*70)
    print("REINFORCE Policy Training with Verifier Rewards")
    print("="*70)

    # Get the reward model
    reward_model = rm_trainer.model
    reward_model.eval()

    # Configuration for RL training
    # OPTION 1: Use base Qwen2-7B model
    # OPTION 2: Use fine-tuned HotpotQA model (recommended for better baseline)
    # OPTION 3: Resume from checkpoint (overrides above options)
    USE_FINETUNED_MODEL = True  # Set to True to use finetuned instruct model

    if resume_from_checkpoint:
        # When resuming, load from checkpoint directory
        policy_model = resume_from_checkpoint
        print(f"\nResuming from checkpoint: {policy_model}")
    elif USE_FINETUNED_MODEL:
        policy_model = "Qwen/Qwen2.5-7B-Instruct"  # Official Qwen base instruct model
    else:
        policy_model = "Qwen/Qwen2.5-7B"  # Base model (no instruction tuning)

    rl_config = SimpleRLConfig(
        policy_model_name=policy_model,
        learning_rate=5e-7,        # ← Reduced from 1e-5
        batch_size=8,              # ← Increased from 2 to 8 for better throughput
        gradient_accumulation_steps=4,  # ← Effective batch size = 8 * 4 = 32
        num_epochs=1,
        max_new_tokens=20,
        temperature=1.0,           # ← Using explicit prompting + 40% bootstrap instead of high temp
        kl_penalty=0.0        # ← Set to 0.0 to disable reference policy and save ~11GB GPU memory
    )

    # Initialize RL trainer
    print(f"\nInitializing REINFORCE trainer...")
    print(f"Device: {rl_config.device}")
    print(f"Policy Model: {rl_config.policy_model_name}")
    if rl_config.kl_penalty > 0.0:
        print(f"KL Penalty: {rl_config.kl_penalty} (prevents policy drift from reference)")
    else:
        print(f"KL Penalty: DISABLED (saves ~11GB GPU memory by not loading reference policy)")
    rl_trainer = SimpleRLTrainer(rl_config, reward_model)

    # Load checkpoint state if resuming
    if resume_from_checkpoint:
        rl_trainer.load_checkpoint(resume_from_checkpoint)

    # Load normal HotpotQA dataset (first 10,000 samples)
    print("\nLoading normal HotpotQA training data (first 10,000 samples)...")
    from datasets import load_dataset

    hotpotqa_dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
    hotpotqa_dataset = hotpotqa_dataset.select(range(10000))
    print(f"Loaded {len(hotpotqa_dataset)} samples from normal HotpotQA")

    # Load abstention dataset to get "I don't know" labels
    print("Loading abstention dataset for labels...")
    abstention_dataset = load_dataset("fsiddiqui2/hotpotqa-abstention-10k", split="train")
    abstention_dataset = abstention_dataset.select(range(10000))
    print(f"Loaded {len(abstention_dataset)} samples from abstention dataset")

    # Debug: Check what fields the abstention dataset has
    print("\nAbstention dataset fields:", abstention_dataset.column_names)
    print("First abstention example:")
    print(abstention_dataset[0])

    # Verify that questions match between the two datasets
    print("\nVerifying dataset alignment (checking first 100 samples)...")
    mismatches = 0
    sample_check_size = min(100, len(hotpotqa_dataset), len(abstention_dataset))

    for i in range(sample_check_size):
        hotpot_q = hotpotqa_dataset[i]['question'].strip()
        abstention_q = abstention_dataset[i]['question'].strip()

        if hotpot_q != abstention_q:
            mismatches += 1
            if mismatches <= 3:  # Print first 3 mismatches as examples
                print(f"  Mismatch at index {i}:")
                print(f"    HotpotQA: {hotpot_q[:80]}...")
                print(f"    Abstention: {abstention_q[:80]}...")

    if mismatches > 0:
        mismatch_rate = (mismatches / sample_check_size) * 100
        print(f"\n⚠️  WARNING: {mismatches}/{sample_check_size} questions don't match ({mismatch_rate:.1f}%)")
        print(f"⚠️  The datasets may not be aligned!")
        print(f"⚠️  This could lead to incorrect abstention labels being applied to wrong questions.")

        # Ask user if they want to continue
        user_input = input("\nContinue training anyway? (yes/no): ").strip().lower()
        if user_input != "yes":
            print("Training aborted by user.")
            return
    else:
        print(f"✓ All {sample_check_size} checked questions match perfectly!")
        print(f"✓ Datasets are properly aligned.")

    # Create training data by combining HotpotQA questions with abstention labels
    training_data = []
    for idx, (hotpot_example, abstention_example) in enumerate(zip(hotpotqa_dataset, abstention_dataset)):
        # Get question and answer from normal HotpotQA
        question = hotpot_example['question']
        hotpot_answer = hotpot_example['answer']

        # Extract context from HotpotQA for verification
        context_titles = hotpot_example['context']['title']
        context_sentences = hotpot_example['context']['sentences']

        # Format context as evidence text
        evidence_parts = []
        for title, sentences in zip(context_titles, context_sentences):
            evidence_parts.append(f"{title}: {' '.join(sentences)}")
        evidence = "\n".join(evidence_parts)

        # Get abstention label from abstention dataset
        # was_correct=False means the model should abstain (short_target is "I don't know")
        is_abstention_label = not abstention_example['was_correct']

        # Simple prompt for base model
        prompt = f"""Question: {question}
Answer:"""

        # Store as tuple: (prompt, hotpot_answer, evidence, is_abstention_label)
        # is_abstention_label tells us if the model SHOULD abstain (from abstention dataset)
        training_data.append((prompt, hotpot_answer, evidence, is_abstention_label))

    # Count abstention labels for statistics
    abstention_label_count = sum(1 for _, _, _, is_abstention in training_data if is_abstention)
    print(f"\nCreated {len(training_data)} training samples")
    print(f"  - Should abstain (abstention dataset says 'I don't know'): {abstention_label_count} ({100*abstention_label_count/len(training_data):.1f}%)")
    print(f"  - Should answer (abstention dataset has real answer): {len(training_data) - abstention_label_count} ({100*(len(training_data) - abstention_label_count)/len(training_data):.1f}%)")

    # Train policy with REINFORCE
    print(f"\nTraining policy with REINFORCE + abstention learning ({len(training_data)} samples)...")
    print(f"Training setup:")
    print(f"  - Model: {rl_config.policy_model_name} (instruct/finetuned model)")
    print(f"  - Training data: Normal HotpotQA questions + answers")
    print(f"  - NEW REWARD STRUCTURE (learn to abstain when uncertain):")
    print(f"    • Abstention: 0.0 (neutral)")
    print(f"    • Wrong answer: -10.0 (huge penalty)")
    print(f"    • Correct answer: factuality-based (positive)")
    print(f"  - Model learns: Better to abstain (0) than be wrong (-10)")
    print(f"  - KL Penalty: β={rl_config.kl_penalty} (prevents catastrophic forgetting)")
    rl_metrics = rl_trainer.train(training_data)

    print(f"\nTraining Results:")
    print(f"Final RL Loss: {rl_metrics['losses'][-1]:.4f}")
    print(f"Final Avg Reward: {rl_metrics['rewards'][-1]:.4f}")
    print(f"Final Avg Factuality: {rl_metrics['factuality'][-1]:.4f}")
    print(f"Final Avg Confidence: {rl_metrics['confidence'][-1]:.4f}")

    # Test generation from trained policy
    print("\n" + "="*70)
    print("Testing Trained Policy:")
    print("="*70)

    test_questions = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is photosynthesis?"
    ]

    for question in test_questions:
        response = rl_trainer.generate_response(question)
        print(f"\nQ: {question}")
        print(f"A: {response}")

        # Score the response
        reward = rm_trainer.score_text(question, response)
        print(f"Reward: {reward:.4f}")

    # Save trained policy (LoRA adapters only)
    print("\n" + "="*70)
    print("Saving trained policy...")
    rl_trainer.save_policy("verifier_rlhf_policy")

    print("\nMerging LoRA adapters with base model...")
    rl_trainer.merge_and_save_full_model(
         adapter_path="verifier_rlhf_policy",
         output_path="verifier_rlhf_full_model",
         push_to_hub=True,
         repo_id="jxrma/Qwen2.5-7B-Instruct-RLHF-Abstention-v2"
 )

    print("\n" + "="*70)
    print("Supervised Abstention RLHF Training Complete!")
    print("="*70)
    print("\nSummary:")
    print(f"- Approach: Supervised abstention learning with explicit labels")
    print(f"- Reward: +1.0 for correct action, -1.0 for wrong action, factuality-based for answers")
    print(f"- Loss: REINFORCE + KL Divergence Penalty (β={rl_config.kl_penalty})")
    print(f"- Final Avg Reward: {rl_metrics['rewards'][-1]:.4f}")
    print(f"- Final Avg Factuality: {rl_metrics['factuality'][-1]:.4f}")
    print(f"- Final Avg Confidence: {rl_metrics['confidence'][-1]:.4f}")
    print(f"- Reward config saved to: verifier_reward_config.pt")
    print(f"- LoRA adapters saved to: verifier_rlhf_policy/")
    print("="*70)


if __name__ == "__main__":
    import glob

    # Auto-detect latest checkpoint for resuming
    checkpoint_dir = "checkpoints"
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
            print("CHECKPOINT DETECTED")
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