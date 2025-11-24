"""
Reward Learning RLHF Implementation
====================================
This module implements reward-based RL (REINFORCE/GRPO style) for factuality optimization.

Reward Components:
1. Verifier factuality score: f = p_ent - p_cont ∈ [-1, 1]
2. Entropy-based confidence: conf = 1 - H_normalized ∈ [0, 1]
3. Final reward: R(x,y) = f * (λ_base + λ_conf * conf)

This is NOT DPO - we use a scalar reward signal with policy gradient methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Dict, Optional
import numpy as np


class FactualityVerifier:
    """
    NLI-based verifier for computing factuality scores.
    Uses DeBERTa-v3-large trained on MNLI, FEVER, ANLI, etc.
    """

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # NLI label mapping: typically 0=entailment, 1=neutral, 2=contradiction
        # Verify this for your specific model
        self.label_map = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }

    @torch.no_grad()
    def compute_factuality_score(
        self,
        evidences: List[str],
        answers: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute factuality scores using NLI verifier.

        Args:
            evidences: List of evidence texts (premises)
            answers: List of generated answers (hypotheses)

        Returns:
            p_ent: Entailment probabilities [B]
            p_cont: Contradiction probabilities [B]
            f: Factuality scores = p_ent - p_cont ∈ [-1, 1] [B]
        """
        batch_size = len(evidences)
        assert len(answers) == batch_size, "Evidences and answers must have same length"

        # Prepare inputs: evidence is premise, answer is hypothesis
        inputs = self.tokenizer(
            evidences,
            answers,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Get predictions
        outputs = self.model(**inputs)
        logits = outputs.logits  # [B, 3]
        probs = F.softmax(logits, dim=-1)  # [B, 3]

        # Extract probabilities
        p_ent = probs[:, self.label_map["entailment"]]      # [B]
        p_neutral = probs[:, self.label_map["neutral"]]     # [B]
        p_cont = probs[:, self.label_map["contradiction"]]  # [B]

        # Compute signed factuality score
        f = p_ent - p_cont  # ∈ [-1, 1]

        return p_ent, p_cont, f


class EntropyConfidenceCalculator:
    """
    Computes entropy-based confidence from model logits.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.H_max = np.log(vocab_size)  # Maximum possible entropy

    def compute_confidence(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute confidence from logits via normalized entropy.

        Args:
            logits: Token logits [B, T, V]
            mask: Optional attention mask [B, T] (1 for real tokens, 0 for padding)

        Returns:
            conf: Confidence scores ∈ [0, 1] [B]
                  conf ≈ 1 → peaked distribution (confident)
                  conf ≈ 0 → uniform distribution (uncertain)
        """
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)  # [B, T, V]

        # Compute per-token entropy: -∑_v p_v log(p_v)
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        entropy_per_token = -(probs * log_probs).sum(dim=-1)  # [B, T]

        # Apply mask if provided
        if mask is not None:
            entropy_per_token = entropy_per_token * mask  # Zero out padding
            seq_lengths = mask.sum(dim=-1)  # [B]
        else:
            seq_lengths = torch.full(
                (logits.size(0),),
                logits.size(1),
                device=logits.device
            )

        # Average entropy per sequence
        H = entropy_per_token.sum(dim=-1) / seq_lengths.clamp(min=1)  # [B]

        # Normalize by max entropy
        H_normalized = H / self.H_max  # ∈ [0, 1]

        # Confidence = 1 - normalized_entropy
        conf = 1.0 - H_normalized  # [B]

        return conf


class RewardFunction(nn.Module):
    """
    Combined reward function for factuality + confidence + abstention.

    Base Reward:
        R(x,y) = f * (λ_base + λ_conf * conf)

    With Anti-Hallucination:
        - Abstention bonus: +abstention_bonus when model says "I don't know" and conf < threshold
        - Hallucination penalty: -hallucination_penalty when conf < threshold and f < 0

    All weights are LEARNABLE parameters optimized during training.
    """

    def __init__(
        self,
        verifier: FactualityVerifier,
        confidence_calc: EntropyConfidenceCalculator,
        lambda_base_init: float = 0.7,  # 70% weight on factuality
        lambda_conf_init: float = 0.3,  # 30% weight on confidence
        enable_abstention: bool = False,
        abstention_bonus_init: float = 0.5,
        hallucination_penalty_init: float = 1.0,
        confidence_threshold_init: float = 0.4
    ):
        super().__init__()

        self.verifier = verifier
        self.confidence_calc = confidence_calc

        # LEARNABLE reward weights (normalized to sum to 1 via softmax)
        # Store logits that will be softmax-normalized
        self.lambda_logits = nn.Parameter(torch.tensor([
            np.log(lambda_base_init),  # logit for factuality weight
            np.log(lambda_conf_init)   # logit for confidence weight
        ]))

        # Anti-hallucination parameters (learnable)
        self.enable_abstention = enable_abstention
        if self.enable_abstention:
            # Positive rewards/penalties (use softplus to ensure positive)
            self.abstention_bonus_raw = nn.Parameter(torch.tensor(abstention_bonus_init))
            self.hallucination_penalty_raw = nn.Parameter(torch.tensor(hallucination_penalty_init))
            # Confidence threshold: logit to keep in [0, 1] after sigmoid
            self.confidence_threshold_logit = nn.Parameter(
                torch.tensor(np.log(confidence_threshold_init / (1 - confidence_threshold_init + 1e-8)))
            )
        else:
            # Register as non-learnable buffers when disabled
            self.register_buffer('abstention_bonus_raw', torch.tensor(abstention_bonus_init))
            self.register_buffer('hallucination_penalty_raw', torch.tensor(hallucination_penalty_init))
            self.register_buffer('confidence_threshold_logit',
                torch.tensor(np.log(confidence_threshold_init / (1 - confidence_threshold_init + 1e-8))))

        # Abstention phrases to detect
        self.abstention_phrases = [
            "i don't know",
            "i do not know",
            "not sure",
            "uncertain",
            "cannot answer",
            "can't answer",
            "no information",
            "insufficient information"
        ]

    @property
    def lambda_weights(self):
        """Get normalized weights (sum to 1) via softmax."""
        return F.softmax(self.lambda_logits, dim=0)

    @property
    def lambda_base(self):
        """Factuality weight (normalized)."""
        return self.lambda_weights[0]

    @property
    def lambda_conf(self):
        """Confidence weight (normalized)."""
        return self.lambda_weights[1]

    @property
    def abstention_bonus(self):
        """Ensure abstention_bonus stays positive via softplus."""
        return F.softplus(self.abstention_bonus_raw)

    @property
    def hallucination_penalty(self):
        """Ensure hallucination_penalty stays positive via softplus."""
        return F.softplus(self.hallucination_penalty_raw)

    @property
    def confidence_threshold(self):
        """Ensure threshold stays in [0, 1] via sigmoid."""
        return torch.sigmoid(self.confidence_threshold_logit)

    def is_abstention(self, answer: str, question: str) -> bool:
        """
        Check if answer is an abstention using NLI verifier.

        Uses the verifier to check if the answer entails "I don't know" or similar abstention.
        More robust than string matching.

        Args:
            answer: Generated answer text
            question: Original question (used as context)

        Returns:
            Boolean indicating if this is an abstention
        """
        # Abstention hypothesis templates
        abstention_templates = [
            "The answer is unknown",
            "I don't know the answer",
            "There is no answer",
            "I cannot answer this question"
        ]

        # Check if answer entails any abstention template
        with torch.no_grad():
            for template in abstention_templates:
                # NLI: answer (premise) entails template (hypothesis)?
                inputs = self.verifier.tokenizer(
                    [answer],  # premise
                    [template],  # hypothesis
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.verifier.device)

                outputs = self.verifier.model(**inputs)
                logits = outputs.logits  # [1, 3]
                probs = F.softmax(logits, dim=-1)  # [1, 3]

                p_ent = probs[0, self.verifier.label_map["entailment"]].item()

                # High entailment with abstention = it's an abstention
                if p_ent > 0.5:
                    return True

        return False

    def compute_rewards(
        self,
        evidences: List[str],
        answers: List[str],
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full reward signal with optional anti-hallucination features.

        Args:
            evidences: Evidence texts [B]
            answers: Generated answers [B]
            logits: Policy logits [B, T, V]
            mask: Attention mask [B, T]

        Returns:
            Dictionary containing:
                - rewards: Final scalar rewards [B]
                - f: Factuality scores [B]
                - conf: Confidence scores [B]
                - p_ent: Entailment probabilities [B]
                - p_cont: Contradiction probabilities [B]
                - is_abstention: Boolean tensor indicating abstentions [B]
                - is_hallucination: Boolean tensor indicating likely hallucinations [B]
        """
        batch_size = len(answers)

        # 1. Compute factuality score
        p_ent, p_cont, f = self.verifier.compute_factuality_score(evidences, answers)

        # 2. Compute confidence
        conf = self.confidence_calc.compute_confidence(logits, mask)

        # 3. Base reward: R(x,y) = f * (λ_base + λ_conf * conf)
        base_rewards = f * (self.lambda_base + self.lambda_conf * conf)

        rewards = base_rewards.clone()

        # 4. Apply anti-hallucination modifications if enabled
        is_abstention = torch.zeros(batch_size, dtype=torch.bool, device=rewards.device)
        is_hallucination = torch.zeros(batch_size, dtype=torch.bool, device=rewards.device)

        if self.enable_abstention:
            for i, (answer, question) in enumerate(zip(answers, evidences)):
                # Check if this is an abstention using NLI verifier
                abstains = self.is_abstention(answer, question)
                is_abstention[i] = abstains

                # Low confidence indicator
                low_conf = conf[i] < self.confidence_threshold

                # Likely hallucination: low confidence + negative factuality
                likely_halluc = low_conf and f[i] < 0
                is_hallucination[i] = likely_halluc

                if abstains and low_conf:
                    # Reward abstention when uncertain
                    rewards[i] = self.abstention_bonus
                elif likely_halluc and not abstains:
                    # Penalize hallucinations (low conf + wrong answer)
                    rewards[i] = -self.hallucination_penalty

        return {
            "rewards": rewards,
            "f": f,
            "conf": conf,
            "p_ent": p_ent,
            "p_cont": p_cont,
            "is_abstention": is_abstention,
            "is_hallucination": is_hallucination
        }


class REINFORCELoss:
    """
    REINFORCE (policy gradient) loss with baseline and optional KL divergence penalty.

    Loss = -E[(R - b) * log π_θ(y|x)] + β * KL(π_θ || π_ref)

    where:
    - R is the reward
    - b is the baseline (mean reward)
    - π_θ is the current policy
    - π_ref is the reference policy (frozen)
    - β is the KL penalty coefficient
    """

    def __init__(self, baseline_type: str = "batch_mean", kl_penalty: float = 0.0):
        """
        Args:
            baseline_type: Type of baseline to use
                - "batch_mean": Use mean reward in batch
                - "none": No baseline (higher variance)
            kl_penalty: KL divergence penalty coefficient (β)
                - 0.0: No KL penalty (standard REINFORCE)
                - 0.01-0.1: Light penalty (recommended for RLHF)
                - >0.1: Strong penalty (limits policy drift)
        """
        self.baseline_type = baseline_type
        self.kl_penalty = kl_penalty

    def compute_log_probs(
        self,
        logits: torch.Tensor,
        sequences: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute log probabilities of sampled sequences.

        Args:
            logits: Model logits [B, T, V]
            sequences: Sampled token IDs [B, T]
            mask: Attention mask [B, T]

        Returns:
            log_probs: Sum of log probabilities per sequence [B]
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Compute log probabilities
        log_probs_all = F.log_softmax(logits, dim=-1)  # [B, T, V]

        # Gather log probs of sampled tokens
        sequences_expanded = sequences.unsqueeze(-1)  # [B, T, 1]
        log_probs_sampled = torch.gather(
            log_probs_all,
            dim=-1,
            index=sequences_expanded
        ).squeeze(-1)  # [B, T]

        # Apply mask and sum
        if mask is not None:
            log_probs_sampled = log_probs_sampled * mask

        log_probs = log_probs_sampled.sum(dim=-1)  # [B]

        return log_probs

    def compute_loss(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute REINFORCE loss with baseline and optional KL divergence penalty.

        Args:
            rewards: Reward values [B]
            log_probs: Log probabilities of sampled sequences from current policy [B]
            ref_log_probs: Log probabilities from reference policy [B] (optional)

        Returns:
            loss: Scalar loss value
            metrics: Dictionary of diagnostic metrics
        """
        # Compute baseline
        if self.baseline_type == "batch_mean":
            baseline = rewards.mean().detach()
        elif self.baseline_type == "none":
            baseline = 0.0
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")

        # Compute advantages
        advantages = rewards - baseline

        # REINFORCE loss: -E[(R - b) * log π(y|x)]
        pg_loss = -(advantages * log_probs).mean()

        # KL divergence penalty (if enabled and reference provided)
        kl_loss = torch.tensor(0.0, device=log_probs.device)
        if self.kl_penalty > 0.0 and ref_log_probs is not None:
            # KL(π_θ || π_ref) ≈ log π_θ(y|x) - log π_ref(y|x)
            kl_div = (log_probs - ref_log_probs)
            kl_loss = self.kl_penalty * kl_div.mean()

        # Total loss
        loss = pg_loss + kl_loss

        # Metrics for monitoring
        metrics = {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            "mean_kl_div": kl_div.mean().item() if self.kl_penalty > 0.0 and ref_log_probs is not None else 0.0,
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "mean_advantage": advantages.mean().item(),
            "baseline": baseline if isinstance(baseline, float) else baseline.item()
        }

        return loss, metrics


class RewardLearningTrainer:
    """
    Complete training loop for reward learning RLHF.
    """

    def __init__(
        self,
        policy_model,
        tokenizer,
        reward_function: RewardFunction,
        loss_function: REINFORCELoss,
        optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.policy = policy_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_function
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.device = device

    def training_step(
        self,
        questions: List[str],
        evidences: List[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            questions: Input questions [B]
            evidences: Corresponding evidence texts [B]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            metrics: Dictionary of training metrics
        """
        batch_size = len(questions)

        # 1. Tokenize questions
        inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # 2. Generate from policy with logits
        with torch.no_grad():
            outputs = self.policy.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # 3. Extract generated sequences and logits
        sequences = outputs.sequences  # [B, T_total]
        scores = outputs.scores  # Tuple of [B, V] for each generated token

        # Stack scores into tensor [B, T_gen, V]
        logits = torch.stack(scores, dim=1)  # [B, T_gen, V]

        # Get only the generated part (remove prompt)
        prompt_length = inputs.input_ids.shape[1]
        generated_sequences = sequences[:, prompt_length:]  # [B, T_gen]

        # 4. Decode answers
        answers = self.tokenizer.batch_decode(
            generated_sequences,
            skip_special_tokens=True
        )

        # 5. Create attention mask for generated tokens
        gen_mask = (generated_sequences != self.tokenizer.pad_token_id).float()

        # 6. Compute rewards
        reward_info = self.reward_fn.compute_rewards(
            evidences=evidences,
            answers=answers,
            logits=logits,
            mask=gen_mask
        )
        rewards = reward_info["rewards"]

        # 7. Compute log probabilities (need to do forward pass)
        # Re-run forward pass to get logits for loss computation
        input_ids_full = sequences
        with torch.enable_grad():
            # Forward pass through policy
            policy_outputs = self.policy(
                input_ids=input_ids_full,
                attention_mask=(input_ids_full != self.tokenizer.pad_token_id).long()
            )
            full_logits = policy_outputs.logits  # [B, T_total, V]

            # Extract logits for generated tokens only
            gen_logits = full_logits[:, prompt_length-1:-1, :]  # Shift by 1 for autoregressive

            # Compute log probs
            log_probs = self.loss_fn.compute_log_probs(
                gen_logits,
                generated_sequences,
                gen_mask
            )

        # 8. Compute loss
        loss, loss_metrics = self.loss_fn.compute_loss(rewards, log_probs)

        # 9. Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 10. Combine all metrics
        metrics = {
            **loss_metrics,
            "mean_factuality": reward_info["f"].mean().item(),
            "mean_confidence": reward_info["conf"].mean().item(),
            "mean_p_ent": reward_info["p_ent"].mean().item(),
            "mean_p_cont": reward_info["p_cont"].mean().item(),
        }

        return metrics


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Initialize components
    print("Initializing verifier...")
    verifier = FactualityVerifier()

    print("Initializing policy model...")
    policy_name = "gpt2"  # Replace with your model
    policy = AutoModelForCausalLM.from_pretrained(policy_name)
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_name)
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

    # Confidence calculator
    confidence_calc = EntropyConfidenceCalculator(
        vocab_size=policy_tokenizer.vocab_size
    )

    # Reward function
    reward_fn = RewardFunction(
        verifier=verifier,
        confidence_calc=confidence_calc,
        lambda_base=0.5,
        lambda_conf=0.5
    )

    # Loss function
    loss_fn = REINFORCELoss(baseline_type="batch_mean")

    # Optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)

    # Trainer
    trainer = RewardLearningTrainer(
        policy_model=policy,
        tokenizer=policy_tokenizer,
        reward_function=reward_fn,
        loss_function=loss_fn,
        optimizer=optimizer
    )

    # Example training step
    print("\nRunning example training step...")
    questions = [
        "What is the capital of France?",
        "Who invented the telephone?"
    ]
    evidences = [
        "Paris is the capital and most populous city of France.",
        "Alexander Graham Bell was awarded the first U.S. patent for the invention of the telephone in 1876."
    ]

    metrics = trainer.training_step(questions, evidences)

    print("\nTraining metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
