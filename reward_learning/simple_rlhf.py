"""
Simple RLHF with Factuality-Only Rewards
=========================================

Minimal RLHF implementation using ONLY factuality from NLI verifier.
No confidence, no abstention - just pure factuality scoring.

Reward: R(x,y) = f = p_entailment - p_contradiction

Usage:
    python simple_rlhf.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from datasets import load_dataset
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass


################################################################################
# Factuality Verifier (NLI-based)
################################################################################

class FactualityVerifier:
    """
    Simple NLI-based verifier for factuality scoring.
    Uses DeBERTa-FEVER to compute: f = p_ent - p_cont
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

        self.label_map = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }

    @torch.no_grad()
    def compute_factuality(
        self,
        gold_answers: List[str],
        generated_answers: List[str]
    ) -> torch.Tensor:
        """
        Compute factuality scores by checking if gold answer entails generated answer.

        Args:
            gold_answers: List of gold/reference answers (premise) [B]
            generated_answers: List of generated answers (hypothesis) [B]

        Returns:
            factuality: Scores f = p_ent - p_cont ∈ [-1, 1] [B]
        """
        # Tokenize (gold_answer, generated_answer) pairs
        # NLI: Does gold answer (premise) entail generated answer (hypothesis)?
        inputs = self.tokenizer(
            gold_answers,  # premise
            generated_answers,  # hypothesis
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Get NLI predictions
        outputs = self.model(**inputs)
        logits = outputs.logits  # [B, 3]
        probs = F.softmax(logits, dim=-1)  # [B, 3]

        # Extract probabilities
        p_ent = probs[:, self.label_map["entailment"]]
        p_cont = probs[:, self.label_map["contradiction"]]

        # Factuality score
        factuality = p_ent - p_cont  # [-1, 1]

        return factuality


################################################################################
# REINFORCE Loss
################################################################################

class REINFORCELoss:
    """Simple REINFORCE loss with batch mean baseline."""

    def compute_loss(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute REINFORCE loss.

        Args:
            rewards: Reward values [B]
            log_probs: Log probabilities [B]

        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        # Baseline = mean reward
        baseline = rewards.mean().detach()

        # Advantages
        advantages = rewards - baseline

        # REINFORCE: -E[(R - b) * log π(y|x)]
        loss = -(advantages * log_probs).mean()

        metrics = {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item()
        }

        return loss, metrics


################################################################################
# RLHF Trainer
################################################################################

@dataclass
class RLHFConfig:
    """Configuration for RLHF training."""
    policy_model: str = "Qwen/Qwen2-7B"
    verifier_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    learning_rate: float = 1e-5
    batch_size: int = 2
    num_epochs: int = 1
    max_length: int = 512
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.9
    num_train_samples: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleRLHFTrainer:
    """
    Simple RLHF trainer with factuality-only rewards.

    Reward: R(x,y) = f = p_ent - p_cont
    """

    def __init__(self, config: RLHFConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load policy model
        print(f"\nLoading policy: {config.policy_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.policy_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy = AutoModelForCausalLM.from_pretrained(
            config.policy_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        # Load verifier
        print(f"Loading verifier: {config.verifier_model}...")
        self.verifier = FactualityVerifier(
            model_name=config.verifier_model,
            device=config.device
        )

        # Loss function
        self.loss_fn = REINFORCELoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate
        )

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int
    ) -> torch.Tensor:
        """
        Compute log π(y|x) for generated sequences.

        Args:
            input_ids: [B, T] full sequence
            attention_mask: [B, T]
            prompt_len: Length of prompt

        Returns:
            log_probs: [B] summed log probs
        """
        # Forward pass
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits[:, :-1, :]  # [B, T-1, V]

        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T-1, V]

        # Get actual next tokens
        next_ids = input_ids[:, 1:]  # [B, T-1]

        # Gather log probs
        token_log_probs = log_probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        # Mask to only response tokens
        mask = torch.zeros_like(token_log_probs)
        mask[:, prompt_len-1:] = 1.0

        # Sum over response
        response_log_probs = (token_log_probs * mask).sum(dim=1)  # [B]

        return response_log_probs

    def train_step(self, batch_data: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch_data: List of (question, gold_answer) tuples

        Returns:
            metrics: Training metrics
        """
        self.policy.train()

        # Unpack batch
        questions = [q for q, _ in batch_data]
        gold_answers = [a for _, a in batch_data]

        # 1. Tokenize questions
        encoded = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_len = input_ids.size(1)

        # 2. Generate responses
        with torch.no_grad():
            gen_outputs = self.policy.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # 3. Decode generated answers
        generated_ids = gen_outputs[:, prompt_len:]
        generated_answers = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        # 4. Compute factuality rewards (gold answer vs generated answer)
        with torch.no_grad():
            rewards = self.verifier.compute_factuality(gold_answers, generated_answers)

        # 5. Compute log probabilities
        gen_attention = (gen_outputs != self.tokenizer.pad_token_id).long()
        log_probs = self.compute_log_probs(gen_outputs, gen_attention, prompt_len)

        # 6. REINFORCE loss
        loss, loss_metrics = self.loss_fn.compute_loss(rewards, log_probs)

        # 7. Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_metrics

    def train(self, training_data: List[Tuple[str, str]]) -> Dict[str, List[float]]:
        """
        Train on question-answer pairs.

        Args:
            training_data: List of (question, gold_answer) tuples

        Returns:
            metrics: Training history
        """
        dataloader = DataLoader(
            training_data,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        losses = []
        rewards = []

        print("\n" + "="*70)
        print("SIMPLE RLHF TRAINING (Factuality-Only)")
        print("="*70)

        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            epoch_rewards = []

            for step, batch in enumerate(dataloader):
                metrics = self.train_step(batch)

                epoch_losses.append(metrics["loss"])
                epoch_rewards.append(metrics["mean_reward"])

                if step % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.config.num_epochs}, Step {step}")
                    print(f"  Loss: {metrics['loss']:.4f}")
                    print(f"  Reward: {metrics['mean_reward']:.4f}")

            avg_loss = np.mean(epoch_losses)
            avg_reward = np.mean(epoch_rewards)

            losses.append(avg_loss)
            rewards.append(avg_reward)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Avg Reward: {avg_reward:.4f}")

        return {
            "losses": losses,
            "rewards": rewards
        }

    def save_policy(self, path: str):
        """Save trained policy."""
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"\nPolicy saved to {path}")


################################################################################
# Main
################################################################################

def main():
    print("="*70)
    print("Simple RLHF with Factuality-Only Rewards")
    print("="*70)

    # Configuration
    config = RLHFConfig(
        policy_model="Qwen/Qwen2-7B",
        num_train_samples=100,
        batch_size=2,
        num_epochs=1
    )

    # Load HotpotQA data
    print(f"\nLoading HotpotQA ({config.num_train_samples} samples)...")
    dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
    dataset = dataset.select(range(config.num_train_samples))

    training_data = []
    for example in dataset:
        question = example['question']
        gold_answer = example['answer']

        # Simple prompt format
        prompt = f"Question: {question}\nAnswer:"
        training_data.append((prompt, gold_answer))

    print(f"Loaded {len(training_data)} question-answer pairs from HotpotQA")

    # Initialize trainer
    trainer = SimpleRLHFTrainer(config)

    # Train
    metrics = trainer.train(training_data)

    # Save
    trainer.save_policy("./simple_rlhf_policy")

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Final Loss: {metrics['losses'][-1]:.4f}")
    print(f"Final Reward: {metrics['rewards'][-1]:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
