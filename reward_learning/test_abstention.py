"""
Quick test to verify abstention detection works correctly
"""
import torch
from reward_and_loss import FactualityVerifier, AbstentionClassifier

# Initialize verifier
print("Loading NLI verifier...")
verifier = FactualityVerifier(device="cuda" if torch.cuda.is_available() else "cpu")
abstention_classifier = AbstentionClassifier(verifier)

# Test answers
test_answers = [
    "I don't know",
    "I don't know the answer",
    "I do not know",
    "I'm not sure",
    "I cannot answer this question",
    "Insufficient information",
    "The answer is unknown",
    "Paris",  # Normal answer
    "The capital is Paris",  # Normal answer
    "I think it's Paris but I'm not certain",  # Uncertain but still answering
    "42",  # Normal answer
    "It's unclear",  # Edge case
    "I'm uncertain about this",  # Edge case
    "No information available",  # Edge case
]

print("\nTesting abstention detection:\n")
print(f"{'Answer':<50} {'Score':<10} {'Abstention?':<12}")
print("=" * 72)

# Get abstention scores
abstention_scores = abstention_classifier.predict_proba(test_answers)

for answer, score in zip(test_answers, abstention_scores):
    is_abstention = score.item() >= 0.5
    print(f"{answer:<50} {score.item():<10.4f} {'YES' if is_abstention else 'NO':<12}")

print("\n" + "=" * 72)
print(f"\nThreshold: 0.5")
print(f"Abstentions detected: {(abstention_scores >= 0.5).sum().item()}/{len(test_answers)}")
print(f"Abstention rate: {(abstention_scores >= 0.5).float().mean().item():.2%}")
