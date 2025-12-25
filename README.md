# A Reasonable Approach to Hallucination Mitigation on HotPotQA

**Columbia COMS4705 Final Project**

**Authors:** Joshua Hegstad, Ahmed Jaber, Farhaan Siddiqui
**Mentor:** Melody Ma
**Keywords:** hallucination, Q&A, reinforcement learning, decoding, abstention

## Abstract

Closed-book question answering (QA) systems must answer questions using only parametric knowledge, which makes them prone to **closed-book hallucination**: confident but factually incorrect answers with no supporting evidence. We view this behavior as arising from two failure modes:

1. **No Knowledge**: The model lacks the relevant knowledge
2. **Cannot Access Knowledge**: The knowledge is implicitly present in its parameters, but the model does not reliably retrieve or apply it at inference time

We study these failure modes on HotPotQA using Qwen2.5-7B and Qwen3-8B and propose a three-stage mitigation pipeline.

To address the "cannot access knowledge" case, we perform Chain-of-Thought (CoT) distillation from a larger Qwen3-235B teacher to unlock latent parametric knowledge. To address the "no knowledge" case, we train the model to abstain by replacing incorrect predictions with "I don't know" and finetuning on a class-balanced mix of correct answers and abstentions. Finally, we apply Reinforcement Learning from Verifier Feedback (RLVF) with an NLI-derived factuality score to heavily penalize confident errors. On HotPotQA, this pipeline reduces hallucinated responses and increases appropriate refusals. Furthermore, we demonstrate that unsupervised Semantic Entropy (SE) effectively flags residual hallucinations, enabling a consensus-based rejection strategy that improves selective F1 score from 0.50 to 0.70.

## Key Results

- **CoT Distillation**: Improved Exact Match accuracy from 18.86% to 24.31% on HotPotQA
- **Abstention Precision**: Achieved 97.5% precision in detecting knowledge boundaries
- **RLVF**: Successfully tuned the risk-coverage trade-off, increasing answer rate by ~5% while maintaining safety
- **Semantic Entropy**: Validated as a robust proxy for correctness, enabling efficient rejection of hallucinations

## Approach

### 1. Chain-of-Thought (CoT) Distillation

To address the "cannot access knowledge" failure mode, we employ CoT distillation from a large teacher model (Qwen3-235B) to our smaller student model (Qwen2.5-7B).

**Key Innovation:** We constrain the teacher to generate "internalized knowledge" reasoning traces that frame answers as step-by-step recall of facts, explicitly forbidding citation markers. This ensures the student learns to function in a closed-book setting.

**Example:**
```
Q: Which magazine was started first Arthur's Magazine or First for Women?

R: Step 1: The question asks which magazine was started first between
   Arthur's Magazine and First for Women.
   Step 2: Arthur's Magazine was published starting in 1844, while First
   for Women was launched in the 1980s.
   Step 3: Since 1844 is earlier than the 1980s, Arthur's Magazine was
   started first.

A: Arthur's Magazine
```

### 2. Abstention-Aware Fine-Tuning

To mitigate the "no knowledge" failure mode, we teach the model to explicitly output "I don't know" when it lacks internal knowledge.

**Process:**
1. Map the knowledge boundary by running inference on 40,000 HotPotQA training examples
2. Partition data into:
   - **Correct Set**: Samples where the model's internal knowledge was sufficient
   - **Hallucination Set**: Samples where the model generated incorrect answers
3. Create Dataset C with 80% correct answers and 20% "I don't know" responses
4. Fine-tune using QLoRA to prevent "laziness" (defaulting to constant abstention)

### 3. Reinforcement Learning from Verifier Feedback (RLVF)

To refine the abstention boundary learned during supervised fine-tuning, we train the Qwen2.5-Abstain model with policy-gradient RL using a reward signal derived entirely from verifier factuality and model confidence.

**Verifier Architecture:** Although inference is closed-book, the reward model is not. The **FactualityVerifier**—a DeBERTa-based NLI model—receives the full HotPotQA supporting paragraphs. For each example, we concatenate all supporting titles and sentences into a single premise and compute:
- p_ent: Entailment probability
- p_cont: Contradiction probability
- f = p_ent - p_cont: Grounded factuality score

Thus RLVF optimizes semantic factuality rather than gold-label EM.

**Confidence Term:** We include a normalized entropy confidence score:
```
conf = 1 - (H_avg / H_max)
```

**Abstention Detection:** Abstentions are not identified via string matching. Instead, the **AbstentionClassifier** applies the same NLI verifier to check whether the model's answer entails an abstention-style statement (e.g., "I don't know"). If so, the model receives a fixed penalty of -1.0.

**Reward Function:** For non-abstaining outputs, reward depends solely on verifier factuality and confidence:
```
R = {
  10 × f × (λ_base + λ_conf × conf)   if f ≥ 0
  5 × f × (λ_base + λ_conf × conf)    if f < 0
}
```

Positive factuality is up-weighted (10×), negative factuality down-weighted (5×).

### 4. Uncertainty Quantification with Semantic Entropy

To estimate SE without access to token probabilities, we sample M=10 stochastic generations per question and cluster them based on semantic equivalence using a bidirectional NLI entailment model (microsoft/deberta-large-mnli). The probability of each semantic cluster C is approximated via Monte Carlo integration:

```
P(C|x) ≈ (1/M) Σ I(s_i ∈ C)
```

The total uncertainty is then given by the entropy over these semantic clusters:

```
H(x) = -Σ P(C|x) log P(C|x)
```

This metric serves as our primary signal for evaluating the efficacy of our abstention mechanism in calculating and rejecting hallucinations.

We also examined both **Standard Semantic Entropy (Std SE)**, which measures total uncertainty over all clusters, and **Conditional Semantic Entropy (Cond SE)**, which normalizes uncertainty by ignoring the probability mass assigned to the "I don't know" cluster.

## Models

| Model | Description |
|-------|-------------|
| **Qwen2.5-Instruct** | Base Qwen2.5-7B-Instruct model |
| **Qwen2.5-SFT** | Fine-tuned on 10,000 Q&A pairs |
| **Qwen2.5-FCoT** | Fine-tuned on CoT reasoning traces from Qwen3-235B |
| **Qwen2.5-Abstain** | Fine-tuned on Q&A/abstention pairs (80/20 split) |
| **Qwen2.5-RLVF** | RLVF-trained model with verifier-based rewards |

## Performance Results

### Generation Performance

| Model | EM | F1 | Corrections | Regressions |
|-------|----|----|-------------|-------------|
| Qwen2.5-Instruct | 18.87% | 27.02% | - | - |
| Qwen2.5-SFT | 21.94% | 31.41% | 477 | 249 |
| **Qwen2.5-FCoT** | **24.31%** | **33.87%** | **697** | 294 |

### Abstention Performance

| Model | EM | F1 | Abs. Rate | Selective EM | Selective F1 |
|-------|----|----|-----------|--------------|--------------|
| Qwen2.5-Abstain | 19.47% | 25.78% | 46.34% | **37.98%** | **48.46%** |
| Qwen2.5-RLVF | 20.37% | 26.24% | **41.17%** | 34.62% | 44.60% |

**Key Insight:** The supervised abstention model achieves higher selective accuracy (37.98% vs 34.62%) but lower coverage (53.65% vs 58.83%). RLVF trades some selective accuracy for increased answer rate.

### Abstention Precision & Recall

The supervised model (Qwen2.5-Abstain) achieves **97.49% precision** against the base model, meaning its abstentions are almost invariably justified. The RLVF model maintains high precision (97.84%) while increasing coverage.

### Abstention Confusion Matrices

**Supervised Fine-Tuning (SFT) Model:**
![Abstention SFT Confusion Matrices](./images/abstention_confusion_matrices_SFT.png)

**RLHF Model:**
![RLHF Confusion Matrices](./images/rlhf_confusion_matrices.png)

### Semantic Entropy Analysis

- **Correct answers**: Heavily clustered around SE ≈ 0 (low uncertainty)
- **Incorrect answers**: Median SE ≈ 0.9 (high uncertainty)
- **Rejection at 50%**: Consensus strategy achieves Selective F1 of 0.69 (up from 0.50 at 0% rejection)

### RAUQ Rejection Curves

**Vanilla Qwen2.5-7B:**
![RAUQ Rejection Curve - Vanilla](./images/rauq_rejection_curve_vanilla.png)

**Fine-tuned Model:**
![RAUQ Rejection Curve - Finetuned](./images/rauq_rejection_curve_finetuned.png)

## Dataset

We use the **HotPotQA distractor dataset** in a strictly closed-book setting (all context paragraphs removed).

### Training Datasets

- **Dataset A1/A2**: The first and second 10,000 question-answer pairs of HotPotQA (distractor setting, train split), respectively
- **Dataset B**: 10,000 Q&A pairs with CoT reasoning traces generated by Qwen3-235B according to our "Finetune-CoT" approach
- **Dataset C**: 10,000 Q&A/abstention pairs (80/20 split) from the first 40,000 samples of HotPotQA based on model's knowledge boundary

## Training Configuration

### QLoRA Configuration
- **Rank**: r=16, α=32
- **Dropout**: 0.05
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Quantization**: 4-bit NF4 with double quantization

### Optimization
- **Optimizer**: AdamW (adamw_torch)
- **Learning rate**: 2e-4 (SFT, FCoT, Abstain), 1e-5 (RLVF)
- **Scheduler**: Cosine with 3% warmup
- **Batch size**: Effective batch size of 16 (4 per device × 4 gradient accumulation)
- **Training duration**: 1 epoch for all models

### Training Data Assignment
- **Qwen2.5-SFT**: Trained on Dataset A1
- **Qwen2.5-FCoT**: Trained on Dataset B
- **Qwen2.5-Abstain**: Trained on Dataset C
- **Qwen2.5-RLVF**: Initialized from Qwen2.5-Abstain checkpoint, trained on Dataset A2 with on-policy generation

### Inference
- **Decoding**: Greedy (deterministic)
- **Max tokens**: 50 (standard), 512 (CoT)
- **Precision**: bfloat16

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Ajaberr/HallucinationMitigationHotpotQA.git
cd HallucinationMitigationHotpotQA

# Install dependencies
pip install -r requirements.txt

# Run evaluation
python eval.py --model qwen2.5-fcot --dataset hotpotqa

# Run with abstention
python eval.py --model qwen2.5-abstain --enable-abstention

# Run with RLVF
python eval.py --model qwen2.5-rlvf --enable-abstention
```

## Key Findings

### Successes

1. **CoT Distillation Works**: Outperformed standard SFT by correcting 697 baseline errors vs 477 for SFT
2. **Abstention is Learnable**: 97.5% precision in detecting knowledge boundaries
3. **Semantic Entropy is Reliable**: Strong correlation with correctness (median SE: 0 for correct, 0.9 for incorrect)
4. **RLVF Enables Trade-off Tuning**: Successfully balanced coverage vs safety

### Limitations

1. **CoT Introduces New Errors**: 294 regressions where base model was correct
   - Hallucination via reasoning (fabricating specific details)
   - Granularity mismatch (e.g., "film director" vs "director")
2. **RLVF Trade-off**: Higher coverage comes at cost of selective accuracy
3. **Computational Cost**: Semantic Entropy requires 10 generations per question

### Qualitative Examples

**Success Case (Base Wrong → CoT Correct):**
```
Q: Were Scott Derrickson and Ed Wood of the same nationality?
Gold: yes
Base Prediction: No ❌
CoT Prediction: yes ✓

CoT Reasoning:
Step 1: The question asks about the nationalities of Scott Derrickson
        and Ed Wood.
Step 2: Scott Derrickson is an American film director... while Ed Wood
        was an American filmmaker...
Step 3: Since both individuals were born and active in the United States...
        the answer is yes.
```

**RLVF vs SFT Abstention:**
```
Q: What government position was held by the woman who portrayed Corliss
   Archer in the film Kiss and Tell?

SFT Model: "I don't know" (conservative, safe)
RLVF Model: "First Lady" ❌ (attempted answer, hallucination)
Gold: "Chief of Protocol"

---

Q: What army did the namesake of the ship launched as the München in 1930
   fight in during the American Revolutionary War?

SFT Model: "I don't know" (over-conservative)
RLVF Model: "Continental Army" ✓ (productive risk-taking)
Gold: "Continental Army"
```

## Future Work

1. **Lightweight Uncertainty Filters**: Cascade RAUQ as a pre-filter for Semantic Entropy to reduce inference latency
2. **Open-Book Transfer**: Assess whether calibration gains transfer when discriminating between parametric knowledge and retrieved context
3. **Improved Reward Modeling**: Explore more sophisticated confidence modeling to distinguish informative risk-taking from unreliable guessing

## Citation

```bibtex
@article{hegstad2024hallucination,
  title={Hallucination Mitigation in Closed-Book Question Answering via Chain-of-Thought Distillation and Abstention-Aware Fine-Tuning},
  author={Hegstad, Joshua and Jaber, Ahmed and Siddiqui, Farhaan},
  journal={Columbia COMS4705 Final Project},
  year={2024}
}
```

## Team Contributions

Joshua wrote the code for training and evaluating the Tinker models and running semantic entropy evaluations, and wrote all analysis thereof.

Farhaan wrote code for training and evaluating the Qwen2.5 finetuned models, including generating the CoT and Abstention datasets and analyzing the reasoning traces.

Ahmed implemented the Reward Learning (RLVF) training and evaluation pipeline, developed the NLI-verifier and Abstention classification, and analyzed the abstention behavior.

## License

This project is for academic purposes as part of Columbia University's COMS4705 course.

## Acknowledgments

Special thanks to our mentor **Melody Ma** for guidance throughout this project.
