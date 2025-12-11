import os
import tinker
from tinker import types as ttypes
from dotenv import load_dotenv
import json
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import re
import string
from collections import Counter

load_dotenv()

# ==========================================
# 0. Metric Utils (Copied from eval_abstention.py)
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

REFUSAL_PHRASE = "i dont know"
def check_abstention(prediction):
    norm_pred = normalize_answer(prediction)
    return REFUSAL_PHRASE in norm_pred

# ==========================================
# 1. Configuration
# ==========================================

with open("tinker_models.json", "r") as f:
    MODEL_REGISTRY = json.load(f)

# Model to Evaluate (Abstention Adapter)
ADAPTER_PATH = MODEL_REGISTRY["qwen_hotpot_abstention_v1"]["uri"]
BASE_MODEL = "Qwen/Qwen3-8B"

# Data
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SPLIT = "validation"
# 10 Samples for Entropy Calculation (Standard)
NUM_SAMPLES_PER_QUESTION = 10  
TOTAL_EVAL_SAMPLES = 50 # Speed setting (Increase for full run)

# ABSTENTION FORMATTING
DELIMITER_PROMPT = " ###\n"
DELIMITER_END = " END"

# NLI Model
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"

OUTPUT_FILE = "tinker_sem_entropy_abstention_results.json"
METRICS_FILE = "tinker_sem_entropy_abstention_metrics.json"

# ==========================================
# 2. NLI Clustering Logic (Shared)
# ==========================================

class SemanticClusterer:
    def __init__(self, model_name, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Loading NLI model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.entailment_idx = 1 

    def predict_entailment(self, premise, hypothesis):
        inputs = self.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        prediction = torch.argmax(logits, dim=-1).item()
        return prediction == self.entailment_idx

    def are_equivalent(self, text1, text2):
        if text1 == text2: return True
        return self.predict_entailment(text1, text2) and self.predict_entailment(text2, text1)

    def cluster_answers(self, answers_with_probs):
        clusters = [] 
        for ans_text, ans_prob in answers_with_probs:
            matched = False
            for cluster in clusters:
                if self.are_equivalent(ans_text, cluster['text']):
                    cluster['prob'] += ans_prob
                    cluster['members'].append(ans_text)
                    matched = True
                    break
            
            if not matched:
                clusters.append({
                    'text': ans_text,
                    'prob': ans_prob,
                    'members': [ans_text]
                })
        return clusters

# ==========================================
# 3. Robust Parsing (Ported from eval_abstention.py)
# ==========================================

def robust_parse_abstention_model(output_text):
    # 1. Cleanup special tokens
    output_text = output_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n")[-1]

    # 2. Check for " END" delimiter (from training)
    if " END" in output_text:
        output_text = output_text.split(" END")[0]
    
    output_text = output_text.strip()
    
    # 3. Robust Parsing
    # Normalize dashes to standard hyphen to catch en-dash, em-dash, etc.
    normalized_text = output_text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    
    parsed_answer = ""
    
    # Regex for arrow (handle variable spacing)
    arrow_match = re.search(r"\s*-{2,}>\s*", normalized_text)
    
    # PRIORITY 1: The Training Delimiter "-->"
    if arrow_match:
        parts = re.split(r"\s*-{2,}>\s*", normalized_text)
        parsed_answer = parts[-1].strip()
    
    # PRIORITY 2: Heuristic "Therefore..." (Only if arrow missing)
    elif "Therefore, the answer is" in normalized_text:
        parsed_answer = normalized_text.split("Therefore, the answer is")[-1].strip()
    
    elif "The answer is" in normalized_text:
        parsed_answer = normalized_text.split("The answer is")[-1].strip()
    
    else:
        # Fallback
        if "Step" in normalized_text: 
            parsed_answer = "" 
        else:
            parsed_answer = normalized_text

    # Final Cleanup
    if parsed_answer.endswith("."):
        parsed_answer = parsed_answer[:-1]
    
    return parsed_answer.strip()

def calculate_sequence_prob(token_ids, logprobs):
    total_logprob = sum(logprobs)
    return np.exp(total_logprob)

# ==========================================
# 4. Main Loop
# ==========================================

def main():
    print("🚀 Connecting to Tinker (SE + Abstention Eval)...")
    if not os.environ.get("TINKER_API_KEY"):
        print("⚠️ TINKER_API_KEY not found.")
        return

    service = tinker.ServiceClient()
    sampling_client = service.create_sampling_client(model_path=ADAPTER_PATH)
    
    training_client = service.create_lora_training_client(base_model=BASE_MODEL, rank=16)
    tokenizer = training_client.get_tokenizer()

    print("init nli...")
    clusterer = SemanticClusterer(NLI_MODEL_NAME)

    print(f"\nLoading Dataset ({DATASET_NAME})...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT)
    if TOTAL_EVAL_SAMPLES:
        dataset = dataset.select(range(min(len(dataset), TOTAL_EVAL_SAMPLES)))
    
    results = []
    
    # Validation Metrics Storage
    standard_em_scores = []
    standard_f1_scores = []
    selective_em_scores = []
    selective_f1_scores = []
    total_abstentions = 0
    
    print(f"Starting eval loop for {len(dataset)} questions...")
    
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        question = item['question']
        
        # 1. Format Prompt (MANUAL FORMAT for Abstention Model)
        prompt_text = f"{question}{DELIMITER_PROMPT}"
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        model_input = ttypes.ModelInput.from_ints(input_ids)
        
        # 2. Generate N Samples
        sampling_params = ttypes.SamplingParams(
            max_tokens=512,  # Long generation for CoT
            temperature=0.7, # Diversity needed for Entropy
            top_p=0.9
        )
        
        try:
            result = sampling_client.sample(model_input, NUM_SAMPLES_PER_QUESTION, sampling_params).result()
            
            samples = []
            
            for seq in result.sequences:
                # Decode
                text = tokenizer.decode(seq.tokens)
                
                # Parse
                parsed_answer = robust_parse_abstention_model(text)
                
                # Store
                final_text = parsed_answer if parsed_answer else "NO_ANSWER_FOUND"
                
                # Prob
                if hasattr(seq, 'logprobs') and seq.logprobs:
                    prob = calculate_sequence_prob(seq.tokens, seq.logprobs)
                else:
                    prob = 1.0 / NUM_SAMPLES_PER_QUESTION
                
                samples.append((final_text, prob))
                
            # Normalize probs
            total_prob = sum(p for t, p in samples)
            if total_prob > 0:
                samples = [(t, p/total_prob) for t, p in samples]
            
            # 3. Cluster
            clusters = clusterer.cluster_answers(samples)
            
            # 3. Cluster
            clusters = clusterer.cluster_answers(samples)
            
            # 4. Metric A: Standard Semantic Entropy (Includes IDK as a cluster)
            entropy = 0.0
            for c in clusters:
                p_c = c['prob']
                if p_c > 0:
                    entropy -= p_c * np.log(p_c)
            
            # 5. Metric B: Conditional Semantic Entropy (Gated)
            # Entropy given that the model provided an answer (ignore IDK clusters)
            non_abstain_clusters = []
            for c in clusters:
                # Check directly for refusal phrase in the cluster representative text
                if "i dont know" not in c['text'].lower().replace("'", ""):
                     non_abstain_clusters.append(c)
            
            conditional_entropy = 0.0
            if non_abstain_clusters:
                total_cond_prob = sum(c['prob'] for c in non_abstain_clusters)
                # If total prob is tiny, it means model almost always abstained. 
                # In that case, conditional entropy is technically undefined or 0? 
                # Let's normalize if significant mass exists.
                if total_cond_prob > 1e-6:
                    for c in non_abstain_clusters:
                        norm_p = c['prob'] / total_cond_prob
                        if norm_p > 0:
                            conditional_entropy -= norm_p * np.log(norm_p)
            else:
                 # All answers were IDK. Conditional entropy is 0 (or undefined).
                 conditional_entropy = 0.0

            # Additional Metric: Is the most likely cluster "I don't know"?
            # Sort clusters by prob
            clusters.sort(key=lambda x: x['prob'], reverse=True)
            top_cluster = clusters[0]
            prediction_text = top_cluster['text']
            
            # --- METRIC CALCULATION (Same as eval_abstention.py) ---
            is_abstention_decision = check_abstention(prediction_text)
            
            gold_answers = item['answer'] if isinstance(item['answer'], list) else [item['answer']]
            
            # Best match against any valid gold answer
            best_em = max([float(exact_match_score(prediction_text, g)) for g in gold_answers])
            best_f1 = max([f1_score(prediction_text, g)[0] for g in gold_answers])
            
            standard_em_scores.append(best_em)
            standard_f1_scores.append(best_f1)
            
            if is_abstention_decision:
                total_abstentions += 1
            else:
                selective_em_scores.append(best_em)
                selective_f1_scores.append(best_f1)

            results.append({
                "question": question,
                "gold": item['answer'],
                "entropy": entropy, # Method A
                "conditional_entropy": conditional_entropy, # Method B
                "is_abstention_decision": is_abstention_decision,
                "metrics": {"em": best_em, "f1": best_f1},
                "num_clusters": len(clusters),
                "num_non_abstain_clusters": len(non_abstain_clusters),
                "clusters": [{"text": c['text'], "prob": c['prob']} for c in clusters],
                "samples": [s[0] for s in samples]
            })

        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    # --- FINAL STATISTICS CAULCULATION ---
    total_samples = len(results)
    if total_samples > 0:
        abstention_rate = (total_abstentions / total_samples) * 100
        std_em = np.mean(standard_em_scores) * 100
        std_f1 = np.mean(standard_f1_scores) * 100
        
        sel_em = np.mean(selective_em_scores) * 100 if selective_em_scores else 0.0
        sel_f1 = np.mean(selective_f1_scores) * 100 if selective_f1_scores else 0.0
    else:
        abstention_rate = std_em = std_f1 = sel_em = sel_f1 = 0.0

    print("\n" + "="*30)
    print("FINAL SE RESULTS (Abstention Model)")
    print("="*30)
    print(f"Abstention Rate: {abstention_rate:.2f}%")
    print("-" * 20)
    print(f"Standard EM:     {std_em:.2f}%")
    print(f"Standard F1:     {std_f1:.2f}%")
    print("-" * 20)
    print(f"Selective EM:    {sel_em:.2f}%  (Accuracy on Answered)")
    print(f"Selective F1:    {sel_f1:.2f}%")
    print("="*30)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
        
    final_metrics = {
        "model": ADAPTER_PATH,
        "abstention_rate": abstention_rate,
        "standard_em": std_em,
        "standard_f1": std_f1,
        "selective_em": sel_em,
        "selective_f1": sel_f1
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print(f"Saved results to {OUTPUT_FILE}")
    print(f"Saved metrics to {METRICS_FILE}")

if __name__ == "__main__":
    main()
