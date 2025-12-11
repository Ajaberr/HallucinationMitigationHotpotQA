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
# 0. Metric Utils (EXACTLY from eval_abstention.py)
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

# NLI Model
NLI_MODEL_NAME = "microsoft/deberta-large-mnli"

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
        
        # --- PASS 1: GREEDY FOR BASELINE ---
        greedy_params = ttypes.SamplingParams(max_tokens=512, temperature=0.0) # Explicitly 0.0
        greedy_res_obj = None
        greedy_text = ""
        greedy_is_abstain = False
        greedy_metrics = {"em": 0.0, "f1": 0.0}

        try:
            greedy_res_obj = sampling_client.sample(model_input, 1, greedy_params).result()
            greedy_raw = tokenizer.decode(greedy_res_obj.sequences[0].tokens)
            greedy_text = robust_parse_abstention_model(greedy_raw)
            if not greedy_text: greedy_text = "NO_ANSWER_FOUND"
            
            greedy_is_abstain = check_abstention(greedy_text)
            
            gold_answers = item['answer'] if isinstance(item['answer'], list) else [item['answer']]
            
            if greedy_is_abstain:
                # If abstain, score is 0 unless gold is technically IDK
                if any(check_abstention(g) for g in gold_answers):
                     greedy_metrics = {"em": 1.0, "f1": 1.0}
            else:
                greedy_metrics["em"] = max([float(exact_match_score(greedy_text, g)) for g in gold_answers])
                greedy_metrics["f1"] = max([f1_score(greedy_text, g)[0] for g in gold_answers])
                
        except Exception as e:
            print(f"Error on greedy {i}: {e}")

        # --- PASS 2: SAMPLING FOR ENTROPY & CONSENSUS ---
        sampling_params = ttypes.SamplingParams(max_tokens=512, temperature=0.7, top_p=0.9)
        try:
            sample_res = sampling_client.sample(model_input, NUM_SAMPLES_PER_QUESTION, sampling_params).result()
            
            samples = []
            raw_texts = []
            
            for seq in sample_res.sequences:
                text = tokenizer.decode(seq.tokens)
                parsed = robust_parse_abstention_model(text)
                final_text = parsed if parsed else "NO_ANSWER_FOUND"
                
                if hasattr(seq, 'logprobs') and seq.logprobs:
                    prob = calculate_sequence_prob(seq.tokens, seq.logprobs)
                else:
                    prob = 1.0 / NUM_SAMPLES_PER_QUESTION
                
                samples.append((final_text, prob))
                raw_texts.append(final_text)

            # Normalize probs
            total_prob = sum(p for t, p in samples)
            if total_prob > 0:
                samples = [(t, p/total_prob) for t, p in samples]
            
            # --- CONSENSUS LOGIC ---
            # 1. Abstention Vote
            idk_count = sum(1 for t, _ in samples if check_abstention(t))
            consensus_is_abstain = idk_count >= (NUM_SAMPLES_PER_QUESTION / 2)
            
            # 2. Prediction Vote
            clusterer_out = clusterer.cluster_answers(samples)
            clusterer_out.sort(key=lambda x: x['prob'], reverse=True)
            
            consensus_text = ""
            if consensus_is_abstain:
                consensus_text = "I don't know"
            else:
                # Pick top NON-IDK cluster
                found = False
                for c in clusterer_out:
                    if not check_abstention(c['text']):
                        consensus_text = c['text']
                        found = True
                        break
                if not found: consensus_text = clusterer_out[0]['text']

            # 3. Consensus Metrics
            consensus_metrics = {"em": 0.0, "f1": 0.0}
            if consensus_is_abstain:
                if any(check_abstention(g) for g in gold_answers):
                    consensus_metrics = {"em": 1.0, "f1": 1.0}
            else:
                consensus_metrics["em"] = max([float(exact_match_score(consensus_text, g)) for g in gold_answers])
                consensus_metrics["f1"] = max([f1_score(consensus_text, g)[0] for g in gold_answers])

            # --- ENTROPY CALCULATION ---
            # A. Standard (Total Uncertainty)
            entropy = 0.0
            for c in clusterer_out:
                p_c = c['prob']
                if p_c > 0: entropy -= p_c * np.log(p_c)
            
            # B. Conditional (Knowledge Uncertainty)
            non_abstain_clusters = [c for c in clusterer_out if not check_abstention(c['text'])]
            conditional_entropy = 0.0
            if non_abstain_clusters:
                total_cond = sum(c['prob'] for c in non_abstain_clusters)
                if total_cond > 1e-6:
                    for c in non_abstain_clusters:
                        norm_p = c['prob'] / total_cond
                        if norm_p > 0: conditional_entropy -= norm_p * np.log(norm_p)

            results.append({
                "question": question,
                "gold": item['answer'],
                "entropy": entropy, 
                "conditional_entropy": conditional_entropy,
                
                # METRICS BUNDLE
                "decisions": {
                    "greedy": {
                        "text": greedy_text,
                        "is_abstain": greedy_is_abstain,
                        "metrics": greedy_metrics
                    },
                    "consensus": {
                        "text": consensus_text,
                        "is_abstain": consensus_is_abstain,
                        "idk_prob": idk_count / NUM_SAMPLES_PER_QUESTION,
                        "metrics": consensus_metrics
                    }
                },
                
                # Compatibility / Flat access (Prefer Decision Dict above)
                "is_abstention_decision": consensus_is_abstain, # Default to consensus for legacy check? Or greedy?
                # Actually, let's keep legacy keys pointing to CONSENSUS for safely running old scripts, 
                # but our new script will use the 'decisions' dict.
                
                "clusters": [{"text": c['text'], "prob": c['prob']} for c in clusterer_out]
            })

        except Exception as e:
            print(f"Error on sampling {i}: {e}")
            continue

    # --- FINAL STATISTICS ---
    # We will print the Hybrid Stats
    print("\n" + "="*30)
    print("FINAL HYBRID RESULTS")
    print("="*30)
    
    # helper
    def get_avg(lst, key):
        vals = [x['decisions'][key]['metrics']['em'] for x in results]
        return np.mean(vals) * 100 if vals else 0.0

    greedy_em = np.mean([x['decisions']['greedy']['metrics']['em'] for x in results]) * 100
    cons_em = np.mean([x['decisions']['consensus']['metrics']['em'] for x in results]) * 100
    
    abst_greedy = sum(1 for x in results if x['decisions']['greedy']['is_abstain'])
    abst_cons = sum(1 for x in results if x['decisions']['consensus']['is_abstain'])
    
    print(f"Samples: {len(results)}")
    print(f"Greedy EM:    {greedy_em:.2f}%  (Abstain Rate: {abst_greedy/len(results)*100:.1f}%)")
    print(f"Consensus EM: {cons_em:.2f}%  (Abstain Rate: {abst_cons/len(results)*100:.1f}%)")
    print("="*30)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved results with HYBRID metrics to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
