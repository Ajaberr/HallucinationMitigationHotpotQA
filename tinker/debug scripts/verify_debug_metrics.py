import json
import numpy as np
import re
import string
from collections import Counter

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)
    if norm_pred in ['yes', 'no', 'noanswer'] and norm_pred != norm_gold:
        return 0.0
    if norm_gold in ['yes', 'no', 'noanswer'] and norm_pred != norm_gold:
        return 0.0
    
    pred_toks = norm_pred.split()
    gold_toks = norm_gold.split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

with open("tinker_sem_entropy_abstention_results.json", "r") as f:
    data = json.load(f)

answered = [x for x in data if not x['is_abstention_decision']]
print(f"Total Samples: {len(data)}")
print(f"Answered Samples: {len(answered)}")
print(f"Abstention Rate: {1 - len(answered)/len(data):.2%}")

scores = []
for item in answered:
    clusters = item['clusters']
    clusters.sort(key=lambda x: x['prob'], reverse=True)
    pred = clusters[0]['text']
    gold = item['gold']
    if not isinstance(gold, list): gold = [gold]
    
    best_f1 = max([f1_score(pred, g) for g in gold])
    scores.append(best_f1)
    
print(f"Selective F1 (Method B Baseline): {np.mean(scores):.4f}")
