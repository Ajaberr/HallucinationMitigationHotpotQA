import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
from sklearn.metrics import auc

class SemanticEntropyFilter:
    def __init__(self, data):
        self.data = data
        self.total_samples = len(data)
        
        # Sort data by Semantic Entropy (Ascending).
        # Lower Entropy = Lower Uncertainty = Higher Confidence.
        # Note: 'entropy' is at the root of the item, not in 'metrics'
        self.sorted_data = sorted(
            self.data, 
            key=lambda x: x.get('entropy', float('inf')) 
        )

    def calculate_auc(self, x, y):
        return auc(x, y)

    def _compute_exact_curve(self, sorted_metric_values, max_rejection):
        """
        Helper function to compute curve using cumulative sums (O(N) complexity).
        Used for both Model (Entropy sorted) and Oracle (Metric sorted).
        """
        # Pre-calculate cumulative sums for fast averaging
        cum_scores = np.cumsum(sorted_metric_values)
        total = len(sorted_metric_values)
        
        rejection_rates = []
        performance_scores = []

        # Iterate from keeping 100% (total) down to 1 item
        for i in range(total, 0, -1):
            rate = 1 - (i / total)
            
            # Stop if we exceed the max rejection threshold
            if rate > max_rejection:
                continue
                
            # Average of top i items = Cumulative Sum at index i-1 / i
            avg_score = cum_scores[i-1] / i
            
            rejection_rates.append(rate)
            performance_scores.append(avg_score)
            
        return np.array(rejection_rates), np.array(performance_scores)

    def calculate_rejection_curve(self, data_subset, sorting_key, metric_evaluator, max_rejection=1.0):
        """
        Generic curve calculator for a subset of data.
        data_subset: List of items to analyze
        sorting_key: function to extract the uncertainty value (for sorting)
        metric_evaluator: function(item) -> float score (0.0 or 1.0 usually)
        """
        # Sort by Uncertainty (Low to High)
        sorted_subset = sorted(data_subset, key=sorting_key)
        
        # Extract Scores
        scores = [metric_evaluator(x) for x in sorted_subset]
        
        return self._compute_exact_curve(scores, max_rejection)

    def evaluate_metric(self, item, metric_name):
        # Helper to compute EM/F1 on the fly since we didn't store it in 'metrics' dict for this script
        # We assume the top cluster text is the prediction.
        
        # Sort clusters by prob to find top prediction
        clusters = item['clusters']
        if not clusters: return 0.0
        clusters.sort(key=lambda x: x['prob'], reverse=True)
        prediction = clusters[0]['text']
        gold_list = item['gold']
        if not isinstance(gold_list, list): gold_list = [gold_list]

        if metric_name == 'exact_match':
            # Simple normalization check
            def norm(s): return "".join(s.lower().split())
            return max([1.0 if norm(prediction) == norm(g) else 0.0 for g in gold_list])
            
        elif metric_name == 'f1_score':
            # Simplified bag of words F1
            def normalize_answer(s): return " ".join(s.lower().split())
            best_f1 = 0.0
            pred_toks = normalize_answer(prediction).split()
            for gold in gold_list:
                gold_toks = normalize_answer(gold).split()
                common = set(pred_toks) & set(gold_toks)
                num_same = len(common)
                if num_same == 0: continue
                precision = 1.0 * num_same / len(pred_toks)
                recall = 1.0 * num_same / len(gold_toks)
                f1 = (2 * precision * recall) / (precision + recall)
                if f1 > best_f1: best_f1 = f1
            return best_f1
        return 0.0

def plot_comparison(results_store, max_rejection, save_path):
    plt.figure(figsize=(14, 8))
    
    # Colors: Method A (Solid), Method B (Dashed)
    colors = {'exact_match': 'tab:red', 'f1_score': 'tab:blue'}
    
    for metric_name in results_store:
        # Method A
        data_A = results_store[metric_name]['Method_A']
        xA = np.array(data_A['rates']) * 100 
        yA = np.array(data_A['scores'])
        aucA = data_A['auc']
        labelA = f"{metric_name} (Std SE) AUC: {aucA:.3f}"
        plt.plot(xA, yA, linestyle='-', linewidth=2, color=colors[metric_name], label=labelA, alpha=0.8)

        # Method B
        data_B = results_store[metric_name]['Method_B']
        xB = np.array(data_B['rates']) * 100 
        yB = np.array(data_B['scores'])
        aucB = data_B['auc']
        labelB = f"{metric_name} (Cond SE) AUC: {aucB:.3f}"
        plt.plot(xB, yB, linestyle='--', linewidth=2, color=colors[metric_name], label=labelB, alpha=0.8)

    plt.title('Semantic Entropy: Standard (A) vs Conditional (B)', fontsize=16)
    plt.xlabel('Rejection Rate (%)', fontsize=12)
    plt.ylabel('Score of Retained Samples', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    print(f"Saving comparison plot to: {save_path}")
    plt.savefig(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, nargs='?', default="tinker_sem_entropy_abstention_results.json")
    args = parser.parse_args()

    input_path = args.filename
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    print(f"Loading: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    processor = SemanticEntropyFilter(data)
    
    metrics = ['exact_match', 'f1_score']
    results_store = {}

    for m in metrics:
        results_store[m] = {}
        
        # --- METHOD A: STANDARD ---
        # Population: ALL
        # Score: entropy
        rates_A, scores_A = processor.calculate_rejection_curve(
            data_subset=data,
            sorting_key=lambda x: x.get('entropy', 100),
            metric_evaluator=lambda x: processor.evaluate_metric(x, m)
        )
        results_store[m]['Method_A'] = {
            'rates': rates_A, 'scores': scores_A, 
            'auc': processor.calculate_auc(rates_A, scores_A)
        }

        # --- METHOD B: CONDITIONAL ---
        # Population: ONLY Non-Abstained
        # Score: conditional_entropy
        subset_B = [x for x in data if not x.get('is_abstention_decision', False)]
        
        if not subset_B:
            print(f"Warning: No non-abstained samples found for Method B analysis.")
            results_store[m]['Method_B'] = {'rates': [0], 'scores': [0], 'auc': 0}
        else:
            rates_B, scores_B = processor.calculate_rejection_curve(
                data_subset=subset_B,
                sorting_key=lambda x: x.get('conditional_entropy', 100),
                metric_evaluator=lambda x: processor.evaluate_metric(x, m)
            )
            results_store[m]['Method_B'] = {
                'rates': rates_B, 'scores': scores_B, 
                'auc': processor.calculate_auc(rates_B, scores_B)
            }

    plot_comparison(results_store, 1.0, "se_abstention_comparison.png")