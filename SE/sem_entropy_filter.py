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
        # PRIORITY: Use valid pre-calculated metrics from the Greedy pass if available
        # This ensures we are evaluating exactly what the baseline evaluated.
        if 'metrics' in item:
            if metric_name == 'exact_match':
                return item['metrics'].get('em', 0.0)
            elif metric_name == 'f1_score':
                return item['metrics'].get('f1', 0.0)

        # Fallback: Calculate from top cluster if metrics not found
        clusters = item.get('clusters', [])
        if not clusters: return 0.0
        
        # Sort clusters by prob to find top prediction
        clusters.sort(key=lambda x: x['prob'], reverse=True)
        prediction = clusters[0]['text']
        
        gold_list = item['gold']
        if not isinstance(gold_list, list): gold_list = [gold_list]

        if metric_name == 'exact_match':
            return max([float(self.exact_match_score(prediction, g)) for g in gold_list])
            
        elif metric_name == 'f1_score':
             return max([self.f1_score(prediction, g)[0] for g in gold_list])
        
        return 0.0

    # =========================================================
    # RIGOROUS METRIC UTILS (Ported from eval_abstention.py)
    # =========================================================
    @staticmethod
    def normalize_answer(s):
        import re
        import string
        def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text): return ' '.join(text.split())
        def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
        def lower(text): return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def f1_score(prediction, ground_truth):
        normalized_prediction = SemanticEntropyFilter.normalize_answer(prediction)
        normalized_ground_truth = SemanticEntropyFilter.normalize_answer(ground_truth)
        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            return 0.0, 0.0, 0.0
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            return 0.0, 0.0, 0.0
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        from collections import Counter
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: return 0.0, 0.0, 0.0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return (SemanticEntropyFilter.normalize_answer(prediction) == SemanticEntropyFilter.normalize_answer(ground_truth))

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

    # ... (Keep existing class methods) ...
    # We will add a new method for the Box Plot
    def plot_abstention_correlation(self, save_path):
        import seaborn as sns
        
        # Prepare Data
        # We use the CONSENSUS decision for this correlation, as it derives from the same samples as Entropy
        data_points = []
        for x in self.data:
            # Handle legacy or new format
            if 'decisions' in x:
                is_abstain = x['decisions']['consensus']['is_abstain']
            else:
                is_abstain = x.get('is_abstention_decision', False)
                
            status = "Abstained" if is_abstain else "Answered"
            data_points.append({'Status': status, 'Entropy': x['entropy']})
            
        plt.figure(figsize=(8, 6))
        # Box Plot
        sns.boxplot(x='Status', y='Entropy', data=data_points, palette="Set2")
        plt.title("Semantic Entropy Distribution: Abstained vs Answered")
        plt.ylabel("Semantic Entropy (Uncertainty)")
        plt.grid(True, linestyle='--', alpha=0.3)
        
        print(f"Saving correlation plot to: {save_path}")
        plt.savefig(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, nargs='?', default="tinker/results/tinker_sem_entropy_abstention_results.json")
    args = parser.parse_args()

    input_path = args.filename
    if not os.path.exists(input_path):
        # Check standard location
        if os.path.exists("tinker_sem_entropy_abstention_results.json"):
             input_path = "tinker_sem_entropy_abstention_results.json"
        else:
            print(f"File not found: {input_path}")
            sys.exit(1)

    print(f"Loading: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    processor = SemanticEntropyFilter(data)
    
    # --- PLOT 1: REJECTION CURVES (All Combinations) ---
    metrics = ['exact_match', 'f1_score']
    
    for m in metrics:
        results_store = {}
        
        # Helper to extract metric
        def get_score(item, source='consensus', metric=m):
            key = 'em' if metric == 'exact_match' else 'f1'
            if 'decisions' in item:
                return item['decisions'][source]['metrics'].get(key, 0.0)
            return item.get('metrics', {}).get(key, 0.0) # Fallback

        # 1. Greedy Acc vs Standard SE
        rates_1, scores_1 = processor.calculate_rejection_curve(
            data_subset=data,
            sorting_key=lambda x: x.get('entropy', 100),
            metric_evaluator=lambda x: get_score(x, 'greedy')
        )
        results_store['Greedy_StdSE'] = {'rates': rates_1, 'scores': scores_1, 'auc': processor.calculate_auc(rates_1, scores_1)}

        # 2. Greedy Acc vs Conditional SE
        rates_2, scores_2 = processor.calculate_rejection_curve(
            data_subset=data,
            sorting_key=lambda x: x.get('conditional_entropy', 100),
            metric_evaluator=lambda x: get_score(x, 'greedy')
        )
        results_store['Greedy_CondSE'] = {'rates': rates_2, 'scores': scores_2, 'auc': processor.calculate_auc(rates_2, scores_2)}

        # 3. Consensus Acc vs Standard SE
        rates_3, scores_3 = processor.calculate_rejection_curve(
            data_subset=data,
            sorting_key=lambda x: x.get('entropy', 100),
            metric_evaluator=lambda x: get_score(x, 'consensus')
        )
        results_store['Consensus_StdSE'] = {'rates': rates_3, 'scores': scores_3, 'auc': processor.calculate_auc(rates_3, scores_3)}

        # 4. Consensus Acc vs Conditional SE (The "Knowledge" Curve)
        # Filter: ONLY Non-Abstained Consensus Outcomes
        subset_4 = [x for x in data if not x['decisions']['consensus']['is_abstain']]
        if subset_4:
            rates_4, scores_4 = processor.calculate_rejection_curve(
                data_subset=subset_4,
                sorting_key=lambda x: x.get('conditional_entropy', 100),
                metric_evaluator=lambda x: get_score(x, 'consensus')
            )
            results_store['Consensus_CondSE_Selective'] = {'rates': rates_4, 'scores': scores_4, 'auc': processor.calculate_auc(rates_4, scores_4)}
        else:
            results_store['Consensus_CondSE_Selective'] = {'rates': [0], 'scores': [0], 'auc': 0}

        # Plot All 4
        plt.figure(figsize=(12, 8))
        colors = {'Greedy_StdSE': 'red', 'Greedy_CondSE': 'orange', 'Consensus_StdSE': 'blue', 'Consensus_CondSE_Selective': 'green'}
        styles = {'Greedy_StdSE': '--', 'Greedy_CondSE': ':', 'Consensus_StdSE': '-', 'Consensus_CondSE_Selective': '-.'}
        
        for name, res in results_store.items():
            plt.plot(np.array(res['rates'])*100, res['scores'], label=f"{name} (AUC: {res['auc']:.3f})", color=colors.get(name,'black'), linestyle=styles.get(name, '-'))
            
        plt.title(f"{m} Rejection Curves (All Variants)")
        plt.xlabel("Rejection Rate (%)")
        plt.ylabel(f"Average {m}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"se_curves_{m}.png")
        print(f"Saved se_curves_{m}.png")
    
    # --- PLOT 2: CORRELATION (Abstained vs Answered) ---
    processor.plot_abstention_correlation("se_correlation_boxplot.png")