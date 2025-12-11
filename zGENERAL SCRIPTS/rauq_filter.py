import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
from sklearn.metrics import auc

class RAUQFilter:
    def __init__(self, data):
        self.data = data
        self.total_samples = len(data)
        
        # Sort data by RAUQ Score (Ascending).
        # Lower RAUQ = Lower Uncertainty = Higher Confidence.
        self.sorted_data = sorted(
            self.data, 
            key=lambda x: x['metrics']['rauq_score']
        )

    def calculate_auc(self, x, y):
        return auc(x, y)

    def _compute_exact_curve(self, sorted_metric_values, max_rejection):
        """
        Helper function to compute curve using cumulative sums (O(N) complexity).
        Used for both Model (RAUQ sorted) and Oracle (Metric sorted).
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

    def calculate_rejection_curve(self, metric_key='exact_match', max_rejection=1.0):
        """
        Calculates the curve based on RAUQ sorting (Model Performance).
        """
        sorted_scores = [x['metrics'][metric_key] for x in self.sorted_data]
        return self._compute_exact_curve(sorted_scores, max_rejection)

    def calculate_prr(self, model_auc, metric_key='exact_match', max_rejection=1.0):
        """
        Calculates PRR comparing Model vs Random vs Oracle.
        """
        # 1. Random Baseline
        initial_accuracy = np.mean([d['metrics'][metric_key] for d in self.data])
        random_auc = initial_accuracy * max_rejection

        # 2. Oracle Baseline
        oracle_data = sorted(self.data, key=lambda x: x['metrics'][metric_key], reverse=True)
        oracle_scores_list = [x['metrics'][metric_key] for x in oracle_data]
        
        o_rates, o_scores = self._compute_exact_curve(oracle_scores_list, max_rejection)
        oracle_auc = self.calculate_auc(o_rates, o_scores)

        # 3. Compute PRR
        denominator = oracle_auc - random_auc
        
        if denominator <= 0:
            return 0.0, random_auc, oracle_auc
            
        prr = (model_auc - random_auc) / denominator
        return prr, random_auc, oracle_auc

def plot_comparison(results, max_rejection, save_path):
    """
    Plots the results and saves to the specified save_path.
    """
    plt.figure(figsize=(12, 7))
    
    colors = {'exact_match': '#e74c3c', 'f1_score': '#2980b9'}
    labels = {'exact_match': 'Exact Match', 'f1_score': 'F1 Score'}
    
    for metric, data in results.items():
        x = np.array(data['rates']) * 100 
        y = np.array(data['scores'])
        
        auc_val = data['auc']
        prr_val = data['prr']
        
        label_text = f"{labels[metric]} (AUC: {auc_val:.3f} | PRR: {prr_val:.3f})"
        plt.plot(x, y, linestyle='-', linewidth=2, color=colors[metric], label=label_text, alpha=0.7)

    # Styling
    plt.title('RAUQ Rejection Curve: Accuracy vs Rejection Rate', fontsize=16)
    plt.xlabel('Rejection Rate (%)', fontsize=12)
    plt.ylabel('Score of Retained Samples', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.axvspan(0, max_rejection * 100, color='gray', alpha=0.1, label='Analyzed Range')
    
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Save to the directory provided in input
    print(f"Saving figure to: {save_path}")
    plt.savefig(save_path)
    # plt.show() # Commented out to prevent blocking execution in bulk runs

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Process RAUQ results from a directory.")
    parser.add_argument("directory", type=str, help="Path to the directory containing detailed_results.json")
    args = parser.parse_args()

    target_dir = args.directory
    json_filename = "detailed_results.json"
    plot_filename = "rauq_rejection_curve.png"

    # 2. Construct Paths
    input_path = os.path.join(target_dir, json_filename)
    output_path = os.path.join(target_dir, plot_filename)
    
    MAX_REJECTION_THRESHOLD = 1.0
    metrics_to_analyze = ['exact_match', 'f1_score']
    results_store = {}

    # 3. Validate Input File
    if not os.path.exists(input_path):
        print(f"Error: File '{json_filename}' not found in directory: {target_dir}")
        sys.exit(1)

    print(f"Loading data from: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    processor = RAUQFilter(data)
    
    print(f"\n{'Metric':<15} | {'Base Score':<10} | {'AUC':<10} | {'PRR':<10}")
    print("-" * 55)

    for metric in metrics_to_analyze:
        # Calculate Curve
        rates, scores = processor.calculate_rejection_curve(
            metric_key=metric, 
            max_rejection=MAX_REJECTION_THRESHOLD
        )
        
        # Calculate AUC
        model_auc = processor.calculate_auc(rates, scores)
        
        # Calculate PRR
        prr, random_auc, oracle_auc = processor.calculate_prr(
            model_auc, 
            metric_key=metric, 
            max_rejection=MAX_REJECTION_THRESHOLD
        )

        results_store[metric] = {
            'rates': rates,
            'scores': scores,
            'auc': model_auc,
            'prr': prr
        }
        
        base_score = scores[0] if len(scores) > 0 else 0
        print(f"{metric:<15} | {base_score:<10.4f} | {model_auc:<10.4f} | {prr:<10.4f}")

    # 4. Plot and Save to Target Directory
    plot_comparison(results_store, MAX_REJECTION_THRESHOLD, output_path)