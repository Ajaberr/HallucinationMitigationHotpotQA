import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def load_results(filename, model_prefix):
    """Loads results and normalizes EM and Abstention columns."""
    print(f"  Loading {filename}...")
    with open(filename, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 0. Normalize ID and question columns
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)

    # Ensure question column exists
    if 'question' not in df.columns:
        raise ValueError(f"File {filename} missing 'question' column")

    # 1. Normalize Exact Match (EM)
    if 'metrics' in df.columns:
        sample_metric = df['metrics'].iloc[0]
        metric_key = 'exact_match' if 'exact_match' in sample_metric else 'em'
        df[f'{model_prefix}_em'] = df['metrics'].apply(lambda x: x.get(metric_key, 0))

    # 2. Normalize Abstention
    if 'is_abstention' in df.columns:
        df[f'{model_prefix}_abstain'] = df['is_abstention']
    else:
        df[f'{model_prefix}_abstain'] = False

    # 3. Normalize Prediction
    if 'prediction' in df.columns:
        df[f'{model_prefix}_pred'] = df['prediction']
    elif 'parsed_answer' in df.columns:
        df[f'{model_prefix}_pred'] = df['parsed_answer']

    return df

def calculate_abstention_stats(df, ref_model, new_model='rlhf'):
    """
    Calculates Precision and Recall of abstention assuming
    'Ref Model Wrong' is the ground truth for 'Should Abstain'.
    """
    # TP: Ref is Wrong (Should Abstain) AND New Model Abstained
    tp = len(df[(df[f'{ref_model}_em'] == 0) & (df[f'{new_model}_abstain'] == True)])

    # FP: Ref is Right (Should Answer) BUT New Model Abstained
    fp = len(df[(df[f'{ref_model}_em'] == 1) & (df[f'{new_model}_abstain'] == True)])

    # FN: Ref is Wrong (Should Abstain) BUT New Model Answered
    fn = len(df[(df[f'{ref_model}_em'] == 0) & (df[f'{new_model}_abstain'] == False)])

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        'Reference Model': ref_model.upper(),
        'Precision': f"{precision:.2%}",
        'Recall': f"{recall:.2%}",
        'TP (Good Abstain)': tp,
        'FP (Bad Abstain)': fp,
        'FN (Missed Abstain)': fn
    }

print("="*80)
print("COMPREHENSIVE RLHF MODEL ANALYSIS")
print("="*80)

# Check if RLHF results exist
if not os.path.exists('rlhf_detailed_results.json'):
    print("ERROR: rlhf_detailed_results.json not found in current directory!")
    print("Looking for results in 'results/' subdirectory...")
    if os.path.exists('results/rlhf_detailed_results.json'):
        os.chdir('results')
    else:
        exit(1)

# Load RLHF model
print("\n1. Loading RLHF Model Results...")
df_rlhf = load_results('rlhf_detailed_results.json', 'rlhf')

print(f"   Loaded {len(df_rlhf)} RLHF samples")

# Basic RLHF Statistics
print("\n" + "="*80)
print("PART 1: RLHF MODEL PERFORMANCE")
print("="*80)

if 'rlhf_em' in df_rlhf.columns:
    overall_em = df_rlhf['rlhf_em'].mean() * 100
    print(f"\nOverall EM: {overall_em:.2f}%")

if 'rlhf_abstain' in df_rlhf.columns:
    abstention_rate = df_rlhf['rlhf_abstain'].mean() * 100
    print(f"Abstention Rate: {abstention_rate:.2f}%")

    answered = df_rlhf[df_rlhf['rlhf_abstain'] == False]
    abstained = df_rlhf[df_rlhf['rlhf_abstain'] == True]

    if len(answered) > 0 and 'rlhf_em' in df_rlhf.columns:
        answered_em = answered['rlhf_em'].mean() * 100
        print(f"\nPerformance on Answered Questions:")
        print(f"  EM: {answered_em:.2f}% ({len(answered)} samples)")

    if len(abstained) > 0:
        print(f"\nAbstained Questions: {len(abstained)} samples")

# Try to compare with baseline models if available
baseline_files = {
    'instruct': 'qwen_base_closedbook_full_results.json',
    'sft': 'qwen_sft_closedbook_full_results.json',
    'cot': 'qwen_ftcot_closedbook_full_results_1.json',
    'abstention_sft': 'eval_results_hotpotqa-abstention-80-20_SHORT_lr0.0002_bs4_20251210_020244.json'
}

# Try to load baseline models
baseline_loaded = {}
print("\n" + "="*80)
print("PART 2: LOADING BASELINE MODELS FOR COMPARISON")
print("="*80)

for name, path in baseline_files.items():
    if os.path.exists(path):
        try:
            baseline_loaded[name] = load_results(path, name)
            print(f"  ✓ Loaded {name}: {len(baseline_loaded[name])} samples")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
    else:
        print(f"  - {name} not found at {path}")

if baseline_loaded:
    print("\n" + "="*80)
    print("PART 3: COMPARATIVE ANALYSIS")
    print("="*80)

    # Merge all available datasets by question text (more reliable than ID)
    merged = df_rlhf[['question']].copy()
    merged['rlhf_em'] = df_rlhf['rlhf_em']
    merged['rlhf_abstain'] = df_rlhf['rlhf_abstain']

    for name, df in baseline_loaded.items():
        if 'question' in df.columns:
            merged = merged.merge(df[['question', f'{name}_em', f'{name}_abstain']], on='question', how='inner')

    print(f"\nAligned {len(merged)} questions across all models.")

    # Compare RLHF with each baseline
    print("\n" + "-"*80)
    print("Overall Accuracy Comparison (EM)")
    print("-"*80)
    print(f"RLHF:            {merged['rlhf_em'].mean():.2%}")

    for name in baseline_loaded.keys():
        if f'{name}_em' in merged.columns:
            print(f"{name.upper():16} {merged[f'{name}_em'].mean():.2%}")

    # Abstention comparison
    print("\n" + "-"*80)
    print("Abstention Rate Comparison")
    print("-"*80)
    print(f"RLHF:            {merged['rlhf_abstain'].mean():.2%}")

    for name in baseline_loaded.keys():
        if f'{name}_abstain' in merged.columns:
            print(f"{name.upper():16} {merged[f'{name}_abstain'].mean():.2%}")

    # Abstention Precision/Recall Analysis
    if any(f'{name}_em' in merged.columns for name in ['instruct', 'sft', 'cot']):
        print("\n" + "="*80)
        print("PART 4: ABSTENTION QUALITY ANALYSIS")
        print("="*80)
        print("\nHow well does RLHF abstain on questions that baseline models get wrong?")
        print("-"*80)

        stats = []
        for ref in ['instruct', 'sft', 'cot']:
            if f'{ref}_em' in merged.columns:
                stats.append(calculate_abstention_stats(merged, ref, 'rlhf'))

        # Hardest questions (all baselines wrong)
        if all(f'{ref}_em' in merged.columns for ref in ['instruct', 'sft', 'cot']):
            merged['all_others_wrong'] = (
                (merged['instruct_em'] == 0) &
                (merged['sft_em'] == 0) &
                (merged['cot_em'] == 0)
            ).astype(int)
            merged['hardest_em'] = 1 - merged['all_others_wrong']
            stats.append(calculate_abstention_stats(merged, 'hardest', 'rlhf'))

        results_df = pd.DataFrame(stats)
        print(results_df.to_string(index=False))
        results_df.to_csv('rlhf_abstention_analysis.csv', index=False)
        print("\n✓ Saved to rlhf_abstention_analysis.csv")

        # Generate Confusion Matrix Visualizations
        print("\n" + "="*80)
        print("GENERATING CONFUSION MATRIX VISUALIZATIONS")
        print("="*80)

        # Set global font scale for better readability
        sns.set_context("talk", font_scale=1.1)
        plt.rcParams.update({'font.size': 14})

        # Define comparisons
        comparisons = ['instruct', 'sft', 'cot', 'hardest']

        # Set up the plot area
        fig, axes = plt.subplots(1, 4, figsize=(28, 7))

        for i, ref in enumerate(comparisons):
            # 1. Define "Ground Truth" for Abstention
            # "Positive" (1) = We SHOULD Abstain (because Ref model is Wrong)
            # "Negative" (0) = We SHOULD Answer (because Ref model is Right)
            if ref == 'hardest':
                y_true = merged['all_others_wrong']  # 1 if all wrong
            else:
                y_true = 1 - merged[f'{ref}_em']  # 1 if wrong, 0 if right

            # 2. Define Prediction
            # "Positive" (1) = Model Abstained
            # "Negative" (0) = Model Answered
            y_pred = merged['rlhf_abstain'].astype(int)

            # 3. Compute Matrix
            # Labels: [0, 1] -> [Should Answer, Should Abstain]
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            # 4. Calculate Precision and Recall
            tn, fp, fn, tp = cm.ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # 5. Create Custom Annotations (Large Text)
            annot_text = [
                f"TN: {tn}\n(Ref Right,\nAns)",     f"FP: {fp}\n(Ref Right,\nAbs)",
                f"FN: {fn}\n(Ref Wrong,\nAns)",     f"TP: {tp}\n(Ref Wrong,\nAbs)"
            ]
            annot_text = np.array(annot_text).reshape(2, 2)

            # 6. Plot Heatmap
            sns.heatmap(cm, annot=annot_text, fmt='', ax=axes[i], cmap='Blues', cbar=False,
                        annot_kws={"size": 18, "weight": "bold"},
                        xticklabels=['Answered', 'Abstained'],
                        yticklabels=['Ref Right', 'Ref Wrong'])

            # 7. Titles and Labels
            title_text = f"RLHF vs {ref.upper()}\nPrec: {precision:.1%} | Rec: {recall:.1%}"
            axes[i].set_title(title_text, fontsize=20, pad=20, fontweight='bold')
            axes[i].set_xlabel("RLHF Model Action", fontsize=16)
            axes[i].set_ylabel("Reference Model Outcome", fontsize=16)

            # Tick Label Size
            axes[i].tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        output_filename = 'rlhf_confusion_matrices.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\n✓ Confusion matrices saved to '{output_filename}'")
        plt.close()

    # Head-to-head with abstention SFT if available
    if 'abstention_sft' in baseline_loaded and 'abstention_sft_em' in merged.columns:
        print("\n" + "="*80)
        print("PART 5: RLHF vs SUPERVISED ABSTENTION TRAINING")
        print("="*80)

        # Both models have abstention capability
        rlhf_wins = merged[(merged['abstention_sft_em'] == 0) & (merged['rlhf_em'] == 1)]
        sft_wins = merged[(merged['abstention_sft_em'] == 1) & (merged['rlhf_em'] == 0)]
        both_correct = merged[(merged['abstention_sft_em'] == 1) & (merged['rlhf_em'] == 1)]
        both_wrong = merged[(merged['abstention_sft_em'] == 0) & (merged['rlhf_em'] == 0)]

        print(f"\nRLHF Improvements (SFT Wrong → RLHF Right): {len(rlhf_wins)}")
        print(f"SFT Wins (SFT Right → RLHF Wrong):          {len(sft_wins)}")
        print(f"Both Correct:                                {len(both_correct)}")
        print(f"Both Wrong:                                  {len(both_wrong)}")

        # Abstention comparison
        rlhf_abstained = merged['rlhf_abstain'].sum()
        sft_abstained = merged['abstention_sft_abstain'].sum()

        print(f"\nRLHF Abstentions:            {rlhf_abstained} ({rlhf_abstained/len(merged)*100:.1f}%)")
        print(f"Abstention SFT Abstentions:  {sft_abstained} ({sft_abstained/len(merged)*100:.1f}%)")

# Final summary
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
if os.path.exists('rlhf_abstention_analysis.csv'):
    print("  - rlhf_abstention_analysis.csv")
if os.path.exists('rlhf_confusion_matrices.png'):
    print("  - rlhf_confusion_matrices.png")
print("\n")
