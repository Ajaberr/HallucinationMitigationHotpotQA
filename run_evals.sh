#!/bin/bash

# Print start message
echo "Starting RAUQ Evaluation pipeline..."
echo "-----------------------------------"

# # 1. Run Baseline Finetune
# echo "[1/4] Running baseline_finetune/eval_rauq.py..."
# python baseline_finetune/eval_rauq.py || { echo "Baseline Finetune failed"; exit 1; }

# 2. Run Baseline Vanilla
echo "[2/4] Running baseline_vanilla/eval_rauq.py..."
python baseline_vanilla/eval_rauq.py || { echo "Baseline Vanilla failed"; exit 1; }

# 4. Run Finetune Abstention
echo "[3/4] Running finetune_abstention/eval_rauq.py..."
python finetune_abstention/eval_rauq.py || { echo "Finetune Abstention failed"; exit 1; }

# 3. Run Finetune CoT
echo "[4/4] Running finetune-cot/eval_rauq.py..."
python finetune-cot/eval_rauq.py || { echo "Finetune CoT failed"; exit 1; }

echo "-----------------------------------"
echo "All evaluations completed successfully!"