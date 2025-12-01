#!/bin/bash

################################################################################
# RLHF Experiments Runner
# Comprehensive script for training and evaluating all model combinations
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
NUM_TRAIN_SAMPLES=100
NUM_EVAL_SAMPLES=500
DEVICE="cuda"
RESULTS_DIR="./results"
MODELS_DIR="./models"

# Create directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${MODELS_DIR}"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

################################################################################
# Training Functions
################################################################################

train_simple_rlhf() {
    print_header "Training Simple RLHF (Factuality Only)"
    python simple_rlhf.py
    print_success "Simple RLHF training completed"
}

train_base_rlhf() {
    print_header "Training Base RLHF (Factuality + Confidence)"
    python base_simple_reward.py
    print_success "Base RLHF training completed"
}

train_custom_rlhf() {
    print_header "Training Custom RLHF (Anti-Hallucination)"
    if [ -f "custom_simple_reward.py" ]; then
        python custom_simple_reward.py
        print_success "Custom RLHF training completed"
    else
        print_error "custom_simple_reward.py not found - skipping"
    fi
}

################################################################################
# Evaluation Functions
################################################################################

eval_baseline() {
    print_header "Evaluating Baseline Model (No RLHF)"
    # Use the RAUQ-enabled eval script from baseline_finetune
    cd ../baseline_finetune
    MODEL_ID="Qwen/Qwen2.5-7B-Instruct" NUM_SAMPLES=${NUM_EVAL_SAMPLES} python eval.py
    cd ../reward_learning
    print_success "Baseline evaluation completed"
}

eval_simple_rlhf() {
    print_header "Evaluating Simple RLHF Model"
    if [ -d "./simple_rlhf_policy" ]; then
        # Copy policy to a location eval.py can access
        print_error "Simple RLHF evaluation not yet integrated with RAUQ eval - skipping"
    else
        print_error "Simple RLHF policy not found - run training first"
    fi
}

eval_base_rlhf() {
    print_header "Evaluating Base RLHF Model"
    if [ -d "./verifier_rlhf_full_model" ]; then
        # Use the RAUQ-enabled eval script with full RLHF model
        cd ../baseline_finetune
        MODEL_ID="../reward_learning/verifier_rlhf_full_model" NUM_SAMPLES=${NUM_EVAL_SAMPLES} python eval.py
        cd ../reward_learning
        print_success "Base RLHF evaluation completed"
    else
        print_error "RLHF full model not found - run training first"
    fi
}

eval_custom_rlhf() {
    print_header "Evaluating Custom RLHF Model (Anti-Hallucination)"
    print_error "Custom RLHF evaluation not yet implemented - skipping"
}

eval_compare_all() {
    print_header "Comparing All Models with RAUQ"

    # Evaluate each model with RAUQ scoring
    cd ../baseline_finetune

    print_info "Evaluating Original Qwen..."
    MODEL_ID="Qwen/Qwen2.5-7B-Instruct" NUM_SAMPLES=${NUM_EVAL_SAMPLES} python eval.py

    print_info "Evaluating Fine-tuned Model..."
    MODEL_ID="fsiddiqui2/Qwen2.5-7B-Instruct-HotpotQA-Finetuned-10000" NUM_SAMPLES=${NUM_EVAL_SAMPLES} python eval.py

    if [ -d "../reward_learning/verifier_rlhf_full_model" ]; then
        print_info "Evaluating RLHF Model..."
        MODEL_ID="../reward_learning/verifier_rlhf_full_model" NUM_SAMPLES=${NUM_EVAL_SAMPLES} python eval.py
    fi

    cd ../reward_learning
    print_success "All models comparison completed - check baseline_finetune/ for timestamped JSON results"
}

eval_baseline_vs_base() {
    print_header "Comparing Baseline vs Base RLHF with RAUQ"

    cd ../baseline_finetune

    print_info "Evaluating Baseline..."
    MODEL_ID="Qwen/Qwen2.5-7B-Instruct" NUM_SAMPLES=${NUM_EVAL_SAMPLES} python eval.py

    if [ -d "../reward_learning/verifier_rlhf_full_model" ]; then
        print_info "Evaluating RLHF..."
        MODEL_ID="../reward_learning/verifier_rlhf_full_model" NUM_SAMPLES=${NUM_EVAL_SAMPLES} python eval.py
    fi

    cd ../reward_learning
    print_success "Baseline vs Base RLHF comparison completed"
}

eval_base_vs_custom() {
    print_header "Comparing Base RLHF vs Custom RLHF"
    print_error "Custom RLHF not implemented - skipping"
}

################################################################################
# Experiment Workflows
################################################################################

experiment_full_pipeline() {
    print_header "FULL EXPERIMENT PIPELINE"
    print_info "This will train all models and run all evaluations"

    # Training phase
    train_simple_rlhf
    train_base_rlhf
    train_custom_rlhf

    # Evaluation phase
    eval_baseline
    eval_simple_rlhf
    eval_base_rlhf
    eval_custom_rlhf
    eval_compare_all

    print_success "Full pipeline completed! Results saved to ${RESULTS_DIR}/"
}

experiment_train_only() {
    print_header "TRAINING ONLY - All Models"
    train_simple_rlhf
    train_base_rlhf
    train_custom_rlhf
    print_success "All training completed"
}

experiment_eval_only() {
    print_header "EVALUATION ONLY - All Models"
    eval_baseline
    eval_simple_rlhf
    eval_base_rlhf
    eval_custom_rlhf
    eval_compare_all
    print_success "All evaluations completed"
}

experiment_quick_test() {
    print_header "QUICK TEST - Baseline vs Base RLHF (100 samples)"
    NUM_EVAL_SAMPLES=100
    eval_baseline
    eval_base_rlhf
    eval_baseline_vs_base
    print_success "Quick test completed"
}

experiment_ablation_study() {
    print_header "ABLATION STUDY - Comparing Reward Components"
    print_info "1. Factuality only (Simple RLHF)"
    train_simple_rlhf
    eval_simple_rlhf

    print_info "2. Factuality + Confidence (Base RLHF)"
    train_base_rlhf
    eval_base_rlhf

    print_info "3. Factuality + Confidence + Anti-Hallucination (Custom RLHF)"
    train_custom_rlhf
    eval_custom_rlhf

    # Final comparison
    eval_compare_all

    print_success "Ablation study completed"
}

################################################################################
# Main Menu
################################################################################

show_menu() {
    echo ""
    print_header "RLHF EXPERIMENTS - SELECT OPTION"
    echo ""
    echo "TRAINING OPTIONS:"
    echo "  1)  Train Simple RLHF (Factuality only)"
    echo "  2)  Train Base RLHF (Factuality + Confidence)"
    echo "  3)  Train Custom RLHF (Anti-Hallucination)"
    echo "  4)  Train ALL models"
    echo ""
    echo "EVALUATION OPTIONS:"
    echo "  5)  Evaluate Baseline (no RLHF)"
    echo "  6)  Evaluate Simple RLHF"
    echo "  7)  Evaluate Base RLHF"
    echo "  8)  Evaluate Custom RLHF"
    echo "  9)  Evaluate ALL models (comparison)"
    echo "  10) Compare Baseline vs Base RLHF"
    echo "  11) Compare Base vs Custom RLHF"
    echo ""
    echo "EXPERIMENT WORKFLOWS:"
    echo "  12) Full Pipeline (train all + eval all)"
    echo "  13) Quick Test (baseline vs base, 100 samples)"
    echo "  14) Ablation Study (compare reward components)"
    echo "  15) Training Only (all models)"
    echo "  16) Evaluation Only (all models)"
    echo ""
    echo "  0)  Exit"
    echo ""
    echo -n "Enter your choice: "
}

################################################################################
# Command Line Arguments
################################################################################

if [ $# -gt 0 ]; then
    case "$1" in
        --full-pipeline)
            experiment_full_pipeline
            ;;
        --train-all)
            experiment_train_only
            ;;
        --eval-all)
            experiment_eval_only
            ;;
        --quick-test)
            experiment_quick_test
            ;;
        --ablation)
            experiment_ablation_study
            ;;
        --train-simple)
            train_simple_rlhf
            ;;
        --train-base)
            train_base_rlhf
            ;;
        --train-custom)
            train_custom_rlhf
            ;;
        --eval-baseline)
            eval_baseline
            ;;
        --eval-simple)
            eval_simple_rlhf
            ;;
        --eval-base)
            eval_base_rlhf
            ;;
        --eval-custom)
            eval_custom_rlhf
            ;;
        --compare-all)
            eval_compare_all
            ;;
        --help)
            echo "Usage: $0 [OPTION]"
            echo ""
            echo "Options:"
            echo "  --full-pipeline    Run complete pipeline (train + eval all)"
            echo "  --train-all        Train all models"
            echo "  --eval-all         Evaluate all models"
            echo "  --quick-test       Quick test (baseline vs base, 100 samples)"
            echo "  --ablation         Ablation study (compare reward components)"
            echo "  --train-simple     Train Simple RLHF only"
            echo "  --train-base       Train Base RLHF only"
            echo "  --train-custom     Train Custom RLHF only"
            echo "  --eval-baseline    Evaluate baseline only"
            echo "  --eval-simple      Evaluate Simple RLHF only"
            echo "  --eval-base        Evaluate Base RLHF only"
            echo "  --eval-custom      Evaluate Custom RLHF only"
            echo "  --compare-all      Compare all models"
            echo "  --help             Show this help message"
            echo ""
            echo "If no option is provided, interactive menu will be shown."
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
    exit 0
fi

################################################################################
# Interactive Menu
################################################################################

while true; do
    show_menu
    read choice

    case $choice in
        1) train_simple_rlhf ;;
        2) train_base_rlhf ;;
        3) train_custom_rlhf ;;
        4) experiment_train_only ;;
        5) eval_baseline ;;
        6) eval_simple_rlhf ;;
        7) eval_base_rlhf ;;
        8) eval_custom_rlhf ;;
        9) eval_compare_all ;;
        10) eval_baseline_vs_base ;;
        11) eval_base_vs_custom ;;
        12) experiment_full_pipeline ;;
        13) experiment_quick_test ;;
        14) experiment_ablation_study ;;
        15) experiment_train_only ;;
        16) experiment_eval_only ;;
        0)
            print_info "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid option: $choice"
            ;;
    esac

    echo ""
    echo -n "Press Enter to continue..."
    read
done
