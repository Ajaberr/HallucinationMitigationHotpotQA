import os
import tinker
from tinker import types as ttypes
from datasets import load_dataset
import tqdm
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. Configuration (Mirrored from qwen-abstain.py)
# ==========================================

# Teammate used Qwen2.5-7B. We use Tinker's supported Qwen3-8B.
# This is the closest equivalent on the platform.
MODEL_ID = "Qwen/Qwen3-8B" 
DATASET_ID = "fsiddiqui2/hotpotqa-abstention-70-30"

# Delimiters (EXACTLY mirroring teammate's script)
DELIMITER_PROMPT = " ###\n"
DELIMITER_END = " END"

# Training Hyperparameters
# Matched to qwen-abstain.py as closely as possible
LORA_RANK = 16          # Teammate used 16
LEARNING_RATE = 2e-4    # Teammate used 2e-4
BATCH_SIZE = 4          # Teammate used per_device_batch_size=4
GRAD_ACCUM = 4          # Teammate used grad_accum=4
LOGICAL_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM # 16

NUM_EPOCHS = 1
TRAINING_MODE = "COT" # Train on reasoning column

def main():
    print("🚀 Connecting to Tinker Service for Abstention Finetuning...")
    if not os.environ.get("TINKER_API_KEY"):
        print("⚠️ TINKER_API_KEY not found.")
        return

    service = tinker.ServiceClient()

    # 1. Initialize Training Client
    print(f"Initializing Training Client for {MODEL_ID}...")
    training_client = service.create_lora_training_client(
        base_model=MODEL_ID,
        rank=LORA_RANK,
        train_mlp=True,
        train_attn=True
    )
    
    tokenizer = training_client.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dataset
    print(f"Loading HF dataset: {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="train")
    print(f"Found {len(dataset)} examples.")

    # 3. Prepare Data (Mirrored formatting)
    print("Format & Tokenizing data (Manual Delimiters)...")
    training_data = []
    
    # Decide column
    target_col = "target" if TRAINING_MODE == "COT" else "short_target"

    for row in dataset:
        q = row['question']
        target = row[target_col]
        
        # --- A. Manual String Construction ---
        # "Question ###\n"
        prompt_text = f"{q}{DELIMITER_PROMPT}"
        # "Reasoning --> Answer END" (or I don't know END)
        completion_text = f"{target}{DELIMITER_END}"
        
        # --- B. Tokenize ---
        # Note: We do NOT use apply_chat_template because the teammate didn't.
        # We assume the base model can handle raw text (or we are training it to).
        
        # Encode with add_special_tokens=False to control EOS manually?
        # Usually prompt gets no EOS, completion gets EOS.
        # Teammate logic:
        # combined_ids = prompt_ids + completion_ids + [eos]
        
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
        
        # Combine + EOS
        full_ids = prompt_ids + completion_ids + [tokenizer.eos_token_id]
        
        # --- C. Create Weights (Mask Prompt) ---
        # 0 for prompt tokens, 1 for completion tokens (and EOS)
        weights = [0] * len(prompt_ids) + [1] * (len(completion_ids) + 1)
        
        # --- D. Shift for Causal LM ---
        # Inputs: [0...N-1]
        # Targets: [1...N] (The 'next' tokens)
        # Weights: correspond to 'targets' positions usually.
        # Tinker datum logic:
        # weights should be length of *output* (predictions), i.e., same len as target_tokens.
        
        input_tokens = full_ids[:-1]
        target_tokens = full_ids[1:]
        weights = weights[1:] 
        
        # Sanity check lengths
        if len(input_tokens) != len(target_tokens) or len(input_tokens) != len(weights):
            print("Length mismatch error!")
            continue

        datum = ttypes.Datum(
            model_input=ttypes.ModelInput.from_ints(input_tokens),
            loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
        )
        training_data.append(datum)

    print(f"Prepared {len(training_data)} training samples.")

    # 4. Training Loop
    optimizer_params = ttypes.AdamParams(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01 
    )

    step_counter = 0
    total_steps = (len(training_data) // LOGICAL_BATCH_SIZE) * NUM_EPOCHS
    
    def get_batches(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    print(f"Starting Training ({NUM_EPOCHS} Epochs, {total_steps} Steps)...")

    for epoch in range(NUM_EPOCHS):
        # Shuffle
        import random
        random.shuffle(training_data)
        
        pbar = tqdm.tqdm(get_batches(training_data, LOGICAL_BATCH_SIZE), total=len(training_data)//LOGICAL_BATCH_SIZE)
        
        for batch in pbar:
            # Skip incomplete batches often good practice
            if len(batch) < LOGICAL_BATCH_SIZE:
                continue

            # Forward/Backward
            fwdbwd_future = training_client.forward_backward(
                data=batch,
                loss_fn="cross_entropy"
            )
            try:
                fwdbwd_result = fwdbwd_future.result()
            except Exception as e:
                print(f"❌ Batch Error: {e}") 
                continue 
            
            # Step
            optim_future = training_client.optim_step(optimizer_params)
            optim_result = optim_future.result()
            
            step_counter += 1
            
            # Metrics
            loss_val = fwdbwd_result.metrics.get("loss:sum", 0.0)
            
            pbar.set_description(f"Ep {epoch+1} | Loss: {loss_val:.2f}")

    # 5. Save
    adapter_name = "qwen-hotpot-abstention-adapter"
    print(f"Saving Adapter as '{adapter_name}'...")
    save_future = training_client.save_weights_for_sampler(adapter_name)
    save_result = save_future.result()
    print(f"✅ Training Complete!")
    print(f"Adapter URI: {save_result.path}")
    
    # Append to registry file nicely if possible, or just print
    print("\nIMPORTANT: Add this to your tinker_models.json:")
    print(f'"{adapter_name}": {{ "description": "Abstention Finetune (Mirrored)", "uri": "{save_result.path}" }}')

if __name__ == "__main__":
    main()
