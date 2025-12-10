import os
from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker import types as ttypes
from datasets import load_dataset
import tqdm

# Configuration
MODEL_ID = "Qwen/Qwen3-8B"
DATASET_NAME = "hotpot_qa"
SUBSET_NAME = "distractor"
SYSTEM_PROMPT = "You are a concise encyclopedia. Answer the question directly with a short phrase or entity name. Do not explain."

# Training Hyperparameters
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 4
LORA_RANK = 16

def main():
    print("🚀 Connecting to Tinker Service...")
    if not os.environ.get("TINKER_API_KEY"):
        print("⚠️ TINKER_API_KEY not found. Check .env file.")
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
    
    # Get tokenizer
    tokenizer = training_client.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare Data
    print(f"Loading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split="train").select(range(80000))
    
    print("Tokenizing data...")
    training_data = []
    
    for row in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row['question']}
        ]
        # Generate prompt without answer
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Encode parts separately to create weights
        # Note: add_special_tokens=False because apply_chat_template handles them or we manage manually
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_tokens = tokenizer.encode(row['answer'] + tokenizer.eos_token, add_special_tokens=False)
        
        full_tokens = prompt_tokens + answer_tokens
        
        # Create weights: 0 for prompt, 1 for answer
        weights = [0] * len(prompt_tokens) + [1] * len(answer_tokens)
        
        # Shift for Causal LM (predict next token)
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]
        weights = weights[1:] # weights align with targets (the token being predicted)
        
        # Create Datum following the 'Pig Latin' example in docs exactly
        # Passing raw lists for weights and target_tokens
        datum = ttypes.Datum(
            model_input=ttypes.ModelInput.from_ints(input_tokens),
            loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
        )
        training_data.append(datum)

    # 3. Training Loop
    print(f"Starting Training: {len(training_data)} samples, {NUM_EPOCHS} epochs")
    
    optimizer_params = ttypes.AdamParams(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01 
    )

    step_counter = 0
    
    def get_batches(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        for batch in tqdm.tqdm(get_batches(training_data, BATCH_SIZE), total=len(training_data)//BATCH_SIZE):
            # A. Forward & Backward
            fwdbwd_future = training_client.forward_backward(
                data=batch,
                loss_fn="cross_entropy"
            )
            
            try:
                fwdbwd_result = fwdbwd_future.result()
            except Exception as e:
                print(f"❌ Batch Error: {e}")
                continue 
            
            # B. Optimize
            optim_future = training_client.optim_step(optimizer_params)
            optim_result = optim_future.result()
            
            step_counter += 1
            if step_counter % 10 == 0:
                # Calculate mean loss
                total_loss_sum = fwdbwd_result.metrics.get("loss:sum", 0.0)
                
                # Calculate total valid tokens in this batch (where weight=1)
                total_valid_tokens = sum(
                    sum(d.loss_fn_inputs['weights']) 
                    for d in batch
                )
                
                if total_valid_tokens > 0:
                    mean_loss = total_loss_sum / total_valid_tokens
                    print(f" Step {step_counter}: Loss = {mean_loss:.4f}")
                else:
                    print(f" Step {step_counter}: Loss = 0.0000 (No tokens)")

    print("Saving Adapter...")
    save_future = training_client.save_weights_for_sampler("qwen-hotpot-adapter")
    save_result = save_future.result()
    print(f"✅ Training Complete! Adapter saved to: {save_result.path}")

if __name__ == "__main__":
    main()
