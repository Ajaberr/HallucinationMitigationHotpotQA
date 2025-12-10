from dotenv import load_dotenv
load_dotenv()

import os
import tinker
from tinker import types as ttypes
from datasets import load_dataset
import tqdm

# --- Configuration ---
# We use Qwen3-8B because it is supported by Tinker and comparable to the 7B model you used.
# It is much faster to train.
MODEL_ID = "Qwen/Qwen3-8B" 
DATASET_NAME = "fsiddiqui2/hotpot-qa-cot-reasoning"
# No subset name needed for private generic datasets usually, or split="train"
# If it fails, we might need to check if there is a 'default' config.

# Instructions for the student model. 
# We keep it simpler than the teacher's prompt since the student doesn't get context.
SYSTEM_PROMPT = "You are an expert Professor of Knowledge Graph Reasoning."
USER_PROMPT_TEMPLATE = "Answer the following question with a step-by-step reasoning trace.\n\nQuestion: {question}"

# Training Hyperparameters
# LoRA Rank 16-32 is usually sufficient for distillation.
LORA_RANK = 32
LEARNING_RATE = 2e-4 # Higher for LoRA
NUM_EPOCHS = 1 # 1-3 epochs is standard for distillation
BATCH_SIZE = 4 # Adjust based on GPU memory/throughput if needed, but Tinker handles it.
# Actually Tinker batch size is dataset logical batch size. 64-128 is good for SGD.
# But we need to feed "batches" to forward_backward.
LOGICAL_BATCH_SIZE = 32 

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
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have uploaded the dataset to HuggingFace!")
        return
        
    print(f"Found {len(dataset)} examples.")
    
    print("Tokenizing data...")
    training_data = []
    
    for row in dataset:
        # Check if reasoning_trace exists (it should based on teacher_hotpot.py)
        if 'reasoning_trace' not in row or not row['reasoning_trace']:
            continue
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=row['question'])}
        ]
        
        # 1. Encode Prompt
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        
        # 2. Encode Answer (The CoT Trace)
        # We append EOS to the answer
        answer_text = row['reasoning_trace'] + tokenizer.eos_token
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        
        full_tokens = prompt_tokens + answer_tokens
        
        # 3. Create Weights (Mask Prompt, Train Answer)
        weights = [0] * len(prompt_tokens) + [1] * len(answer_tokens)
        
        # 4. Shift for Causal LM (Next Token Prediction)
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]
        weights = weights[1:] 
        
        # 5. Create Datum
        datum = ttypes.Datum(
            model_input=ttypes.ModelInput.from_ints(input_tokens),
            loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
        )
        training_data.append(datum)

    print(f"Prepared {len(training_data)} training samples.")

    # 3. Training Loop
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
        
        # Shuffle data each epoch (optional but good practice)
        import random
        random.shuffle(training_data)
        
        pbar = tqdm.tqdm(get_batches(training_data, LOGICAL_BATCH_SIZE), total=len(training_data)//LOGICAL_BATCH_SIZE)
        
        for batch in pbar:
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
            
            # Metrics
            loss_val = fwdbwd_result.metrics.get("loss:sum", 0.0)
            # Normalize loss by number of tokens (approximate) or just rely on sum
            # Tinker usually returns sum of losses.
            
            if step_counter % 5 == 0:
                pbar.set_description(f"Loss: {loss_val:.2f}")

    # 4. Save
    adapter_name = "qwen-cot-distilled"
    print(f"Saving Adapter as '{adapter_name}'...")
    save_future = training_client.save_weights_for_sampler(adapter_name)
    save_result = save_future.result()
    print(f"✅ Training Complete! Adapter saved to: {save_result.path}")
    print(f"📝 NOTE: Use this path in your RL script: {save_result.path}")

if __name__ == "__main__":
    main()
