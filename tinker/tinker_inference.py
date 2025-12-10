import os
import tinker
from tinker import types as ttypes
from dotenv import load_dotenv
# from tinker_inference import ADAPTER_PATH

load_dotenv()

# Configuration
# Use the adapter path from your training output
# ADAPTER_PATH (1024 batch size) = "tinker://62d8bafc-dd3e-5661-ba6f-0095855f6f9a:train:0/sampler_weights/qwen-hotpot-adapter"

# batch size: 128
ADAPTER_PATH = "tinker://0c59c24e-9098-51ca-b421-192c15a5a1f3:train:0/sampler_weights/qwen-hotpot-adapter"
BASE_MODEL = "Qwen/Qwen3-8B"

def main():
    print("🚀 Connecting to Tinker Service for Inference...")
    if not os.environ.get("TINKER_API_KEY"):
        print("⚠️ TINKER_API_KEY not found. Check .env file.")
        return

    service = tinker.ServiceClient()

    print(f"Initializing Sampling Client for {ADAPTER_PATH}...")
    # Create a sampling client using the saved adapter
    # We need to create a training client first to access the helper, or use service.create_sampling_client if available
    # Looking at docs, service.create_sampling_client exists.
    
    sampling_client = service.create_sampling_client(
        model_path=ADAPTER_PATH
    )

    # Get tokenizer from a temporary training client or just use huggingface if local
    # For simplicity, let's assume we can send raw text if the client supports it, 
    # but Tinker usually expects token IDs. 
    # Let's create a training client just to get the tokenizer easily, or use the one from the base model.
    print("Getting tokenizer...")
    # We can use the base model's tokenizer
    training_client = service.create_lora_training_client(base_model=BASE_MODEL, rank=16)
    tokenizer = training_client.get_tokenizer()
    
    # Test Questions (from HotpotQA or similar)
    questions = [
        "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "What nationality was the co-founder of the Distinguished Conduct Medal?",
        "Which magazine has published articles by Scott Baker, The New York Times Magazine or The Opium Magazine?"
        "What is the name of the first human to orbit the Earth?",
    ]

    print("\n--- Generating Responses ---")
    
    for q in questions:
        prompt_text = f"You are a concise encyclopedia. Answer the question directly with a short phrase or entity name. Do not explain.\nuser\n{q}\nassistant\n"
        
        # Tokenize
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        
        # Create Tinker inputs
        model_input = ttypes.ModelInput.from_ints(input_ids)
        sampling_params = ttypes.SamplingParams(max_tokens=32, temperature=0.1)
        
        # Generate
        # sample() returns an APIFuture
        result = sampling_client.sample(model_input, 1, sampling_params).result()
        
        # Decode
        # result.sequences is list[SampledSequence]
        output_ids = result.sequences[0].tokens
        output_text = tokenizer.decode(output_ids)
        
        print(f"\nQ: {q}")
        print(f"A: {output_text}")

if __name__ == "__main__":
    main()

