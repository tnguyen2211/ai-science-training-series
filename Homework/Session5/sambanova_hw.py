import time
from openai import OpenAI
from inference_auth_token import get_access_token
from datasets import load_dataset

# --- CONFIGURATION ---
NUM_SAMPLES = 5 

# Endpoint URLs

ENDPOINTS = {
    "Metis (SambaNova)": {
        "url": "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1",
        "model": "gpt-oss-120b-131072"  
    },
    "Sophia (NVIDIA)": {
        "url": "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        "model": "openai/gpt-oss-120b"  
    }
}


def run_benchmark():
    # 1. Get Auth Token
    try:
        token = get_access_token()
    except Exception as e:
        print(f"Auth Error: {e}")
        return

    # 2. Load Dataset
    print(f"Loading {NUM_SAMPLES} samples from HuggingFace IMDB dataset...")
    try:
        dataset = load_dataset("imdb", split=f"test[:{NUM_SAMPLES}]")
        inputs = [f"Summarize this review in one sentence: {row['text'][:1000]}" for row in dataset]
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    # 3. Run Comparison
    print("\nStarting Benchmark...")
    
    for backend_name, config in ENDPOINTS.items():
        print(f"\n--- Testing {backend_name} (Model: {config['model']}) ---")
        client = OpenAI(api_key=token, base_url=config['url'])

        start_time = time.time()
        total_tokens = 0
        success_count = 0

        for i, prompt in enumerate(inputs):
            try:
                response = client.chat.completions.create(
                    model=config['model'],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7 # Slight temp might prevent empty outputs on Metis
                )

                # Robustness Fix: Handle None content
                output_text = response.choices[0].message.content
                if output_text:
                    token_count = len(output_text.split())
                    total_tokens += token_count
                    success_count += 1
                    print(f"  [Req {i+1}] Success ({token_count} tokens)")
                else:
                    print(f"  [Req {i+1}] Failed: Empty response from model")

            except Exception as e:
                print(f"  [Req {i+1}] API Error: {e}")

        duration = time.time() - start_time
        tps = total_tokens / duration if duration > 0 else 0

        print(f"  > Duration: {duration:.2f}s")
        print(f"  > Throughput: {tps:.2f} tokens/sec")

if __name__ == "__main__":
    run_benchmark()
