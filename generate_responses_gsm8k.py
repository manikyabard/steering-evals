#!/usr/bin/env python3
"""
Generate responses from GSM8K dataset using SGLang.

This script generates and saves model responses from the GSM8K dataset
with thinking enabled to create datasets for identifying reasoning length direction.

Prerequisites:
Start the SGLang server in another terminal:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --port 30000 \
    --reasoning-parser qwen3
```
"""

# %% [markdown]
# # Generate Responses from GSM8K Dataset
#
# This notebook demonstrates how to generate model responses from the GSM8K math dataset
# with thinking enabled. We'll walk through the process step by step, from setting up
# the SGLang server connection to processing responses in parallel.
#
# ## Overview
# The process involves:
# 1. Setting up connection to SGLang server
# 2. Loading the GSM8K dataset
# 3. Generating responses with thinking enabled
# 4. Processing responses in parallel batches
# 5. Saving results for later analysis

# %% Setup and imports
import os
import json
import torch
import argparse
import requests
import concurrent.futures
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import sglang as sgl
from logging_setup import setup_logging, get_logger

# %% [markdown]
# ## Configuration and Arguments
#
# First, let's set up the configuration parameters for our generation process.


# %% Command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate responses from GSM8K dataset using SGLang"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="responses",
        help="Directory to save responses",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of samples to process from GSM8K",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--top_k",
        type=float,
        default=20,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of requests to process in parallel",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:30000",
        help="URL of the SGLang server",
    )
    return parser.parse_args()


# %% Interactive configuration (for notebook use)
# Uncomment and modify this cell when running interactively in a notebook
"""
# Test configuration for interactive development
test_args = type('Args', (), {
    'model': "Qwen/Qwen3-0.6B",
    'output_dir': "responses",
    'num_samples': 5,  # Small number for testing
    'max_new_tokens': 32768,
    'seed': 42,
    'temperature': 0.6,
    'top_p': 0.95,
    'top_k': 20,
    'batch_size': 4,
    'server_url': "http://localhost:30000"
})()

# Use test_args instead of parse_args() when running interactively
# args = test_args
"""

# %% [markdown]
# ## Step 1: Dataset Loading
#
# Let's start by loading the GSM8K dataset and examining its structure.


# %% Dataset loading functions
def load_gsm8k_dataset(num_samples, seed=42):
    """Load and prepare the GSM8K dataset."""
    print(f"Loading GSM8K with num_samples={num_samples}")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")

    if num_samples and num_samples < len(gsm8k):
        gsm8k = gsm8k.shuffle(seed=seed).select(range(num_samples))
        print(f"Selected {num_samples} examples after shuffling")

    return gsm8k


# %% Test data loading (uncomment to test interactively)
"""
def test_data_loading():
    \"\"\"Test loading a small sample of GSM8K data.\"\"\"
    try:
        print("Testing data loading...")
        dataset = load_gsm8k_dataset(num_samples=3, seed=42)
        print(f"✓ Loaded {len(dataset)} examples")
        
        # Show sample data
        for i, example in enumerate(dataset[:2]):
            print(f"\\nExample {i+1}:")
            print(f"  Question: {example['question']}")
            print(f"  Answer: {example['answer']}")
        
    except Exception as e:
        print(f"Data loading test failed: {e}")

# Uncomment to test:
# test_data_loading()
"""

# %% [markdown]
# ## Step 2: Server Connection
#
# Next, we need to establish a connection to the SGLang server.
# Make sure the server is running before proceeding.


# %% Server connection functions
def connect_to_sglang(server_url):
    """Test connection to the SGLang server."""
    print(f"Testing connection to SGLang server at {server_url}...")

    try:
        response = requests.get(f"{server_url}/health")
        response.raise_for_status()
        print(f"Successfully connected to SGLang server at {server_url}")
        return server_url
    except Exception as e:
        print(f"Error connecting to SGLang server: {e}")
        print("\nPlease ensure the SGLang server is running in another terminal.")
        raise


# %% Test server connection (uncomment to test interactively)
"""
def test_server_connection():
    \"\"\"Test connection to SGLang server.\"\"\"
    try:
        print("Testing server connection...")
        server_url = connect_to_sglang("http://localhost:30000")
        print(f"✓ Connected to SGLang server at {server_url}")
        
    except Exception as e:
        print(f"Server connection test failed: {e}")
        print("Make sure the SGLang server is running:")
        print("python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port 30000 --reasoning-parser qwen3")

# Uncomment to test:
# test_server_connection()
"""

# %% [markdown]
# ## Step 3: Response Generation
#
# Now we'll implement the core response generation functionality.
# This includes generating responses with thinking and parsing the results.


# %% Response generation functions
def generate_response(server_url, tokenizer, question):
    """Generate a response from the model with thinking enabled using direct HTTP requests."""
    prompt = f"Solve this math problem step by step, and put your final answer within \\boxed{{}}:\n{question}"

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    generate_payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
    }

    try:
        response = requests.post(f"{server_url}/generate", json=generate_payload)
        response.raise_for_status()
        result = response.json()
        generated_text = result["text"]

        # Use separate_reasoning to extract thinking and response
        reasoning_payload = {"text": generated_text, "reasoning_parser": "qwen3"}
        reasoning_response = requests.post(
            f"{server_url}/separate_reasoning", json=reasoning_payload
        )
        reasoning_response.raise_for_status()
        reasoning_result = reasoning_response.json()

        return {
            "thinking": reasoning_result.get("reasoning_text", ""),
            "response": reasoning_result.get("text", generated_text),
        }
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error in generation: {e}")
        if hasattr(e, "response") and hasattr(e.response, "text"):
            print(f"Response content: {e.response.text}")
        return {
            "thinking": "",
            "response": f"Error: {str(e)}",
        }
    except Exception as e:
        print(f"Error in generation: {e}")
        return {
            "thinking": "",
            "response": (
                generated_text if "generated_text" in locals() else f"Error: {str(e)}"
            ),
        }


# %% Test single generation (uncomment to test interactively)
"""
def test_single_example():
    \"\"\"Test with a simple math problem.\"\"\"
    try:
        print("Testing single example generation...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        
        # Test server connection
        response = requests.get("http://localhost:30000/health")
        response.raise_for_status()
        print(f"✓ Connected to SGLang server")
        
        test_question = "If there are 5 apples and 3 are eaten, how many remain?"
        
        # Note: This requires args to be defined globally
        global args
        args = type('Args', (), {
            'max_new_tokens': 1024,
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20
        })()
        
        result = generate_response("http://localhost:30000", tokenizer, test_question)
        
        print(f"\\nQuestion: {test_question}")
        print(f"Thinking length: {len(result['thinking'].split())} words")
        print(f"Thinking: {result['thinking'][:200]}...")
        print(f"Response: {result['response']}")
        
    except Exception as e:
        print(f"Test failed: {e}")

# Uncomment to test:
# test_single_example()
"""

# %% [markdown]
# ## Step 4: Batch Processing
#
# For efficiency, we'll process multiple examples in parallel batches.


# %% Batch processing functions
def process_batch(server_url, tokenizer, batch, start_idx):
    """Process a batch of examples in parallel."""
    results = []
    questions = batch["question"]
    answers = batch["answer"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(questions)) as executor:
        future_to_idx = {}

        for i, question in enumerate(questions):
            answer = answers[i] if i < len(answers) else ""
            future = executor.submit(generate_response, server_url, tokenizer, question)
            future_to_idx[future] = (i, question, answer)

        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(questions),
            desc=f"Batch starting at {start_idx}",
        ):
            i, question, answer = future_to_idx[future]
            try:
                thinking_result = future.result()

                output = {
                    "id": start_idx + i,
                    "question": question,
                    "answer": answer,
                    "with_thinking": thinking_result,
                }
                results.append(output)

            except Exception as e:
                print(f"Error processing example {start_idx + i}: {e}")
                output = {
                    "id": start_idx + i,
                    "question": question,
                    "answer": answer,
                    "with_thinking": {"thinking": "", "response": f"Error: {str(e)}"},
                    "error": str(e),
                }
                results.append(output)

    return results


# %% Test batch processing (uncomment to test interactively)
"""
def test_batch_processing():
    \"\"\"Test batch processing with a small sample.\"\"\"
    try:
        print("Testing batch processing...")
        
        # Load small dataset
        dataset = load_gsm8k_dataset(num_samples=3, seed=42)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        
        # Set up global args for generation
        global args
        args = type('Args', (), {
            'max_new_tokens': 1024,
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20
        })()
        
        # Process as a batch
        batch = {
            "question": [ex["question"] for ex in dataset],
            "answer": [ex["answer"] for ex in dataset]
        }
        
        results = process_batch("http://localhost:30000", tokenizer, batch, 0)
        
        print(f"✓ Processed {len(results)} examples in batch")
        
        for i, result in enumerate(results):
            thinking_len = len(result["with_thinking"]["thinking"].split())
            print(f"  Example {i+1}: {thinking_len} thinking words")
        
    except Exception as e:
        print(f"Batch processing test failed: {e}")

# Uncomment to test:
# test_batch_processing()
"""

# %% [markdown]
# ## Step 5: Main Processing Pipeline
#
# This brings everything together into a complete pipeline.


# %% Main processing function
def main(args):
    """Main function to process GSM8K examples and save responses in parallel."""
    logger = setup_logging("generate_responses_gsm8k")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    model_short_name = args.model.split("/")[-1]

    logger.info(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    try:
        server_url = connect_to_sglang(args.server_url)

        logger.info(f"Loading GSM8K dataset...")
        dataset = load_gsm8k_dataset(args.num_samples, args.seed)

        logger.info(f"Dataset has {len(dataset)} examples")

        if args.num_samples < len(dataset):
            dataset = dataset.select(range(args.num_samples))
            logger.info(f"Selected {args.num_samples} examples from the dataset")

        outputs = []
        num_examples = len(dataset)
        for i in range(0, num_examples, args.batch_size):
            end_idx = min(i + args.batch_size, num_examples)
            batch = dataset[i:end_idx]
            logger.info(
                f"Processing batch of {len(batch['question'])} examples starting at index {i}"
            )
            batch_results = process_batch(server_url, tokenizer, batch, i)
            outputs.extend(batch_results)

            # Save responses after each batch
            output_path = os.path.join(
                args.output_dir, f"{model_short_name}_gsm8k_responses.json"
            )
            with open(output_path, "w") as f:
                json.dump(outputs, f, indent=2)

            logger.info(f"Saved {len(outputs)} responses so far to {output_path}")

        logger.info(f"All responses saved to {output_path}")
        return outputs

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        import traceback

        traceback.print_exc()
        return []


# %% [markdown]
# ## Summary and Next Steps
#
# Congratulations! You've successfully generated model responses from GSM8K.
#
# **What we accomplished:**
# 1. Connected to the SGLang server
# 2. Loaded the GSM8K dataset
# 3. Generated responses with thinking enabled
# 4. Processed examples in parallel batches
# 5. Saved results for further analysis
#
# **Next steps:**
# - Use the generated responses with `extract_reasoning_length_direction_improved.py`
# - Analyze the distribution of thinking lengths
# - Experiment with different generation parameters
# - Scale up to larger datasets

# %% Script execution
if __name__ == "__main__":
    args = parse_args()
    outputs = main(args)
