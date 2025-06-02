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

Resume Options:
- Use --resume-from-id N to resume from a specific example ID
- Use --responses-file path/to/file.json to specify a particular response file 
  (useful if the file was renamed or moved)
- Use --force-resume with --resume-from-id to start from a specific ID even if 
  no existing file is found
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
import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import sglang as sgl
from logging_setup import setup_logging, get_logger

# GSM8K answer parsing patterns
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

# %% [markdown]
# ## Configuration and Arguments
#
# First, let's set up the configuration parameters for our generation process.


# %% Command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Generate GSM8K responses")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="responses",
        help="Directory to save responses",
    )
    parser.add_argument(
        "--num-samples", type=int, default=2000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32768,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:30000",
        help="SGLang server URL",
    )
    parser.add_argument(
        "--resume-from-id",
        type=int,
        default=None,
        help="Resume generation from specific sample ID",
    )
    parser.add_argument(
        "--force-resume",
        action="store_true",
        help="Force resume even if ID doesn't exist in current responses",
    )
    parser.add_argument(
        "--responses-file",
        type=str,
        default=None,
        help="Specific responses file to resume from or save to (overrides auto-generated filename)",
    )
    return parser.parse_args()


# %% Answer parsing functions
def extract_answer_hf(completion):
    """Extract answer using GSM8K format (#### answer)."""
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            return eval(match_str)
        except:
            return INVALID_ANS
    else:
        return INVALID_ANS


def extract_answer_fallback(completion):
    """Fallback: extract last number from completion."""
    try:
        # Look for boxed answer first
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(boxed_pattern, completion)
        if match:
            boxed_content = match.group(1).strip()
            # Try to extract number from boxed content
            numbers = re.findall(r"-?\d+(?:\.\d+)?", boxed_content)
            if numbers:
                return eval(numbers[-1])
        
        # Fallback to last number in text
        numbers = re.findall(r"-?\d+(?:\.\d+)?", completion)
        if numbers:
            return eval(numbers[-1])
        return INVALID_ANS
    except:
        return INVALID_ANS


def extract_answer(completion):
    """Extract answer from completion using multiple strategies."""
    # Try GSM8K format first
    answer = extract_answer_hf(completion)
    if answer != INVALID_ANS:
        return answer
    
    # Try fallback method
    return extract_answer_fallback(completion)


def is_correct(completion, ground_truth_answer):
    """Check if completion answer matches ground truth."""
    try:
        # Extract ground truth
        gold = extract_answer_hf(ground_truth_answer)
        if gold == INVALID_ANS:
            # Try to extract from ground truth using fallback
            gold = extract_answer_fallback(ground_truth_answer)
        
        if gold == INVALID_ANS:
            return False
        
        # Extract predicted answer
        predicted = extract_answer(completion)
        
        if predicted == INVALID_ANS:
            return False
        
        # Compare (handle floating point comparison)
        if isinstance(predicted, float) or isinstance(gold, float):
            return abs(predicted - gold) < 1e-6
        else:
            return predicted == gold
            
    except Exception as e:
        return False


# %% Interactive configuration (for notebook use)
# Uncomment and modify this cell when running interactively in a notebook
"""
# Test configuration for interactive development
test_args = type('Args', (), {
    'model': "Qwen/Qwen3-0.6B",
    'output_dir': "responses",
    'num_samples': 5,  # Small number for testing
    'max_new_tokens': 32768,
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
def load_gsm8k_dataset(num_samples):
    """Load and prepare the GSM8K dataset."""
    print(f"Loading GSM8K with num_samples={num_samples}")
    gsm8k = load_dataset("openai/gsm8k", "main", split=f"train[:{num_samples}]")

    return gsm8k


# %% Test data loading (uncomment to test interactively)
"""
def test_data_loading():
    \"\"\"Test loading a small sample of GSM8K data.\"\"\"
    try:
        print("Testing data loading...")
        dataset = load_gsm8k_dataset(num_samples=3)
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
def generate_response(server_url, tokenizer, question, ground_truth_answer):
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

        thinking_text = reasoning_result.get("reasoning_text", "")
        response_text = reasoning_result.get("text", generated_text)
        
        # Extract answer and check correctness
        extracted_answer = extract_answer(response_text)
        correct = is_correct(response_text, ground_truth_answer)

        return {
            "thinking": thinking_text,
            "response": response_text,
            "extracted_answer": extracted_answer,
            "correct": correct,
            "full_text": generated_text
        }
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error in generation: {e}")
        if hasattr(e, "response") and hasattr(e.response, "text"):
            print(f"Response content: {e.response.text}")
        return {
            "thinking": "",
            "response": f"Error: {str(e)}",
            "extracted_answer": INVALID_ANS,
            "correct": False,
            "full_text": f"Error: {str(e)}"
        }
    except Exception as e:
        print(f"Error in generation: {e}")
        return {
            "thinking": "",
            "response": (
                generated_text if "generated_text" in locals() else f"Error: {str(e)}"
            ),
            "extracted_answer": INVALID_ANS,
            "correct": False,
            "full_text": (
                generated_text if "generated_text" in locals() else f"Error: {str(e)}"
            )
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
        test_answer = "#### 2"
        
        # Note: This requires args to be defined globally
        global args
        args = type('Args', (), {
            'max_new_tokens': 1024,
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20
        })()
        
        result = generate_response("http://localhost:30000", tokenizer, test_question, test_answer)
        
        print(f"\\nQuestion: {test_question}")
        print(f"Thinking length: {len(result['thinking'].split())} words")
        print(f"Thinking: {result['thinking'][:200]}...")
        print(f"Response: {result['response']}")
        print(f"Extracted answer: {result['extracted_answer']}")
        print(f"Correct: {result['correct']}")
        
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
def process_batch(server_url, tokenizer, batch, start_idx, processed_ids=None):
    """Process a batch of examples in parallel."""
    if processed_ids is None:
        processed_ids = set()
    
    results = []
    
    # Handle both old format (questions/answers lists) and new format (list of dicts)
    if isinstance(batch, dict):
        # Old format: batch is {"question": [...], "answer": [...]}
        questions = batch["question"]
        answers = batch["answer"]
        batch_items = [(i + start_idx, questions[i], answers[i]) for i in range(len(questions))]
    else:
        # New format: batch is [{"original_id": ..., "question": ..., "answer": ...}, ...]
        batch_items = [(item["original_id"], item["question"], item["answer"]) for item in batch]

    # Filter out already processed items
    batch_items = [(item_id, question, answer) for item_id, question, answer in batch_items 
                   if item_id not in processed_ids]
    
    if not batch_items:
        # All items in this batch were already processed
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_items)) as executor:
        future_to_data = {}

        for item_id, question, answer in batch_items:
            future = executor.submit(generate_response, server_url, tokenizer, question, answer)
            future_to_data[future] = (item_id, question, answer)

        for future in tqdm(
            concurrent.futures.as_completed(future_to_data),
            total=len(batch_items),
            desc=f"Processing batch",
        ):
            item_id, question, answer = future_to_data[future]
            try:
                thinking_result = future.result()

                output = {
                    "id": item_id,
                    "question": question,
                    "answer": answer,
                    "with_thinking": thinking_result,
                }
                results.append(output)

            except Exception as e:
                print(f"Error processing example {item_id}: {e}")
                output = {
                    "id": item_id,
                    "question": question,
                    "answer": answer,
                    "with_thinking": {
                        "thinking": "", 
                        "response": f"Error: {str(e)}",
                        "extracted_answer": INVALID_ANS,
                        "correct": False,
                        "full_text": f"Error: {str(e)}"
                    },
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
        dataset = load_gsm8k_dataset(num_samples=3)
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
            correct = result["with_thinking"]["correct"]
            print(f"  Example {i+1}: {thinking_len} thinking words, Correct: {correct}")
        
    except Exception as e:
        print(f"Batch processing test failed: {e}")

# Uncomment to test:
# test_batch_processing()
"""

# %% [markdown]
# ## Step 5: Main Processing Pipeline
#
# This brings everything together into a complete pipeline.


# %% Resume functionality
def load_existing_responses(output_path):
    """Load existing responses from output file if it exists."""
    if not os.path.exists(output_path):
        return []
    
    try:
        with open(output_path, 'r') as f:
            responses = json.load(f)
        return responses
    except Exception as e:
        print(f"Warning: Could not load existing responses from {output_path}: {e}")
        return []


def get_resume_point(existing_responses, resume_from_id=None):
    """Determine the resume point based on existing responses."""
    if not existing_responses:
        return 0
    
    if resume_from_id is not None:
        return resume_from_id
    
    # Auto-detect: find the highest ID + 1
    max_id = max(resp.get('id', -1) for resp in existing_responses)
    return max_id + 1


def create_processed_ids_set(existing_responses):
    """Create a set of already processed IDs for quick lookup."""
    return set(resp.get('id', -1) for resp in existing_responses)


def filter_dataset_for_resume(dataset, resume_point, processed_ids=None):
    """Filter dataset to only include examples that need processing."""
    if processed_ids is None:
        processed_ids = set()
    
    filtered_examples = []
    for i, example in enumerate(dataset):
        example_id = i
        if example_id >= resume_point or example_id not in processed_ids:
            filtered_examples.append({
                'original_id': example_id,
                'question': example['question'],
                'answer': example['answer']
            })
    
    return filtered_examples


# %% Main processing function
def main(args):
    """Main function to process GSM8K examples and save responses in parallel."""
    logger = setup_logging("generate_responses_gsm8k")

    # Determine output path and create necessary directories
    if args.responses_file:
        output_path = args.responses_file
        # Create directory for the specified file if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created directory for responses file: {output_dir}")
        logger.info(f"Using specified responses file: {output_path}")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        model_short_name = args.model.split("/")[-1]
        output_path = os.path.join(
            args.output_dir, f"{model_short_name}_gsm8k_responses.json"
        )
        logger.info(f"Using auto-generated responses file: {output_path}")
    
    logger.info(f"Output directory: {os.path.dirname(output_path) or '.'}")

    # Resume functionality
    existing_responses = []
    if args.resume_from_id is not None or os.path.exists(output_path):
        logger.info("Checking for existing responses...")
        existing_responses = load_existing_responses(output_path)
        
        if existing_responses:
            logger.info(f"Found {len(existing_responses)} existing responses")
        elif args.resume_from_id is not None and not args.force_resume:
            logger.warning(f"Resume requested but no existing file found at {output_path}")
            logger.warning("Use --force-resume to start from scratch at the specified ID")
            return []

    # Determine resume point
    resume_point = get_resume_point(existing_responses, args.resume_from_id)
    processed_ids = create_processed_ids_set(existing_responses)
    
    if resume_point > 0 or processed_ids:
        logger.info(f"Resuming from ID {resume_point}")
        logger.info(f"Already processed {len(processed_ids)} examples")
    else:
        logger.info("Starting fresh processing")

    logger.info(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    try:
        server_url = connect_to_sglang(args.server_url)

        logger.info(f"Loading GSM8K dataset...")
        dataset = load_gsm8k_dataset(args.num_samples)

        logger.info(f"Dataset has {len(dataset)} examples")

        if args.num_samples < len(dataset):
            dataset = dataset.select(range(args.num_samples))
            logger.info(f"Selected {args.num_samples} examples from the dataset")

        # Filter dataset for resume
        if resume_point > 0 or processed_ids:
            remaining_examples = filter_dataset_for_resume(dataset, resume_point, processed_ids)
            logger.info(f"Found {len(remaining_examples)} examples to process (after resume filtering)")
            
            if not remaining_examples:
                logger.info("All examples have already been processed!")
                return existing_responses
        else:
            # Convert dataset to the new format for consistency
            remaining_examples = [
                {
                    'original_id': i,
                    'question': dataset[i]['question'],
                    'answer': dataset[i]['answer']
                }
                for i in range(len(dataset))
            ]

        # Initialize outputs with existing responses
        outputs = existing_responses.copy()
        
        # Process remaining examples in batches
        num_remaining = len(remaining_examples)
        for i in range(0, num_remaining, args.batch_size):
            end_idx = min(i + args.batch_size, num_remaining)
            batch = remaining_examples[i:end_idx]
            
            # Log batch info
            batch_ids = [item['original_id'] for item in batch]
            logger.info(
                f"Processing batch of {len(batch)} examples (IDs: {min(batch_ids)}-{max(batch_ids)})"
            )
            
            batch_results = process_batch(server_url, tokenizer, batch, 0, processed_ids)
            
            # Add new results to outputs
            outputs.extend(batch_results)
            
            # Update processed_ids to avoid reprocessing in case of errors
            for result in batch_results:
                processed_ids.add(result['id'])

            # Save responses after each batch
            try:
                # Sort outputs by ID to maintain order
                outputs_sorted = sorted(outputs, key=lambda x: x.get('id', 0))
                with open(output_path, "w") as f:
                    json.dump(outputs_sorted, f, indent=2)

                # Calculate and log accuracy so far
                correct_count = sum(1 for o in outputs_sorted if o.get("with_thinking", {}).get("correct", False))
                accuracy = correct_count / len(outputs_sorted) if outputs_sorted else 0
                logger.info(f"Saved {len(outputs_sorted)} responses so far to {output_path}")
                logger.info(f"Current accuracy: {accuracy:.2%} ({correct_count}/{len(outputs_sorted)})")
                
            except Exception as e:
                logger.error(f"Error saving responses: {e}")
                # Continue processing even if save fails

        # Final statistics and save
        outputs_sorted = sorted(outputs, key=lambda x: x.get('id', 0))
        correct_count = sum(1 for o in outputs_sorted if o.get("with_thinking", {}).get("correct", False))
        accuracy = correct_count / len(outputs_sorted) if outputs_sorted else 0
        thinking_lengths = [len(o.get("with_thinking", {}).get("thinking", "").split()) for o in outputs_sorted]
        avg_thinking_length = np.mean(thinking_lengths) if thinking_lengths else 0
        
        logger.info(f"=== FINAL RESULTS ===")
        logger.info(f"Total examples: {len(outputs_sorted)}")
        logger.info(f"Final accuracy: {accuracy:.2%} ({correct_count}/{len(outputs_sorted)})")
        logger.info(f"Average thinking length: {avg_thinking_length:.1f} words")
        logger.info(f"All responses saved to {output_path}")
        
        return outputs_sorted

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
# **Resume options:**
# - Resume from auto-generated file: `--resume-from-id 500`
# - Resume from specific file: `--responses-file path/to/my_responses.json`
# - Combine both: `--responses-file path/to/my_responses.json --resume-from-id 500`
# - Force resume from ID: `--resume-from-id 500 --force-resume`
#
# **Example usage:**
# ```bash
# # Start fresh generation with custom output file
# python generate_responses_gsm8k.py --responses-file /path/to/my_custom_responses.json
# 
# # Resume from a renamed/moved file
# python generate_responses_gsm8k.py --responses-file /path/to/renamed_responses.json
# 
# # Resume from specific ID in custom file
# python generate_responses_gsm8k.py --responses-file /path/to/my_responses.json --resume-from-id 1000
# ```
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
