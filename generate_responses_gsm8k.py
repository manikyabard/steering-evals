# %% [markdown]
# # Generating Responses from GSM8K
#
# This notebook generates and saves model responses from the GSM8K dataset.
# We'll generate responses with thinking enabled to create our dataset
# for identifying reasoning length direction.
#
# The notebook follows these steps:
# 1. Load the GSM8K dataset
# 2. Generate responses with thinking enabled
# 3. Save the responses for later analysis
#
# This implementation uses SGLang for faster inference and parallelization:
# - Using appropriate sampling parameters (Temperature=0.6, TopP=0.95, TopK=20)
# - Including step-by-step reasoning instructions
# - Using the reasoning parser to separate thinking from responses
# - Batch processing multiple examples in parallel
# - Connecting to an existing SGLang server

# %% [markdown]
# ## Prerequisites
#
# Before running this script, start the SGLang server in another terminal:
#
# ```bash
# python -m sglang.launch_server \
#     --model-path Qwen/Qwen3-0.6B \
#     --port 30000 \
#     --reasoning-parser qwen3 \
# ```
#
# You can add other server arguments as needed:
# - `--attention-backend fa3` to use the FA3 attention backend instead of FlashInfer
# - `--mem-fraction-static 0.7` to reduce memory usage if you're getting OOM errors
#
# See the full list of server options at https://docs.sglang.ai/backend/server_arguments.html

# %% [markdown]
# ## Setup
#
# First, let's import the necessary libraries and set up the argument parser.

# %%
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

# Display installed versions for reproducibility
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
try:
    import transformers

    print(f"Transformers version: {transformers.__version__}")
except:
    print("Transformers not installed")
try:
    import sglang

    print(f"SGLang version: {sglang.__version__}")
except:
    print("SGLang not installed")

# %% [markdown]
# ## Command Line Arguments
#
# When running this as a script, you can provide command-line arguments.
# In notebook mode, we'll define default values that you can modify in the next cell.


# %%
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
        default=2000,  # Similar to ThinkEdit paper
        help="Number of samples to process from GSM8K",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,  # Recommended for Qwen models
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,  # Recommended for thinking mode
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,  # Recommended for thinking mode
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--top_k",
        type=float,
        default=20,  # Recommended for thinking mode
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


# %% [markdown]
# ## Interactive Configuration
#
# If you're running this as a notebook, you can modify these parameters directly.
# Change the values in this cell to customize your experiment.


# %%
# Interactive notebook parameters - modify these values as needed
class NotebookArgs:
    def __init__(self):
        self.model = "Qwen/Qwen3-0.6B"  # Model to use
        self.output_dir = "responses"  # Directory to save responses
        self.num_samples = 5  # Use a small number for quick testing
        self.max_new_tokens = 32768  # Recommended for Qwen models
        self.seed = 42  # Random seed for reproducibility
        self.temperature = 0.6  # Recommended for thinking mode
        self.top_p = 0.95  # Recommended for thinking mode
        self.top_k = 20  # Recommended for thinking mode
        self.batch_size = 4  # Process multiple requests in parallel
        self.server_url = "http://localhost:30000"  # URL of the SGLang server


# Use NotebookArgs when running as notebook, otherwise parse command line arguments
import sys

if "ipykernel" in sys.modules:
    args = NotebookArgs()
    print("Running in notebook mode with these parameters:")
    print(f"- Model: {args.model}")
    print(f"- Number of samples: {args.num_samples}")
    print(f"- Output directory: {args.output_dir}")
    print(f"- Temperature: {args.temperature}")
    print(f"- Top-p: {args.top_p}")
    print(f"- Top-k: {args.top_k}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Server URL: {args.server_url}")
else:
    args = parse_args()
    print("Running in script mode with parsed arguments")

# %% [markdown]
# ## Load and Prepare the GSM8K Dataset
#
# Now we'll load the GSM8K dataset and take a subset for our experiments.


# %%
def load_gsm8k_dataset(num_samples, seed=42):
    """Load and prepare the GSM8K dataset."""
    print(f"Loading GSM8K with num_samples={num_samples}")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")

    # Shuffle and select subset if needed
    if num_samples and num_samples < len(gsm8k):
        gsm8k = gsm8k.shuffle(seed=seed).select(range(num_samples))
        print(f"Selected {num_samples} examples after shuffling")

    return gsm8k


# %%
# Let's load the dataset and examine a sample
dataset = load_gsm8k_dataset(args.num_samples, args.seed)
print(f"Loaded {len(dataset)} examples from GSM8K")

# Display an example
if len(dataset) > 0:
    example = dataset[0]
    print("\nExample problem:")
    print("-" * 50)
    print(f"Question: {example['question']}")
    print(f"Answer: {example['answer']}")

# %% [markdown]
# ## Connect to SGLang Server
#
# Let's connect to the SGLang server that's already running.


# %%
def connect_to_sglang(server_url):
    """Test connection to the SGLang server."""
    print(f"Testing connection to SGLang server at {server_url}...")

    try:
        # Test the connection with a health check
        response = requests.get(f"{server_url}/health")
        response.raise_for_status()
        print(f"Successfully connected to SGLang server at {server_url}")

        return server_url
    except Exception as e:
        print(f"Error connecting to SGLang server: {e}")
        print("\nPlease ensure the SGLang server is running in another terminal with:")
        print(
            f"""
python -m sglang.launch_server \\
    --model-path {args.model} \\
    --host 0.0.0.0 \\
    --port 30000 \\
    --reasoning-parser qwen3 \\
    --disable-cuda-graph
        """
        )
        raise


# %% [markdown]
# ## Generate Responses
#
# Now let's define a function to generate responses from our model with thinking enabled
# using SGLang's reasoning parser capabilities.


# %%
def generate_response(server_url, tokenizer, question):
    """Generate a response from the model with thinking enabled using direct HTTP requests."""
    # Include step-by-step reasoning instructions as recommended
    prompt = f"Solve this math problem step by step, and put your final answer within \\boxed{{}}:\n{question}"

    messages = [{"role": "user", "content": prompt}]

    # Format the prompt using the model's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Define request payload for generation
    generate_payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
    }

    # Generate with SGLang's API via direct HTTP request
    try:
        # Generate the text
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
        # If there's an issue with the reasoning parser, try to return the full text
        return {
            "thinking": "",
            "response": (
                generated_text if "generated_text" in locals() else f"Error: {str(e)}"
            ),
        }


# %% [markdown]
# ## Process Multiple Requests in Parallel
#
# Let's define a function to process batches of examples in parallel.


# %%
def process_batch(server_url, tokenizer, batch, start_idx):
    """Process a batch of examples in parallel."""
    results = []

    # Extract the lists of questions and answers from the batch
    questions = batch["question"]
    answers = batch["answer"]

    # Create a thread pool to process the batch
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(questions)) as executor:
        future_to_idx = {}

        for i, question in enumerate(questions):
            answer = answers[i] if i < len(answers) else ""

            # Submit the task to the executor
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

                # Save the results
                output = {
                    "id": start_idx + i,
                    "question": question,
                    "answer": answer,
                    "with_thinking": thinking_result,
                }
                results.append(output)

            except Exception as e:
                print(f"Error processing example {start_idx + i}: {e}")
                # Add a placeholder for failed requests
                output = {
                    "id": start_idx + i,
                    "question": question,
                    "answer": answer,
                    "with_thinking": {"thinking": "", "response": f"Error: {str(e)}"},
                    "error": str(e),
                }
                results.append(output)

    return results


# %% [markdown]
# ## Test the Model on a Single Example
#
# Let's first test the model on a single example to make sure everything is working properly.

# %%
# Test with a simple problem
try:
    print("Setting up connection to SGLang server...")
    # Initialize tokenizer directly
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Connect to the SGLang server
    server_url = args.server_url

    # Test server connection
    response = requests.get(f"{server_url}/health")
    response.raise_for_status()
    print(f"Successfully connected to SGLang server at {server_url}")

    test_question = "If there are 5 apples and 3 are eaten, how many remain?"

    # Generate with thinking enabled
    print("\nGenerating response with thinking enabled...")

    # Use the same generate_response function as used for the full dataset
    thinking_result = generate_response(server_url, tokenizer, test_question)

    print("\nThinking content:")
    print("-" * 50)
    print(thinking_result["thinking"])

    print("\nResponse content:")
    print("-" * 50)
    print(thinking_result["response"])

except Exception as e:
    print(f"Error testing the model: {e}")
    print(f"Type of error: {type(e)}")
    if hasattr(e, "response") and hasattr(e.response, "text"):
        print(f"Response content: {e.response.text}")
    print(
        "Please ensure the SGLang server is running in another terminal before executing this script."
    )

# %% [markdown]
# ## Main Function
#
# Now let's define the main function to process the whole dataset and save the responses.


# %%
def main(args):
    """Main function to process GSM8K examples and save responses in parallel."""
    # Setup logging
    logger = setup_logging("generate_responses_gsm8k")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    model_short_name = args.model.split("/")[-1]

    # Load tokenizer for creating prompts
    logger.info(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Connect to SGLang server
    try:
        server_url = connect_to_sglang(args.server_url)

        # Load dataset
        logger.info(f"Loading GSM8K dataset...")
        dataset = load_gsm8k_dataset(args.num_samples, args.seed)

        # Debug dataset structure
        logger.info(f"Dataset has {len(dataset)} examples")
        logger.debug(f"Dataset features: {dataset.features}")
        if len(dataset) > 0:
            logger.debug(f"Sample batch keys: {list(dataset[0:1].keys())}")
            logger.debug(
                f"Number of questions in sample: {len(dataset[0:1]['question'])}"
            )
            logger.debug(f"First question: {dataset[0:1]['question'][0]}")

        # Take a subset of the dataset based on num_samples
        if args.num_samples < len(dataset):
            dataset = dataset.select(range(args.num_samples))
            logger.info(f"Selected {args.num_samples} examples from the dataset")

        # Process dataset in batches
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

            # Save all responses to a JSON file after each batch
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
# ## Execute the Main Function
#
# Let's run our main function to generate and save the responses.
# With SGLang, this should be significantly faster than the original implementation
# due to parallelization and optimized inference.

# %%
# Execute the main function when running as a script or if explicitly requested
if __name__ == "__main__" or "ipykernel" in sys.modules:
    if "ipykernel" in sys.modules:
        print("Running in notebook mode, processing a few examples...")
        # Use a smaller number of samples for interactive testing
        args.num_samples = min(args.num_samples, 5)

    outputs = main(args)

    # In notebook mode, let's also examine the first saved response
    if "ipykernel" in sys.modules and outputs and len(outputs) > 0:
        print("\nExample of a saved response:")
        print("-" * 50)
        print(f"Question: {outputs[0]['question']}")
        print(f"\nThinking: {outputs[0]['with_thinking']['thinking'][:200]}...")
        print(
            f"\nResponse (with thinking): {outputs[0]['with_thinking']['response'][:200]}..."
        )

# %% [markdown]
# ## Next Steps
#
# Now that we've generated responses with thinking using SGLang's parallel processing,
# we can use this data to extract the reasoning length direction.
#
# Continue to the next notebook: `extract_reasoning_length_direction.py`
