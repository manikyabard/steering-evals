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

# %% [markdown]
# ## Setup
#
# First, let's import the necessary libraries and set up the argument parser.

# %%
import os
import json
import torch
import argparse
import concurrent.futures
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import sglang as sgl
from sglang.srt.reasoning_parser import ReasoningParser

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
        default=200,  # Similar to ThinkEdit paper
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
        default=4,
        help="Number of requests to process in parallel",
    )
    parser.add_argument(
        "--reasoning_parser",
        type=str,
        default="qwen3",
        help="Reasoning parser type (qwen3, deepseek-r1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Port to use for SGLang server",
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
        self.reasoning_parser = "qwen3"  # Reasoning parser type
        self.port = 30000  # Port for SGLang server


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
    print(f"- Reasoning parser: {args.reasoning_parser}")
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
    # dataset = load_dataset("gsm8k", "main")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train[:2000]")

    # Use the train split and take a subset
    # train_data = gsm8k.shuffle(seed=seed).select(range(num_samples))

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
# ## Initialize SGLang Engine/Server
#
# Now let's set up the SGLang engine to handle our requests efficiently.


# %%
def initialize_sglang(args):
    """Initialize SGLang engine with the specified model."""
    print(f"Initializing SGLang with model {args.model}...")

    # Launch the server with reasoning parser enabled
    from sglang.utils import launch_server_cmd, wait_for_server

    server_process, port = launch_server_cmd(
        f"python -m sglang.launch_server --model-path {args.model} "
        f"--host 0.0.0.0 --port {args.port} "
        f"--reasoning-parser {args.reasoning_parser}"
    )

    wait_for_server(f"http://localhost:{port}")

    # Create the SGLang client
    client = sgl.Client(api_base=f"http://localhost:{port}")

    print(f"SGLang initialized on port {port}")

    return client, server_process, port


# %% [markdown]
# ## Generate Responses
#
# Now let's define a function to generate responses from our model with thinking enabled
# using SGLang's reasoning parser capabilities.


# %%
def generate_response(client, tokenizer, question):
    """Generate a response from the model with thinking enabled using SGLang."""
    # Include step-by-step reasoning instructions as recommended
    prompt = f"Solve this math problem step by step, and put your final answer within \\boxed{{}}:\n{question}"

    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Print what the prompt looks like (for debugging)
    print(f"Prompt (first 100 chars): {text[:100]}...")

    # Define sampling parameters
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    # Generate with SGLang's API
    response = client.generate(
        prompt=text,
        sampling_params=sampling_params,
    )

    # Get the completed text
    generated_text = response["text"]

    # Use the separate_reasoning endpoint to extract thinking and response
    separate_response = client.separate_reasoning(content=generated_text)

    return {
        "thinking": separate_response.get("reasoning", ""),
        "response": separate_response.get("text", generated_text),
    }


# %% [markdown]
# ## Process Multiple Requests in Parallel
#
# Let's define a function to process batches of examples in parallel.


# %%
def process_batch(client, tokenizer, batch, start_idx):
    """Process a batch of examples in parallel."""
    results = []

    # Create a thread pool to process the batch
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
        future_to_idx = {
            executor.submit(
                generate_response, client, tokenizer, example["question"]
            ): i
            for i, example in enumerate(batch)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(batch),
            desc=f"Batch starting at {start_idx}",
        ):
            idx = future_to_idx[future]
            example = batch[idx]
            try:
                thinking_result = future.result()

                # Save the results
                output = {
                    "id": start_idx + idx,
                    "question": example["question"],
                    "answer": example["answer"],
                    "with_thinking": thinking_result,
                }
                results.append(output)

            except Exception as e:
                print(f"Error processing example {start_idx + idx}: {e}")
                # Add a placeholder for failed requests
                output = {
                    "id": start_idx + idx,
                    "question": example["question"],
                    "answer": example["answer"],
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
    print("Setting up SGLang for testing...")
    # Initialize tokenizer directly
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Initialize SGLang client and server
    client, server_process, port = initialize_sglang(args)

    test_question = "If there are 5 apples and 3 are eaten, how many remain?"

    # Generate with thinking enabled
    print("\nGenerating response with thinking enabled...")
    thinking_result = generate_response(client, tokenizer, test_question)

    print("\nThinking content:")
    print("-" * 50)
    print(thinking_result["thinking"])

    print("\nResponse content:")
    print("-" * 50)
    print(thinking_result["response"])

    # Clean up
    from sglang.utils import terminate_process

    terminate_process(server_process)

except Exception as e:
    print(f"Error testing the model: {e}")
    print(
        "If you're running in a limited environment, comment out this test cell and proceed to the main function."
    )

# %% [markdown]
# ## Main Function
#
# Now let's define the main function to process the whole dataset and save the responses.


# %%
def main(args):
    """Main function to process GSM8K examples and save responses in parallel."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    model_short_name = args.model.split("/")[-1]

    # Load tokenizer for creating prompts
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Initialize SGLang
    client, server_process, port = initialize_sglang(args)

    try:
        # Load dataset
        print(f"Loading GSM8K dataset...")
        dataset = load_gsm8k_dataset(args.num_samples, args.seed)

        # Process dataset in batches
        outputs = []
        for i in range(0, len(dataset), args.batch_size):
            batch = dataset[i : i + args.batch_size]
            batch_results = process_batch(client, tokenizer, batch, i)
            outputs.extend(batch_results)

            # Save all responses to a JSON file after each batch
            output_path = os.path.join(
                args.output_dir, f"{model_short_name}_gsm8k_responses.json"
            )
            with open(output_path, "w") as f:
                json.dump(outputs, f, indent=2)

            print(f"Saved {len(outputs)} responses so far to {output_path}")

        print(f"All responses saved to {output_path}")
        return outputs

    finally:
        # Clean up SGLang server
        from sglang.utils import terminate_process

        terminate_process(server_process)
        print("SGLang server terminated")


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
