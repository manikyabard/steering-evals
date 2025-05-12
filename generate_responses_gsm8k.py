# %% [markdown]
# # Generating Responses from GSM8K
#
# This notebook generates and saves model responses from the GSM8K dataset.
# We'll generate both thinking and non-thinking responses to create our paired dataset
# for identifying reasoning length direction.
#
# The notebook follows these steps:
# 1. Load the GSM8K dataset
# 2. Generate responses with thinking enabled
# 3. Generate responses with thinking disabled
# 4. Save the paired responses for later analysis

# %% [markdown]
# ## Setup
#
# First, let's import the necessary libraries and set up the argument parser.

# %%
import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Display installed versions for reproducibility
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
try:
    import transformers

    print(f"Transformers version: {transformers.__version__}")
except:
    print("Transformers not installed")

# %% [markdown]
# ## Command Line Arguments
#
# When running this as a script, you can provide command-line arguments.
# In notebook mode, we'll define default values that you can modify in the next cell.


# %%
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate responses from GSM8K dataset"
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
        default=100,
        help="Number of samples to process from GSM8K",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
        self.max_new_tokens = 1024  # Maximum new tokens to generate
        self.seed = 42  # Random seed for reproducibility


# Use NotebookArgs when running as notebook, otherwise parse command line arguments
import sys

if "ipykernel" in sys.modules:
    args = NotebookArgs()
    print("Running in notebook mode with these parameters:")
    print(f"- Model: {args.model}")
    print(f"- Number of samples: {args.num_samples}")
    print(f"- Output directory: {args.output_dir}")
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
    dataset = load_dataset("gsm8k", "main")

    # Use the train split and take a subset
    train_data = dataset["train"].shuffle(seed=seed).select(range(num_samples))

    return train_data


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
# ## Generate Responses
#
# Now let's define a function to generate responses from our model with and without thinking.


# %%
def generate_response(model, tokenizer, question, enable_thinking=True):
    """Generate a response from the model with or without thinking."""
    messages = [
        {
            "role": "user",
            "content": f"Solve this math problem step by step:\n{question}",
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,  # Toggle thinking mode
    )

    # Let's see what the prompt looks like (for debugging)
    if enable_thinking:
        print(f"Prompt with thinking enabled (first 100 chars): {text[:100]}...")

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,  # Use greedy decoding for deterministic outputs
        )

    output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()

    # Parse thinking content if applicable
    if enable_thinking:
        try:
            # Find index of </think> token
            think_end_token = tokenizer.encode("</think>", add_special_tokens=False)[-1]
            think_end_index = (
                output_ids.index(think_end_token)
                if think_end_token in output_ids
                else -1
            )

            if think_end_index != -1:
                thinking_content = tokenizer.decode(
                    output_ids[:think_end_index], skip_special_tokens=True
                ).strip()
                content = tokenizer.decode(
                    output_ids[think_end_index + 1 :], skip_special_tokens=True
                ).strip()
                return {"thinking": thinking_content, "response": content}
        except ValueError:
            pass

        # If no thinking token found or error occurred, return everything as response
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return {"thinking": "", "response": content}
    else:
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return {"thinking": "", "response": content}


# %% [markdown]
# ## Test the Model on a Single Example
#
# Let's first test the model on a single example to make sure everything is working properly.

# %%
# Load the model and tokenizer
try:
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )

    # Test with a simple problem
    test_question = "If there are 5 apples and 3 are eaten, how many remain?"

    # With thinking enabled
    print("\nGenerating response with thinking enabled...")
    thinking_result = generate_response(
        model, tokenizer, test_question, enable_thinking=True
    )

    print("\nThinking content:")
    print("-" * 50)
    print(thinking_result["thinking"])

    print("\nResponse content:")
    print("-" * 50)
    print(thinking_result["response"])

    # With thinking disabled
    print("\nGenerating response with thinking disabled...")
    non_thinking_result = generate_response(
        model, tokenizer, test_question, enable_thinking=False
    )

    print("\nResponse content (no thinking):")
    print("-" * 50)
    print(non_thinking_result["response"])

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
    """Main function to process GSM8K examples and save paired responses."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    model_short_name = args.model.split("/")[-1]

    # Load model and tokenizer
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )

    # Load dataset
    print(f"Loading GSM8K dataset...")
    dataset = load_gsm8k_dataset(args.num_samples, args.seed)

    # Generate and save responses
    outputs = []
    for i, example in enumerate(tqdm(dataset, desc="Generating responses")):
        question = example["question"]

        # Generate with thinking enabled
        thinking_result = generate_response(
            model, tokenizer, question, enable_thinking=True
        )

        # Generate with thinking disabled
        # non_thinking_result = generate_response(
        #     model, tokenizer, question, enable_thinking=False
        # )

        # Save the results
        output = {
            "id": i,
            "question": question,
            "answer": example["answer"],
            "with_thinking": thinking_result,
            # "without_thinking": non_thinking_result,
        }
        outputs.append(output)

        # Save all responses to a JSON file
        output_path = os.path.join(
            args.output_dir, f"{model_short_name}_gsm8k_responses.json"
        )
        with open(output_path, "w") as f:
            json.dump(outputs, f, indent=2)

    print(f"Responses saved to {output_path}")
    return outputs


# %% [markdown]
# ## Execute the Main Function
#
# Let's run our main function to generate and save the paired responses.
# This might take a while depending on the number of samples and the model size.

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
        print("\nExample of a saved response pair:")
        print("-" * 50)
        print(f"Question: {outputs[0]['question']}")
        print(f"\nThinking: {outputs[0]['with_thinking']['thinking'][:200]}...")
        print(
            f"\nResponse (with thinking): {outputs[0]['with_thinking']['response'][:200]}..."
        )
        # print(
        #     f"\nResponse (without thinking): {outputs[0]['without_thinking']['response'][:200]}..."
        # )

# %% [markdown]
# ## Next Steps
#
# Now that we've generated paired responses with and without thinking, we can use this data to extract the reasoning length direction.
#
# Continue to the next notebook: `extract_reasoning_length_direction.py`
