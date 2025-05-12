# %% [markdown]
# # Steering Reasoning Length
#
# This notebook applies the extracted reasoning length direction during model generation
# to control the length of reasoning in the model's responses.
#
# We can steer the model toward longer or shorter reasoning by adjusting the
# steering strength (alpha parameter). This allows us to:
# 1. Control the verbosity of the model's reasoning
# 2. Observe how reasoning length affects accuracy
# 3. Find optimal reasoning length for different tasks
#
# By the end of this notebook, you'll have a visualization showing how steering strength
# affects both reasoning length and accuracy on math problems.

# %% [markdown]
# ## Setup
#
# First, let's import necessary libraries and check versions.

# %%
import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Display installed versions
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
# When running as a script, you can provide these command-line arguments.
# In notebook mode, we'll define default values you can modify.


# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Steer model reasoning length")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--directions_dir",
        type=str,
        default="directions",
        help="Directory with saved direction vectors",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="steering_results",
        help="Directory to save steering results",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--direction_weights",
        nargs="+",
        type=float,
        default=[-0.08, -0.04, 0, 0.04, 0.08],
        help="Direction weights to apply (alpha values)",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["attn", "mlp", "both"],
        default="attn",
        help="Which component to steer (attention, MLP, or both)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# %% [markdown]
# ## Interactive Configuration
#
# If you're running this as a notebook, modify these parameters as needed.


# %%
# Interactive notebook parameters - modify these values as needed
class NotebookArgs:
    def __init__(self):
        self.model = "Qwen/Qwen3-0.6B"  # Model name
        self.directions_dir = "directions"  # Directory with saved directions
        self.output_dir = "steering_results"  # Directory to save results
        self.num_samples = 3  # Use a small number for quick testing
        self.direction_weights = [-0.08, 0.0, 0.08]  # Alpha values to test
        self.component = "attn"  # Component to steer (attn, mlp, or both)
        self.max_new_tokens = 1024  # Maximum new tokens to generate
        self.seed = 42  # Random seed for reproducibility


# Use NotebookArgs when running as notebook, otherwise parse command line arguments
import sys

if "ipykernel" in sys.modules:
    args = NotebookArgs()
    print("Running in notebook mode with these parameters:")
    print(f"- Model: {args.model}")
    print(f"- Component to steer: {args.component}")
    print(f"- Direction weights (alpha): {args.direction_weights}")
    print(f"- Number of samples: {args.num_samples}")
else:
    args = parse_args()
    print("Running in script mode with parsed arguments")

# %% [markdown]
# ## Step 1: Check if Directions Exist
#
# Let's check if the direction vectors from the previous notebook exist.

# %%
# Check if directions file exists
model_short_name = args.model.split("/")[-1]
directions_file = os.path.join(
    args.directions_dir, f"{model_short_name}_reasoning_length_directions.pt"
)

if os.path.exists(directions_file):
    print(f"Found directions file: {directions_file}")
    try:
        # Load directions
        directions = torch.load(directions_file)
        print(f"Loaded directions for {len(directions)} layers")

        # Print some basic stats
        attn_layers = [l for l in directions.keys() if "attn" in l]
        mlp_layers = [l for l in directions.keys() if "mlp" in l]
        print(f"Attention layers: {len(attn_layers)}")
        print(f"MLP layers: {len(mlp_layers)}")
    except Exception as e:
        print(f"Error loading directions: {e}")
        print("Will need to run extract_reasoning_length_direction.py first")
else:
    print(f"Directions file not found: {directions_file}")
    print("Please run extract_reasoning_length_direction.py first")

# %% [markdown]
# ## Step 2: Define Custom Layers for Steering
#
# We'll create wrapper classes for attention and MLP layers to apply steering during generation.


# %%
class SteeringAttentionLayer(torch.nn.Module):
    """Wrapper around an attention layer to apply steering during forward pass."""

    def __init__(self, attn_layer, direction, alpha=0.0):
        super().__init__()
        self.attn_layer = attn_layer
        self.direction = direction  # Direction vector to apply
        self.alpha = alpha  # Steering strength

        # Save original forward method
        self.original_forward = attn_layer.forward

        # Replace forward method
        def new_forward(*args, **kwargs):
            output = self.original_forward(*args, **kwargs)

            # Apply steering direction
            if isinstance(output, tuple):
                # Some models return tuple, modify the first element (output tensor)
                modified_output = output[0] + self.alpha * self.direction
                return (modified_output,) + output[1:]
            else:
                # Direct output tensor
                return output + self.alpha * self.direction

        self.attn_layer.forward = new_forward

    def restore_original(self):
        """Restore the original forward method."""
        self.attn_layer.forward = self.original_forward


# %%
class SteeringMLPLayer(torch.nn.Module):
    """Wrapper around an MLP layer to apply steering during forward pass."""

    def __init__(self, mlp_layer, direction, alpha=0.0):
        super().__init__()
        self.mlp_layer = mlp_layer
        self.direction = direction  # Direction vector to apply
        self.alpha = alpha  # Steering strength

        # Save original forward method
        self.original_forward = mlp_layer.forward

        # Replace forward method
        def new_forward(*args, **kwargs):
            output = self.original_forward(*args, **kwargs)

            # Apply steering direction
            if isinstance(output, tuple):
                # Some models return tuple, modify the first element (output tensor)
                modified_output = output[0] + self.alpha * self.direction
                return (modified_output,) + output[1:]
            else:
                # Direct output tensor
                return output + self.alpha * self.direction

        self.mlp_layer.forward = new_forward

    def restore_original(self):
        """Restore the original forward method."""
        self.mlp_layer.forward = self.original_forward


# %% [markdown]
# ## Step 3: Load the GSM8K Test Data
#
# We'll use a subset of the GSM8K test set to evaluate our steering mechanism.


# %%
def load_test_data(num_samples=10, seed=42):
    """Load and prepare the GSM8K test dataset."""
    dataset = load_dataset("gsm8k", "main")

    # Use the test split and take a subset
    test_data = dataset["test"].shuffle(seed=seed).select(range(num_samples))

    return test_data


# %%
# Load a small sample of test data to preview
test_data = load_test_data(num_samples=2, seed=args.seed)

# Display samples
print(f"Loaded {len(test_data)} test examples")
if len(test_data) > 0:
    print("\nExample test problem:")
    print("-" * 50)
    print(f"Question: {test_data[0]['question']}")
    print(f"Answer: {test_data[0]['answer']}")

# %% [markdown]
# ## Step 4: Define Functions for Applying Steering
#
# Now let's define functions to apply steering layers and generate responses.


# %%
def apply_steering_layers(model, directions, alpha=0.0, component="attn"):
    """Apply steering layers to the model."""
    steering_layers = []

    # Apply steering to selected components
    if component in ["attn", "both"]:
        attn_directions = {k: v for k, v in directions.items() if "attn" in k}
        for layer_name, direction in attn_directions.items():
            layer_idx = int(layer_name.split("_")[-1])
            if layer_idx < len(model.model.layers):
                attn_layer = model.model.layers[layer_idx].self_attn
                steering_layer = SteeringAttentionLayer(attn_layer, direction, alpha)
                steering_layers.append(steering_layer)

    if component in ["mlp", "both"]:
        mlp_directions = {k: v for k, v in directions.items() if "mlp" in k}
        for layer_name, direction in mlp_directions.items():
            layer_idx = int(layer_name.split("_")[-1])
            if layer_idx < len(model.model.layers):
                mlp_layer = model.model.layers[layer_idx].mlp
                steering_layer = SteeringMLPLayer(mlp_layer, direction, alpha)
                steering_layers.append(steering_layer)

    print(f"Applied {len(steering_layers)} steering layers with alpha={alpha}")
    return steering_layers


# %%
def remove_steering_layers(steering_layers):
    """Remove steering layers from the model."""
    for layer in steering_layers:
        layer.restore_original()
    print(f"Removed {len(steering_layers)} steering layers")


# %%
def generate_with_steering(
    model,
    tokenizer,
    question,
    alpha=0.0,
    directions=None,
    component="attn",
    max_new_tokens=1024,
):
    """Generate a response with steering applied."""
    if directions is None:
        raise ValueError("Directions must be provided for steering")

    # Create the prompt
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
        enable_thinking=True,  # Enable thinking mode
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Apply steering
    steering_layers = apply_steering_layers(model, directions, alpha, component)

    # Generate with steering applied
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for deterministic outputs
        )

    # Remove steering layers
    remove_steering_layers(steering_layers)

    # Process the output
    output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()

    # Parse thinking content
    try:
        # Find index of </think> token
        think_end_token = tokenizer.encode("</think>", add_special_tokens=False)[-1]
        think_end_index = (
            output_ids.index(think_end_token) if think_end_token in output_ids else -1
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


# %% [markdown]
# ## Step 5: Load Model and Test Steering
#
# Let's load the model and test our steering mechanism on a simple example.

# %%
# Load model
try:
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )

    # Print model information
    print(f"Model loaded: {args.model}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    # Test steering if directions are available
    if "directions" in locals():
        test_question = "If there are 5 apples and 3 are eaten, how many remain?"

        print("\nTesting steering with a simple example...")
        print("Test question:", test_question)

        # Generate with positive alpha (encouraging longer reasoning)
        print("\nGenerating with α = 0.08 (longer reasoning)...")
        long_response = generate_with_steering(
            model,
            tokenizer,
            test_question,
            alpha=0.08,
            directions=directions,
            component=args.component,
        )

        # Generate with zero alpha (neutral)
        print("\nGenerating with α = 0.0 (neutral)...")
        neutral_response = generate_with_steering(
            model,
            tokenizer,
            test_question,
            alpha=0.0,
            directions=directions,
            component=args.component,
        )

        # Generate with negative alpha (encouraging shorter reasoning)
        print("\nGenerating with α = -0.08 (shorter reasoning)...")
        short_response = generate_with_steering(
            model,
            tokenizer,
            test_question,
            alpha=-0.08,
            directions=directions,
            component=args.component,
        )

        # Print statistics
        print("\nThinking length comparison:")
        print(f"- Long reasoning: {len(long_response['thinking'].split())} words")
        print(f"- Neutral reasoning: {len(neutral_response['thinking'].split())} words")
        print(f"- Short reasoning: {len(short_response['thinking'].split())} words")

        # Print samples of thinking content
        print("\nSample of long reasoning thinking:")
        print(long_response["thinking"][:200] + "...")

        print("\nSample of neutral reasoning thinking:")
        print(neutral_response["thinking"][:200] + "...")

        print("\nSample of short reasoning thinking:")
        print(short_response["thinking"][:200] + "...")
    else:
        print("No directions available for testing")
except Exception as e:
    print(f"Error loading model or testing: {e}")
    print("If you're running in a limited environment, you can skip this test cell.")

# %% [markdown]
# ## Step 6: Evaluation Functions
#
# These functions evaluate the responses and calculate metrics.


# %%
def calculate_metrics(response, alpha):
    """Calculate metrics for a response."""
    thinking = response["thinking"]

    # Calculate thinking length (number of words)
    thinking_length = len(thinking.split()) if thinking else 0

    # Calculate thinking length (number of characters)
    thinking_chars = len(thinking) if thinking else 0

    return {
        "alpha": alpha,
        "thinking_words": thinking_length,
        "thinking_chars": thinking_chars,
    }


# %%
def is_correct(response, answer):
    """Check if the response has the correct answer."""
    # Extract the expected answer (usually a number)
    # This is a simplified check - would need to be more robust for real evaluation
    expected_answer = extract_answer_from_gsm8k(answer)

    # Check if this answer appears in the response
    full_response = response["thinking"] + " " + response["response"]
    return expected_answer in full_response


# %%
def extract_answer_from_gsm8k(answer_text):
    """Extract the numerical answer from a GSM8K answer string."""
    # GSM8K answers typically end with "#### number"
    if "####" in answer_text:
        return answer_text.split("####")[1].strip()
    return answer_text.strip()


# %% [markdown]
# ## Step 7: Visualization Functions
#
# Let's enhance our visualization capabilities to better understand the results.


# %%
def visualize_results(results, output_dir):
    """Visualize the relationship between steering strength, reasoning length, and accuracy."""
    # Calculate average metrics per alpha value
    alpha_values = sorted(set(r["alpha"] for r in results))
    avg_metrics = []

    for alpha in alpha_values:
        alpha_results = [r for r in results if r["alpha"] == alpha]
        avg_thinking_words = np.mean([r["thinking_words"] for r in alpha_results])
        avg_thinking_chars = np.mean([r["thinking_chars"] for r in alpha_results])
        accuracy = np.mean([r["is_correct"] for r in alpha_results])

        avg_metrics.append(
            {
                "alpha": alpha,
                "avg_thinking_words": avg_thinking_words,
                "avg_thinking_chars": avg_thinking_chars,
                "accuracy": accuracy,
            }
        )

    # Create table of results
    print("\nSteering Results Summary:")
    print("-" * 60)
    print(f"{'Alpha':^10} | {'Words':^12} | {'Characters':^12} | {'Accuracy':^10}")
    print("-" * 60)
    for m in avg_metrics:
        print(
            f"{m['alpha']:^10.2f} | {m['avg_thinking_words']:^12.1f} | {m['avg_thinking_chars']:^12.1f} | {m['accuracy']:^10.1%}"
        )

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot thinking length
    ax1.set_xlabel("Steering Strength (α)", fontsize=14)
    ax1.set_ylabel("Average Thinking Length (words)", color="blue", fontsize=14)
    ax1.plot(
        [m["alpha"] for m in avg_metrics],
        [m["avg_thinking_words"] for m in avg_metrics],
        "bo-",
        linewidth=3,
        markersize=8,
        label="Thinking Length",
    )
    ax1.tick_params(axis="y", labelcolor="blue", labelsize=12)

    # Add data point labels
    for m in avg_metrics:
        ax1.annotate(
            f"{m['avg_thinking_words']:.1f}",
            (m["alpha"], m["avg_thinking_words"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=11,
            color="blue",
        )

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="red", fontsize=14)
    ax2.plot(
        [m["alpha"] for m in avg_metrics],
        [m["accuracy"] for m in avg_metrics],
        "ro-",
        linewidth=3,
        markersize=8,
        label="Accuracy",
    )
    ax2.tick_params(axis="y", labelcolor="red", labelsize=12)
    ax2.set_ylim([0, 1.1])  # Set reasonable bounds for accuracy

    # Add data point labels for accuracy
    for m in avg_metrics:
        ax2.annotate(
            f"{m['accuracy']:.1%}",
            (m["alpha"], m["accuracy"]),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=11,
            color="red",
        )

    # Add a title and adjust layout
    plt.title(
        "Effect of Reasoning Length Steering on Thinking Length and Accuracy",
        fontsize=16,
    )
    fig.tight_layout()

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)

    # Add grid for better readability
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "steering_effect.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")

    # Show the plot in the notebook
    plt.show()

    return avg_metrics


# %% [markdown]
# ## Step 8: Main Function
#
# Let's define the main function that will run our experiments.


# %%
def main(args):
    model_short_name = args.model.split("/")[-1]
    directions_file = os.path.join(
        args.directions_dir, f"{model_short_name}_reasoning_length_directions.pt"
    )

    # Check if directions file exists
    if not os.path.exists(directions_file):
        print(f"Error: Directions file {directions_file} not found.")
        print(f"Please run extract_reasoning_length_direction.py first.")
        return None

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the directions
    print(f"Loading directions from {directions_file}...")
    directions = torch.load(directions_file)

    # Load model and tokenizer
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )

    # Load test data
    print(f"Loading test data...")
    test_data = load_test_data(args.num_samples, args.seed)

    # List to store all results
    all_results = []

    # For each steering strength (alpha)
    for alpha in args.direction_weights:
        print(f"Testing with steering strength α = {alpha}...")

        # Process each test example
        for i, example in enumerate(
            tqdm(test_data, desc=f"Processing examples (α = {alpha})")
        ):
            question = example["question"]
            answer = example["answer"]

            # Generate response with steering
            response = generate_with_steering(
                model,
                tokenizer,
                question,
                alpha=alpha,
                directions=directions,
                component=args.component,
                max_new_tokens=args.max_new_tokens,
            )

            # Calculate metrics
            metrics = calculate_metrics(response, alpha)
            metrics["question_id"] = i
            metrics["is_correct"] = is_correct(response, answer)
            metrics["response"] = response

            all_results.append(metrics)

    # Save the results
    results_file = os.path.join(
        args.output_dir, f"{model_short_name}_{args.component}_steering_results.json"
    )
    with open(results_file, "w") as f:
        # Convert responses to JSON-serializable format
        serializable_results = []
        for r in all_results:
            result_copy = r.copy()
            result_copy["response"] = {
                "thinking": r["response"]["thinking"],
                "response": r["response"]["response"],
            }
            serializable_results.append(result_copy)

        json.dump(serializable_results, f, indent=2)

    # Visualize the results
    avg_metrics = visualize_results(all_results, args.output_dir)

    print(f"Done! Results saved to {results_file}")

    return all_results, avg_metrics


# %% [markdown]
# ## Execute the Main Function
#
# Let's run the main function to apply steering and evaluate the results.

# %%
# Execute the main function when running as a script or if explicitly requested
if __name__ == "__main__" or "ipykernel" in sys.modules:
    if "ipykernel" in sys.modules:
        print("Running in notebook mode with a smaller test set...")
        # Use a smaller number of samples and fewer alpha values for interactive testing
        args.num_samples = min(args.num_samples, 3)

    # Run the main function
    results, avg_metrics = main(args)

    # In notebook mode, let's examine one result in detail
    if "ipykernel" in sys.modules and results and len(results) >= 3:
        # Select one example with different alphas to compare
        question_id = results[0]["question_id"]
        alpha_results = [r for r in results if r["question_id"] == question_id]

        print("\nDetailed Example Comparison:")
        print("=" * 80)
        print(f"Question: {test_data[question_id]['question']}")
        print(
            f"Expected answer: {extract_answer_from_gsm8k(test_data[question_id]['answer'])}"
        )
        print("=" * 80)

        for r in alpha_results:
            print(f"\nWith α = {r['alpha']}:")
            print(f"- Thinking length: {r['thinking_words']} words")
            print(f"- Correct: {r['is_correct']}")
            print(f"- Thinking excerpt: {r['response']['thinking'][:150]}...")

# %% [markdown]
# ## Analysis and Conclusion
#
# Based on our experiments, we can observe how steering the reasoning length affects:
#
# 1. The verbosity of the model's reasoning process
# 2. The accuracy on math problems
# 3. The quality of explanations
#
# This demonstrates the power of controlling reasoning length through weight steering,
# and provides insights for optimizing language models for reasoning tasks.
#
# **Takeaways:**
# - Positive alpha values increase reasoning length, while negative values decrease it
# - There may be an optimal reasoning length for accuracy
# - The relationship between reasoning length and accuracy is task-dependent
