# %% [markdown]
# # Extracting Reasoning Length Direction
#
# This notebook identifies and extracts the reasoning length direction from self-attention and MLP layers
# based on responses generated from the GSM8K dataset.
#
# We use the Contrastive Activation Addition (CAA) technique to identify direction vectors
# that control reasoning length by comparing activations from long thinking vs short thinking examples.
# The extracted directions can later be used to steer the model's reasoning.
#
# Workflow:
# 1. Load responses with thinking from the previous notebook
# 2. Extract activations from model layers using hooks
# 3. Compute direction vectors by contrasting long vs. short thinking examples
# 4. Save and visualize the extracted directions

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
from collections import defaultdict
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
# When running this as a script, you can provide command-line arguments.
# In notebook mode, we'll define default values that you can modify directly.


# %%
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract reasoning length direction from model"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--responses_dir",
        type=str,
        default="responses",
        help="Directory with saved responses",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="directions",
        help="Directory to save extracted directions",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for processing examples"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to use for direction extraction",
    )
    return parser.parse_args()


# %% [markdown]
# ## Interactive Configuration
#
# If you're running this as a notebook, you can modify these parameters directly.


# %%
# Interactive notebook parameters - modify these values as needed
class NotebookArgs:
    def __init__(self):
        self.model = "Qwen/Qwen3-0.6B"  # Model name
        self.responses_dir = "responses"  # Directory with saved responses
        self.output_dir = "directions"  # Directory to save extracted directions
        self.batch_size = 4  # Batch size for processing
        self.num_samples = 5  # Use a small number for testing


# Use NotebookArgs when running as notebook, otherwise parse command line arguments
import sys

if "ipykernel" in sys.modules:
    args = NotebookArgs()
    print("Running in notebook mode with these parameters:")
    print(f"- Model: {args.model}")
    print(f"- Responses directory: {args.responses_dir}")
    print(f"- Number of samples: {args.num_samples}")
else:
    args = parse_args()
    print("Running in script mode with parsed arguments")

# %% [markdown]
# ## Step 1: Check if Responses Exist
#
# Let's check if the responses from the previous notebook exist.

# %%
# Check if responses file exists
model_short_name = args.model.split("/")[-1]
responses_file = os.path.join(
    args.responses_dir, f"{model_short_name}_gsm8k_responses.json"
)

if os.path.exists(responses_file):
    print(f"Found responses file: {responses_file}")
    # Load a small sample just to preview
    with open(responses_file, "r") as f:
        responses = json.load(f)
    print(f"Loaded {len(responses)} responses")

    # Preview first example
    if len(responses) > 0:
        print("\nExample response:")
        print("-" * 50)
        print(f"Question: {responses[0]['question']}")
        print(
            f"Thinking (first 100 chars): {responses[0]['with_thinking']['thinking'][:100]}..."
        )
        print(
            f"Response (first 100 chars): {responses[0]['with_thinking']['response'][:100]}..."
        )
else:
    print(f"Responses file not found: {responses_file}")
    print(
        "Please run generate_responses_gsm8k.py first or adjust the responses_dir parameter."
    )

# %% [markdown]
# ## Activation Extractor
#
# Now let's define a class to extract activations from the model using hooks.
# We'll use these activations to compute the direction vectors.


# %%
class ActivationExtractor:
    """Extract and store activations from specific layers of the model."""

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

        # Determine model architecture and set up accordingly
        if "Qwen" in model.__class__.__name__:
            self.setup_qwen_hooks()
        else:
            # Default setup for other models
            self.setup_default_hooks()

    def setup_qwen_hooks(self):
        """Set up hooks for Qwen models."""
        # Get all transformer blocks
        for i, block in enumerate(self.model.model.layers):
            # Add hooks for self-attention layers
            attn_hook = block.self_attn.register_forward_hook(
                lambda module, input, output, layer_idx=i: self._save_attention_output(
                    output, layer_idx
                )
            )

            # Add hooks for MLP layers
            mlp_hook = block.mlp.register_forward_hook(
                lambda module, input, output, layer_idx=i: self._save_mlp_output(
                    output, layer_idx
                )
            )

            self.hooks.extend([attn_hook, mlp_hook])

    def setup_default_hooks(self):
        """Set up hooks for other model architectures."""
        # Generic approach for transformer models
        for i, block in enumerate(self.model.model.layers):
            # Try common attention layer names
            if hasattr(block, "self_attn"):
                attn_hook = block.self_attn.register_forward_hook(
                    lambda module, input, output, layer_idx=i: self._save_attention_output(
                        output, layer_idx
                    )
                )
                self.hooks.append(attn_hook)

            # Try common MLP layer names
            for mlp_name in ["mlp", "feed_forward", "ffn"]:
                if hasattr(block, mlp_name):
                    mlp_layer = getattr(block, mlp_name)
                    mlp_hook = mlp_layer.register_forward_hook(
                        lambda module, input, output, layer_idx=i: self._save_mlp_output(
                            output, layer_idx
                        )
                    )
                    self.hooks.append(mlp_hook)
                    break

    def _save_attention_output(self, output, layer_idx):
        """Save self-attention output activations."""
        # For some models, output is a tuple; take the first element
        if isinstance(output, tuple):
            output = output[0]

        self.activations[f"attn_layer_{layer_idx}"] = output.detach()

    def _save_mlp_output(self, output, layer_idx):
        """Save MLP output activations."""
        # For some models, output is a tuple; take the first element
        if isinstance(output, tuple):
            output = output[0]

        self.activations[f"mlp_layer_{layer_idx}"] = output.detach()

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_active_layers(self):
        """Return the list of layer names that have hooks attached."""
        layer_types = []
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            num_layers = len(self.model.model.layers)
            for i in range(num_layers):
                layer_types.append(f"attn_layer_{i}")
                layer_types.append(f"mlp_layer_{i}")
        return layer_types


# %% [markdown]
# ## Step 2: Load Model and Set Up Activation Extractor
#
# Let's load the model and create an activation extractor to capture layer outputs.

# %%
# Load model and create activation extractor
try:
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )

    # Create activation extractor
    activation_extractor = ActivationExtractor(model)

    # Print model information
    print(f"Model loaded: {args.model}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )
    print(
        f"Number of layers with hooks: {len(activation_extractor.get_active_layers()) // 2}"
    )

except Exception as e:
    print(f"Error loading model: {e}")
    print("If you're running in a limited environment, you can skip this test cell.")

# %% [markdown]
# ## Functions to Extract Reasoning Length Direction
#
# Now we'll define the functions to extract activations and compute the reasoning length direction.
# We'll compare examples with long thinking vs short thinking to derive the direction vector.


# %%
def get_activations_for_text(model, tokenizer, activation_extractor, text):
    """Get model activations for the given text."""
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Clear previous activations
    activation_extractor.clear_activations()

    # Forward pass through the model to capture activations
    with torch.no_grad():
        model(**inputs)

    # Return a copy of the activations
    return {k: v.clone() for k, v in activation_extractor.activations.items()}


# %%
def compute_reasoning_length_direction(
    model, tokenizer, response_data, activation_extractor, num_samples=50
):
    """Compute the reasoning length direction by comparing long thinking vs. short thinking examples."""
    # Filter examples based on thinking length
    valid_responses = [
        ex
        for ex in response_data
        if "thinking" in ex["with_thinking"] and ex["with_thinking"]["thinking"]
    ]

    # Calculate thinking length for each example
    for ex in valid_responses:
        ex["thinking_length"] = len(ex["with_thinking"]["thinking"])

    # Sort by thinking length
    valid_responses.sort(key=lambda x: x["thinking_length"])

    # Select short thinking examples (bottom 20%)
    short_thinking_count = max(5, int(len(valid_responses) * 0.2))
    short_thinking_examples = valid_responses[:short_thinking_count]

    # Select long thinking examples (top 20%)
    long_thinking_count = max(5, int(len(valid_responses) * 0.2))
    long_thinking_examples = valid_responses[-long_thinking_count:]

    print(f"Using {len(long_thinking_examples)} long thinking examples")
    print(f"Using {len(short_thinking_examples)} short thinking examples")

    # Limit the number of samples to process
    long_examples_to_process = min(len(long_thinking_examples), num_samples // 2)
    short_examples_to_process = min(len(short_thinking_examples), num_samples // 2)

    # Storage for activations
    long_thinking_activations = defaultdict(list)
    short_thinking_activations = defaultdict(list)

    # Process long thinking examples
    for idx in tqdm(
        range(long_examples_to_process), desc="Processing long thinking examples"
    ):
        item = long_thinking_examples[idx]
        question = item["question"]
        thinking = item["with_thinking"]["thinking"]

        # Create prompt for thinking content
        prompt = f"Solve this math problem step by step, and put your final answer within \\boxed{{}}:\n{question}\n{thinking}"

        # Get activations
        activations = get_activations_for_text(
            model, tokenizer, activation_extractor, prompt
        )

        # Store activations for each layer
        for layer_name, layer_activation in activations.items():
            long_thinking_activations[layer_name].append(layer_activation)

    # Process short thinking examples
    for idx in tqdm(
        range(short_examples_to_process), desc="Processing short thinking examples"
    ):
        item = short_thinking_examples[idx]
        question = item["question"]
        thinking = item["with_thinking"]["thinking"]

        # Create prompt for thinking content
        prompt = f"Solve this math problem step by step, and put your final answer within \\boxed{{}}:\n{question}\n{thinking}"

        # Get activations
        activations = get_activations_for_text(
            model, tokenizer, activation_extractor, prompt
        )

        # Store activations for each layer
        for layer_name, layer_activation in activations.items():
            short_thinking_activations[layer_name].append(layer_activation)

    # Calculate mean activations for long and short thinking examples
    long_mean_activations = {}
    short_mean_activations = {}

    for layer_name in long_thinking_activations:
        if layer_name in short_thinking_activations:
            # Stack and mean for long thinking examples
            if long_thinking_activations[layer_name]:
                stacked_long = torch.stack(long_thinking_activations[layer_name])
                long_mean_activations[layer_name] = torch.mean(stacked_long, dim=0)

            # Stack and mean for short thinking examples
            if short_thinking_activations[layer_name]:
                stacked_short = torch.stack(short_thinking_activations[layer_name])
                short_mean_activations[layer_name] = torch.mean(stacked_short, dim=0)

    # Compute direction as long - short
    directions = {}
    for layer_name in long_mean_activations:
        if layer_name in short_mean_activations:
            # Calculate the direction vector (long - short)
            diff = (
                long_mean_activations[layer_name] - short_mean_activations[layer_name]
            )

            # Normalize the direction vector
            norm = torch.norm(diff)
            if norm > 0:
                directions[layer_name] = diff / norm
            else:
                directions[layer_name] = diff

    print(f"Extracted directions for {len(directions)} layers")
    return directions


# %% [markdown]
# ## Test Activation Extraction
#
# Let's test the activation extraction on a simple example.

# %%
# Test activation extraction on a simple example
try:
    test_prompt = "Solve this math problem step by step, and put your final answer within \\boxed{}:\nIf there are 5 apples and 3 are eaten, how many remain?"

    if (
        "model" in locals()
        and "tokenizer" in locals()
        and "activation_extractor" in locals()
    ):
        print("Testing activation extraction...")

        # Get activations
        activations = get_activations_for_text(
            model, tokenizer, activation_extractor, test_prompt
        )

        # Print activation statistics
        print(f"Number of layers with activations: {len(activations)}")

        # Print shape and statistics of one activation
        if activations:
            layer_name = list(activations.keys())[0]
            layer_activation = activations[layer_name]
            print(f"Layer: {layer_name}")
            print(f"Shape: {layer_activation.shape}")
            print(f"Mean: {layer_activation.mean().item():.4f}")
            print(f"Std: {layer_activation.std().item():.4f}")
    else:
        print("Model, tokenizer, or activation_extractor not available. Skipping test.")
except Exception as e:
    print(f"Error testing activation extraction: {e}")

# %% [markdown]
# ## Visualize the Extracted Directions
#
# This function visualizes the magnitude of direction vectors across different layers.


# %%
def visualize_directions(directions, output_dir, model_name):
    """Visualize the magnitude of direction vectors across layers."""
    plt.figure(figsize=(12, 8))

    # Separate attention and MLP layers
    attn_layers = sorted([l for l in directions.keys() if "attn" in l])
    mlp_layers = sorted([l for l in directions.keys() if "mlp" in l])

    # Plot attention layer magnitudes
    if attn_layers:
        attn_magnitudes = [torch.norm(directions[l]).item() for l in attn_layers]
        layer_indices = [int(l.split("_")[-1]) for l in attn_layers]
        plt.plot(layer_indices, attn_magnitudes, "b-", label="Attention Layers")
        plt.scatter(layer_indices, attn_magnitudes, color="blue")

    # Plot MLP layer magnitudes
    if mlp_layers:
        mlp_magnitudes = [torch.norm(directions[l]).item() for l in mlp_layers]
        layer_indices = [int(l.split("_")[-1]) for l in mlp_layers]
        plt.plot(layer_indices, mlp_magnitudes, "r-", label="MLP Layers")
        plt.scatter(layer_indices, mlp_magnitudes, color="red")

    plt.xlabel("Layer Index")
    plt.ylabel("Direction Vector Magnitude")
    plt.title(f"Reasoning Length Direction Magnitudes - {model_name}")
    plt.grid(True)
    plt.legend()

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    model_short_name = model_name.split("/")[-1]
    output_path = os.path.join(
        output_dir, f"{model_short_name}_direction_magnitudes.png"
    )
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.show()  # Display the plot in the notebook


# %% [markdown]
# ## Create a Small Test for Direction Extraction
#
# Let's test the direction extraction with a small sample to make sure everything works.


# %%
# Mock responses for testing (if needed)
def create_mock_responses(num_samples=2):
    """Create mock responses for testing if real ones aren't available."""
    mock_responses = []
    for i in range(num_samples):
        mock_responses.append(
            {
                "id": i,
                "question": f"Test question {i}",
                "answer": f"Test answer {i}",
                "with_thinking": {
                    "thinking": f"This is thinking content for test {i}. It's a bit longer to simulate thinking.",
                    "response": f"The answer is {i}",
                },
            }
        )
    return mock_responses


# Test the direction extraction with a very small sample
try:
    if (
        "model" in locals()
        and "tokenizer" in locals()
        and "activation_extractor" in locals()
    ):
        print("Testing direction extraction with small sample...")

        # Use real responses if available, otherwise use mock responses
        if "responses" in locals() and len(responses) > 0:
            test_responses = responses[:2]  # Just use the first 2 responses
            print("Using real responses for testing")
        else:
            test_responses = create_mock_responses(2)
            print("Using mock responses for testing")

        # Extract directions with very small sample just for testing
        test_directions = compute_reasoning_length_direction(
            model, tokenizer, test_responses, activation_extractor, num_samples=2
        )

        # Print direction statistics
        print(f"Extracted directions for {len(test_directions)} layers")

        # Visualize the test directions
        if test_directions:
            visualize_directions(test_directions, ".", args.model)
    else:
        print("Model, tokenizer, or activation_extractor not available. Skipping test.")
except Exception as e:
    print(f"Error testing direction extraction: {e}")

# %% [markdown]
# ## Main Function
#
# Now let's define the main function to extract directions from all samples.


# %%
def main(args):
    model_short_name = args.model.split("/")[-1]
    responses_file = os.path.join(
        args.responses_dir, f"{model_short_name}_gsm8k_responses.json"
    )

    # Check if responses file exists
    if not os.path.exists(responses_file):
        print(f"Error: Responses file {responses_file} not found.")
        print(f"Please run generate_responses_gsm8k.py first.")
        return None

    # Load responses
    print(f"Loading responses from {responses_file}...")
    with open(responses_file, "r") as f:
        responses = json.load(f)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )

    # Create activation extractor
    activation_extractor = ActivationExtractor(model)

    # Extract reasoning length directions
    print("Extracting reasoning length directions...")
    directions = compute_reasoning_length_direction(
        model, tokenizer, responses, activation_extractor, args.num_samples
    )

    # Remove hooks to free up resources
    activation_extractor.remove_hooks()

    # Save directions
    print("Saving directions...")
    output_file = os.path.join(
        args.output_dir, f"{model_short_name}_reasoning_length_directions.pt"
    )
    torch.save(directions, output_file)

    # Visualize directions
    print("Visualizing directions...")
    visualize_directions(directions, args.output_dir, args.model)

    print(f"Done! Directions saved to {output_file}")

    return directions


# %% [markdown]
# ## Execute the Main Function
#
# Let's run the main function to extract reasoning length directions.

# %%
# Execute the main function when running as a script or if explicitly requested
if __name__ == "__main__" or "ipykernel" in sys.modules:
    if "ipykernel" in sys.modules:
        print("Running in notebook mode, processing a limited number of samples...")
        # Use a smaller number of samples for interactive testing
        args.num_samples = min(args.num_samples, 10)

    # Run the main function
    directions = main(args)

    # In notebook mode, let's examine the directions
    if "ipykernel" in sys.modules and directions:
        print("\nExtracted directions:")
        print("-" * 50)

        # Get layer names and their magnitude
        layer_norms = {
            layer: torch.norm(vec).item() for layer, vec in directions.items()
        }

        # Find the layer with the highest magnitude
        max_layer = max(layer_norms.items(), key=lambda x: x[1])
        print(
            f"Layer with highest magnitude: {max_layer[0]} (norm: {max_layer[1]:.4f})"
        )

        # Calculate average magnitude for attention and MLP layers
        attn_layers = [l for l in directions.keys() if "attn" in l]
        mlp_layers = [l for l in directions.keys() if "mlp" in l]

        if attn_layers:
            attn_avg = np.mean([layer_norms[l] for l in attn_layers])
            print(f"Average attention layer magnitude: {attn_avg:.4f}")

        if mlp_layers:
            mlp_avg = np.mean([layer_norms[l] for l in mlp_layers])
            print(f"Average MLP layer magnitude: {mlp_avg:.4f}")

# %% [markdown]
# ## Next Steps
#
# Now that we've extracted the reasoning length direction vectors, we can use them to steer the model's reasoning.
#
# Continue to the next notebook: `steer_reasoning_length.py`
