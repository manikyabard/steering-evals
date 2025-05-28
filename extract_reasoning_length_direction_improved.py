#!/usr/bin/env python3
"""
Improved Reasoning Length Direction Extraction

This script extracts reasoning length directions following the ThinkEdit approach:
1. Uses fixed sample sizes (100 examples each) instead of percentages
2. Computes thinking lengths using word counts
3. Better sample selection and direction computation
4. More thorough debugging and validation

Based on: https://github.com/Trustworthy-ML-Lab/ThinkEdit/blob/main/extract_thinking_length_direction_gsm8k_attn.py
"""

# %% [markdown]
# # Extract Reasoning Length Direction Vectors
#
# This notebook extracts direction vectors that control reasoning length in language models.
# We'll walk through the complete process step by step, from loading data to extracting
# direction vectors that can be used to steer model reasoning length.
#
# ## Overview
# The process involves:
# 1. Loading model responses with varying reasoning lengths
# 2. Computing thinking lengths for each response
# 3. Selecting examples with short vs long reasoning
# 4. Extracting activations from the model for these examples
# 5. Computing direction vectors by contrasting short vs long examples

# %% Setup and imports
import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from logging_setup import setup_logging, get_logger

# %% [markdown]
# ## Configuration and Arguments
#
# First, let's set up the configuration parameters. These can be modified for different
# experiments or when running in interactive mode.


# %% Command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract reasoning length direction from model (improved version)"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--responses_file",
        type=str,
        default="responses/Qwen3-0.6B_gsm8k_responses.json",
        help="Path to the responses JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="directions",
        help="Directory to save extracted directions",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="attn",
        choices=["attn", "mlp", "both"],
        help="Which component to extract directions for",
    )
    parser.add_argument(
        "--n_short",
        type=int,
        default=None,
        help="Number of short thinking examples to use (default: None = use all available)",
    )
    parser.add_argument(
        "--n_long",
        type=int,
        default=None,
        help="Number of long thinking examples to use (default: None = use all available)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
        help="Device to use for computation",
    )
    parser.add_argument(
        "--recompute_lengths",
        action="store_true",
        help="Recompute thinking lengths (useful if not already computed)",
    )
    parser.add_argument(
        "--use_percentiles",
        action="store_true",
        help="Use percentile-based selection instead of fixed thresholds",
    )
    parser.add_argument(
        "--short_percentile",
        type=float,
        default=20.0,
        help="Percentile threshold for short thinking examples (bottom X%%, default: 20.0)",
    )
    parser.add_argument(
        "--long_percentile",
        type=float,
        default=20.0,
        help="Percentile threshold for long thinking examples (top X%%, default: 20.0)",
    )
    parser.add_argument(
        "--short_threshold",
        type=int,
        default=100,
        help="Fixed threshold for short thinking examples (< X tokens, default: 100)",
    )
    parser.add_argument(
        "--long_threshold",
        type=int,
        default=1000,
        help="Fixed threshold for long thinking examples (> X tokens, default: 1000)",
    )
    return parser.parse_args()


# %% Interactive configuration (for notebook use)
# Uncomment and modify this cell when running interactively in a notebook
"""
# Test configuration for interactive development
test_args = type('Args', (), {
    'model': "Qwen/Qwen3-0.6B",
    'responses_file': "responses/Qwen3-0.6B_gsm8k_responses.json",
    'output_dir': "directions",
    'component': "attn",
    'n_short': None,  # Use all available
    'n_long': None,   # Use all available  
    'device': "cuda:0" if torch.cuda.is_available() else "cpu",
    'recompute_lengths': False,
    'use_percentiles': False,
    'short_percentile': 20.0,
    'long_percentile': 20.0,
    'short_threshold': 100,
    'long_threshold': 1000
})()

# Use test_args instead of parse_args() when running interactively
# args = test_args
"""

# %% [markdown]
# ## Step 1: Data Loading and Analysis
#
# Let's start by loading the response data and examining its structure.
# This data should contain model responses with thinking processes of varying lengths.


# %% Data loading and initial analysis
def load_and_analyze_responses(responses_file):
    """Load responses and provide basic analysis."""
    print(f"Loading responses from: {responses_file}")

    if not os.path.exists(responses_file):
        raise FileNotFoundError(f"Responses file not found: {responses_file}")

    with open(responses_file, "r") as f:
        responses = json.load(f)

    print(f"✓ Loaded {len(responses)} responses")

    # Analyze response structure
    if len(responses) > 0:
        sample = responses[0]
        print(f"\nSample response keys: {list(sample.keys())}")

        if "with_thinking" in sample:
            thinking_keys = list(sample["with_thinking"].keys())
            print(f"Thinking keys: {thinking_keys}")

            thinking_text = sample["with_thinking"].get("thinking", "")
            print(f"Sample thinking length: {len(thinking_text.split())} words")
            print(f"Sample thinking preview: {thinking_text[:200]}...")

    return responses


# %% Test data loading (uncomment to test interactively)
"""
# Test the data loading function
try:
    responses = load_and_analyze_responses("responses/Qwen3-0.6B_gsm8k_responses.json")
    print(f"Successfully loaded {len(responses)} responses for analysis")
except Exception as e:
    print(f"Data loading failed: {e}")
    print("Make sure to run generate_responses_gsm8k.py first to create the responses file")
"""

# %% [markdown]
# ## Step 2: Computing Thinking Lengths
#
# Next, we need to compute the length of thinking for each response.
# We'll use token counts to be consistent with the model's internal representation.


# %% Thinking length computation functions
def count_thinking_tokens(thinking_text, tokenizer):
    """Count tokens in thinking text, matching ThinkEdit approach exactly."""
    if not thinking_text or thinking_text.strip() == "":
        return 0

    tokens = tokenizer.encode(thinking_text, add_special_tokens=False)
    return len(tokens)


def compute_thinking_lengths(responses, tokenizer):
    """Compute thinking lengths for all responses."""
    logger = get_logger()
    logger.info("Computing thinking lengths...")

    for item in tqdm(responses, desc="Computing thinking lengths"):
        if "thinking" in item and item["thinking"]:
            thinking_text = item["thinking"]
            item["thinking_length"] = count_thinking_tokens(thinking_text, tokenizer)
        elif "with_thinking" in item and "thinking" in item["with_thinking"]:
            thinking_text = item["with_thinking"]["thinking"]
            item["thinking_length"] = count_thinking_tokens(thinking_text, tokenizer)
        else:
            item["thinking_length"] = -1

    return responses


# %% Test thinking length computation (uncomment to test interactively)
"""
def test_length_computation():
    \"\"\"Test thinking length computation on a small sample.\"\"\"
    try:
        print("Testing length computation...")
        
        # Load a small sample
        with open("responses/Qwen3-0.6B_gsm8k_responses.json", "r") as f:
            responses = json.load(f)
            
        # Take first 5 examples for testing
        sample_responses = responses[:5]
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        sample_responses = compute_thinking_lengths(sample_responses, tokenizer)
        
        print("✓ Computed thinking lengths:")
        for i, resp in enumerate(sample_responses):
            length = resp.get("thinking_length", -1)
            thinking_text = ""
            if "with_thinking" in resp and "thinking" in resp["with_thinking"]:
                thinking_text = resp["with_thinking"]["thinking"][:100] + "..."
            print(f"  Example {i}: {length} tokens - {thinking_text}")
            
    except Exception as e:
        print(f"Length computation test failed: {e}")

# Uncomment to test:
# test_length_computation()
"""

# %% [markdown]
# ## Step 3: Example Selection by Length
#
# Now we'll select examples with short vs long thinking processes.
# We can use either fixed thresholds or percentile-based selection.


# %% Example selection functions
def select_examples_by_length(
    responses,
    n_short=None,
    n_long=None,
    use_percentiles=False,
    short_percentile=20.0,
    long_percentile=20.0,
    short_threshold=100,
    long_threshold=1000,
):
    """Select examples based on thinking length, using either fixed thresholds or percentiles."""

    valid_responses = [
        item
        for item in responses
        if "thinking_length" in item and item["thinking_length"] != -1
    ]

    print(f"Found {len(valid_responses)} responses with valid thinking")

    if len(valid_responses) == 0:
        raise ValueError("No valid responses found!")

    thinking_lengths = [item["thinking_length"] for item in valid_responses]
    thinking_lengths_array = np.array(thinking_lengths)

    print(f"Thinking length statistics:")
    print(f"  Min: {min(thinking_lengths)} tokens")
    print(f"  Max: {max(thinking_lengths)} tokens")
    print(f"  Mean: {np.mean(thinking_lengths):.1f} tokens")
    print(f"  Median: {np.median(thinking_lengths):.1f} tokens")
    print(f"  Std: {np.std(thinking_lengths):.1f} tokens")

    if use_percentiles:
        print(f"\nUsing percentile-based selection:")
        print(f"  Short: bottom {short_percentile}%")
        print(f"  Long: top {long_percentile}%")

        short_threshold_percentile = np.percentile(
            thinking_lengths_array, short_percentile
        )
        long_threshold_percentile = np.percentile(
            thinking_lengths_array, 100 - long_percentile
        )

        print(f"  Short threshold: ≤ {short_threshold_percentile:.1f} tokens")
        print(f"  Long threshold: ≥ {long_threshold_percentile:.1f} tokens")

        short_thinking_examples = [
            ex
            for ex in valid_responses
            if ex["thinking_length"] <= short_threshold_percentile
        ]
        long_thinking_examples = [
            ex
            for ex in valid_responses
            if ex["thinking_length"] >= long_threshold_percentile
        ]

    else:
        print(f"\nUsing fixed thresholds:")
        print(f"  Short: < {short_threshold} tokens")
        print(f"  Long: > {long_threshold} tokens")

        short_thinking_examples = [
            ex for ex in valid_responses if ex["thinking_length"] < short_threshold
        ]
        long_thinking_examples = [
            ex for ex in valid_responses if ex["thinking_length"] > long_threshold
        ]

    print(f"Found {len(long_thinking_examples)} long thinking examples")
    print(f"Found {len(short_thinking_examples)} short thinking examples")

    if n_short is None:
        short_examples = short_thinking_examples
        print(f"Using ALL {len(short_examples)} short examples (ThinkEdit style)")
    else:
        short_examples = (
            short_thinking_examples[:n_short]
            if len(short_thinking_examples) >= n_short
            else short_thinking_examples
        )
        if len(short_examples) < n_short:
            print(
                f"Warning: Only {len(short_examples)} short examples available, requested {n_short}"
            )

    if n_long is None:
        long_examples = long_thinking_examples
        print(f"Using ALL {len(long_examples)} long examples (ThinkEdit style)")
    else:
        long_examples = (
            long_thinking_examples[:n_long]
            if len(long_thinking_examples) >= n_long
            else long_thinking_examples
        )
        if len(long_examples) < n_long:
            print(
                f"Warning: Only {len(long_examples)} long examples available, requested {n_long}"
            )

    if len(short_examples) == 0 or len(long_examples) == 0:
        raise ValueError(
            f"Insufficient examples: {len(short_examples)} short, {len(long_examples)} long. Need at least 1 of each."
        )

    if short_examples:
        short_lengths = [item["thinking_length"] for item in short_examples]
        print(f"\nSelected short examples: {len(short_examples)}")
        print(f"  Length range: {min(short_lengths)} - {max(short_lengths)} tokens")
        print(f"  Average: {np.mean(short_lengths):.1f} tokens")

    if long_examples:
        long_lengths = [item["thinking_length"] for item in long_examples]
        print(f"\nSelected long examples: {len(long_examples)}")
        print(f"  Length range: {min(long_lengths)} - {max(long_lengths)} tokens")
        print(f"  Average: {np.mean(long_lengths):.1f} tokens")

    if short_examples and long_examples:
        print(f"  Separation: {min(long_lengths) - max(short_lengths)} tokens gap")

    return short_examples, long_examples


# %% Test example selection (uncomment to test interactively)
"""
def test_example_selection():
    \"\"\"Test example selection by length.\"\"\"
    try:
        print("Testing example selection...")
        
        with open("responses/Qwen3-0.6B_gsm8k_responses.json", "r") as f:
            responses = json.load(f)
            
        # Use first 50 examples for testing
        sample_responses = responses[:50]
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        sample_responses = compute_thinking_lengths(sample_responses, tokenizer)
        
        short_examples, long_examples = select_examples_by_length(
            sample_responses,
            n_short=None,
            n_long=None,
            use_percentiles=False,
            short_percentile=20.0,
            long_percentile=20.0,
            short_threshold=100,
            long_threshold=1000,
        )
        
        print(f"✓ Selected {len(short_examples)} short and {len(long_examples)} long examples")
        
        # Show some example short and long thinking
        if short_examples:
            print("\\nSample short thinking:")
            for i, ex in enumerate(short_examples[:2]):
                thinking = ex.get("with_thinking", {}).get("thinking", "")
                print(f"  {i+1}: {thinking[:150]}...")
                
        if long_examples:
            print("\\nSample long thinking:")
            for i, ex in enumerate(long_examples[:2]):
                thinking = ex.get("with_thinking", {}).get("thinking", "")
                print(f"  {i+1}: {thinking[:150]}...")
        
    except Exception as e:
        print(f"Example selection test failed: {e}")

# Uncomment to test:
# test_example_selection()
"""

# %% [markdown]
# ## Step 4: Activation Extraction
#
# Now we'll set up the infrastructure to extract activations from the model.
# We'll capture the residual stream activations that will be used to compute direction vectors.


# %% Activation extraction classes
class ImprovedActivationExtractor:
    """Improved activation extractor following ThinkEdit approach exactly."""

    def __init__(self, model, component="attn"):
        self.model = model
        self.component = component
        self.activations = []
        self.hooks = []
        self.setup_hooks()

    def setup_hooks(self):
        """Setup hooks for the specified component - matching ThinkEdit exactly."""
        if self.component == "attn":

            def capture_residual_hook():
                def hook_fn(module, input, output):
                    self.activations.append(input[0].detach())

                return hook_fn

            for layer in self.model.model.layers:
                hook = layer.post_attention_layernorm.register_forward_hook(
                    capture_residual_hook()
                )
                self.hooks.append(hook)

        elif self.component == "mlp":
            pass

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = []

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# %% Activation processing functions
def get_activations_for_example(model, tokenizer, extractor, question, thinking):
    """Get activations for a single example - matching ThinkEdit approach exactly."""

    if extractor.component == "attn":
        prompt_start = f"{question}<｜Assistant｜>"
        prompt_full = f"{question}<｜Assistant｜>{thinking}"

        toks_start = tokenizer(prompt_start).input_ids
        start = len(toks_start)
        toks_full = tokenizer(prompt_full).input_ids
        end = len(toks_full)

        toks = tokenizer(prompt_full, return_tensors="pt")
        toks = {k: v.to(model.device) for k, v in toks.items()}

        extractor.clear_activations()

        with torch.no_grad():
            _ = model(**toks)

        stacked_activations = torch.stack(extractor.activations, dim=0)[
            :, :, start - 1 : end - 1, :
        ]
        mean_activations = stacked_activations.mean(dim=2).cpu()

        return mean_activations.squeeze(1)

    elif extractor.component == "mlp":
        prompt_start = f"{question}<｜Assistant｜>"
        prompt_full = f"{question}<｜Assistant｜>{thinking}"

        toks_start = tokenizer(prompt_start).input_ids
        start = len(toks_start)
        toks_full = tokenizer(prompt_full).input_ids
        end = len(toks_full)

        toks = tokenizer(prompt_full, return_tensors="pt")
        toks = {k: v.to(model.device) for k, v in toks.items()}

        with torch.no_grad():
            outputs = model(**toks, output_hidden_states=True)
            residual_outputs = outputs.hidden_states[1:]

        stacked_activations = torch.stack(residual_outputs, dim=0)[
            :, :, start - 1 : end - 1, :
        ]
        mean_activations = stacked_activations.mean(dim=2).cpu()

        return mean_activations.squeeze(1)

    else:
        raise ValueError(f"Unsupported component: {extractor.component}")


# %% Test activation extraction (uncomment to test interactively)
"""
def test_single_activation_extraction():
    \"\"\"Test activation extraction on a single example.\"\"\"
    try:
        print("Testing single activation extraction...")
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", torch_dtype="auto", device_map="cpu"  # Use CPU for testing
        )
        model.eval()
        
        extractor = ImprovedActivationExtractor(model, "attn")
        
        # Test with a simple example
        test_question = "What is 2 + 3?"
        test_thinking = "I need to add 2 and 3. Let me think: 2 + 3 = 5."
        
        activations = get_activations_for_example(
            model, tokenizer, extractor, test_question, test_thinking
        )
        
        print(f"✓ Extracted activations with shape: {activations.shape}")
        print(f"  Number of layers: {activations.shape[0]}")
        print(f"  Hidden dimension: {activations.shape[1]}")
        print(f"  Mean activation magnitude: {activations.abs().mean().item():.6f}")
        
        extractor.remove_hooks()
        
    except Exception as e:
        print(f"Activation extraction test failed: {e}")

# Uncomment to test:
# test_single_activation_extraction()
"""

# %% [markdown]
# ## Step 5: Direction Extraction
#
# This is the core step where we compute direction vectors by contrasting
# the activations from short vs long thinking examples.


# %% Direction extraction functions
def extract_directions(
    model, tokenizer, short_examples, long_examples, component="attn"
):
    """Extract direction vectors by contrasting short vs long thinking examples - ThinkEdit style."""

    logger = get_logger()
    logger.info(f"Extracting directions for {component} component...")

    extractor = ImprovedActivationExtractor(model, component)

    short_activations = []
    long_activations = []

    logger.info(f"Processing {len(short_examples)} short thinking examples...")
    for i, example in enumerate(tqdm(short_examples)):
        try:
            question = example["question"]
            if "thinking" in example and example["thinking"]:
                thinking = example["thinking"]
            elif "with_thinking" in example and "thinking" in example["with_thinking"]:
                thinking = example["with_thinking"]["thinking"]
            else:
                continue

            activations = get_activations_for_example(
                model, tokenizer, extractor, question, thinking
            )
            short_activations.append(activations)

            if i % 20 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing short example {i}: {e}")
            continue

    logger.info(f"Processing {len(long_examples)} long thinking examples...")
    for i, example in enumerate(tqdm(long_examples)):
        try:
            question = example["question"]
            if "thinking" in example and example["thinking"]:
                thinking = example["thinking"]
            elif "with_thinking" in example and "thinking" in example["with_thinking"]:
                thinking = example["with_thinking"]["thinking"]
            else:
                continue

            activations = get_activations_for_example(
                model, tokenizer, extractor, question, thinking
            )
            long_activations.append(activations)

            if i % 20 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing long example {i}: {e}")
            continue

    logger.info("Computing direction vectors...")

    if len(short_activations) == 0:
        raise ValueError(
            "No short thinking examples processed successfully. Cannot compute direction."
        )
    if len(long_activations) == 0:
        raise ValueError(
            "No long thinking examples processed successfully. Cannot compute direction."
        )

    logger.info(
        f"Successfully processed {len(short_activations)} short and {len(long_activations)} long examples"
    )

    short_stack = torch.stack(short_activations, dim=0)
    long_stack = torch.stack(long_activations, dim=0)

    mean_embedding_short = short_stack.mean(dim=0)
    mean_embedding_long = long_stack.mean(dim=0)

    thinking_length_direction = mean_embedding_long - mean_embedding_short

    for layer_idx in range(min(3, thinking_length_direction.shape[0])):
        direction_layer = thinking_length_direction[layer_idx]
        print(
            f"  Layer {layer_idx}: short_mean={mean_embedding_short[layer_idx].mean().item():.6f}, "
            f"long_mean={mean_embedding_long[layer_idx].mean().item():.6f}, "
            f"direction_mean={direction_layer.mean().item():.6f}, "
            f"norm={torch.norm(direction_layer).item():.6f}"
        )

    extractor.remove_hooks()
    del short_activations, long_activations
    torch.cuda.empty_cache()

    logger.info(
        f"Extracted direction tensor with shape: {thinking_length_direction.shape}"
    )
    return thinking_length_direction


# %% [markdown]
# ## Step 6: Saving and Visualization
#
# Finally, we'll save the extracted direction vectors and create visualizations
# to understand their properties.


# %% Save and visualization functions
def save_directions(
    directions, output_dir, model_name, component, use_percentiles=False
):
    """Save the extracted directions."""
    os.makedirs(output_dir, exist_ok=True)

    model_short_name = model_name.split("/")[-1]
    method_suffix = "percentiles" if use_percentiles else "thresholds"
    filename = f"{model_short_name}_thinking_length_direction_gsm8k_{component}_{method_suffix}.pt"
    filepath = os.path.join(output_dir, filename)

    torch.save(directions, filepath)
    print(f"Directions saved to: {filepath}")

    return filepath


def visualize_directions(
    directions, output_dir, model_name, component, use_percentiles=False
):
    """Visualize direction magnitudes across layers."""
    num_layers = directions.shape[0]
    layer_indices = list(range(num_layers))
    direction_norms = [torch.norm(directions[i]).item() for i in range(num_layers)]

    plt.figure(figsize=(12, 6))
    plt.plot(layer_indices, direction_norms, "o-", linewidth=2, markersize=6)
    plt.xlabel("Layer Index")
    plt.ylabel("Direction Magnitude")

    selection_method = "Percentiles" if use_percentiles else "Fixed Thresholds"
    plt.title(
        f"Reasoning Length Direction Magnitudes ({component.upper()}) - {model_name}\n"
        f"Selection: {selection_method}"
    )
    plt.grid(True, alpha=0.3)

    model_short_name = model_name.split("/")[-1]
    method_suffix = "percentiles" if use_percentiles else "thresholds"
    plot_filename = f"{model_short_name}_thinkedit_style_{component}_direction_magnitudes_{method_suffix}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {plot_path}")

    plt.show()


# %% Test visualization (uncomment to test interactively)
"""
def test_visualization():
    \"\"\"Test direction visualization with dummy data.\"\"\"
    try:
        print("Testing visualization with dummy direction data...")
        
        # Create dummy direction vectors
        num_layers = 12
        hidden_dim = 768
        dummy_directions = torch.randn(num_layers, hidden_dim) * 0.1
        
        visualize_directions(
            dummy_directions, 
            "test_output", 
            "test_model", 
            "attn", 
            use_percentiles=False
        )
        
        print("✓ Visualization test completed")
        
    except Exception as e:
        print(f"Visualization test failed: {e}")

# Uncomment to test:
# test_visualization()
"""

# %% [markdown]
# ## Main Processing Function
#
# This ties everything together into a complete pipeline.


# %% Main processing function
def main():
    args = parse_args()

    logger = setup_logging("extract_reasoning_length_direction_improved")

    logger.info("=" * 60)
    logger.info("IMPROVED REASONING LENGTH DIRECTION EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Component: {args.component}")
    logger.info(
        f"Short examples: {args.n_short if args.n_short is not None else 'ALL (ThinkEdit style)'}"
    )
    logger.info(
        f"Long examples: {args.n_long if args.n_long is not None else 'ALL (ThinkEdit style)'}"
    )
    logger.info(f"Device: {args.device}")

    if args.use_percentiles:
        logger.info(f"Selection method: Percentiles")
        logger.info(f"  Short: bottom {args.short_percentile}%")
        logger.info(f"  Long: top {args.long_percentile}%")
    else:
        logger.info(f"Selection method: Fixed thresholds")
        logger.info(f"  Short: < {args.short_threshold} tokens")
        logger.info(f"  Long: > {args.long_threshold} tokens")

    logger.info(f"Loading responses from: {args.responses_file}")
    with open(args.responses_file, "r") as f:
        responses = json.load(f)
    logger.info(f"Loaded {len(responses)} responses")

    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device
    )
    model.eval()
    logger.info(f"Model loaded on device: {args.device}")

    if args.recompute_lengths or not any(
        "thinking_length" in item for item in responses
    ):
        logger.info("Computing thinking lengths...")
        responses = compute_thinking_lengths(responses, tokenizer)

    short_examples, long_examples = select_examples_by_length(
        responses,
        args.n_short,
        args.n_long,
        args.use_percentiles,
        args.short_percentile,
        args.long_percentile,
        args.short_threshold,
        args.long_threshold,
    )

    if args.component == "both":
        components = ["attn", "mlp"]
    else:
        components = [args.component]

    for comp in components:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {comp.upper()} component")
        logger.info(f"{'='*40}")

        directions = extract_directions(
            model, tokenizer, short_examples, long_examples, comp
        )

        save_directions(
            directions, args.output_dir, args.model, comp, args.use_percentiles
        )

        visualize_directions(
            directions, args.output_dir, args.model, comp, args.use_percentiles
        )

    logger.info(f"\n{'='*60}")
    logger.info("DIRECTION EXTRACTION COMPLETE!")
    logger.info(f"{'='*60}")


# %% [markdown]
# ## Summary and Next Steps
#
# Congratulations! You've successfully extracted reasoning length direction vectors.
#
# **What we accomplished:**
# 1. Loaded and analyzed model responses with varying reasoning lengths
# 2. Computed thinking lengths using tokenization
# 3. Selected contrasting examples (short vs long reasoning)
# 4. Extracted neural activations for these examples
# 5. Computed direction vectors by contrasting activations
# 6. Saved and visualized the results
#
# **Next steps:**
# - Use these direction vectors with `steer_reasoning_length.py` to control reasoning length
# - Experiment with different thresholds or percentiles for example selection
# - Try extracting directions from different model components (attention vs MLP)
# - Analyze the relationship between direction magnitude and steering effectiveness

# %% Script execution
if __name__ == "__main__":
    main()
