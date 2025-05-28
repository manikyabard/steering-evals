#!/usr/bin/env python3
"""
Improved Reasoning Length Direction Extraction

This script extracts reasoning length directions following the ThinkEdit approach more closely:
1. Uses fixed sample sizes (100 examples each) instead of percentages
2. Computes thinking lengths using word counts
3. Better sample selection and direction computation
4. More thorough debugging and validation

Based on: https://github.com/Trustworthy-ML-Lab/ThinkEdit/blob/main/extract_thinking_length_direction_gsm8k_attn.py
"""

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
        default=100,
        help="Number of short thinking examples to use",
    )
    parser.add_argument(
        "--n_long",
        type=int,
        default=100,
        help="Number of long thinking examples to use",
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
    return parser.parse_args()


def count_thinking_tokens(thinking_text, tokenizer):
    """Count tokens in thinking text, matching ThinkEdit approach exactly."""
    if not thinking_text or thinking_text.strip() == "":
        return 0

    # Use tokenizer to count tokens like ThinkEdit
    tokens = tokenizer.encode(thinking_text, add_special_tokens=False)
    return len(tokens)


def compute_thinking_lengths(responses, tokenizer):
    """Compute thinking lengths for all responses."""
    print("Computing thinking lengths...")

    for item in tqdm(responses, desc="Computing thinking lengths"):
        # Handle both formats: direct 'thinking' field or nested 'with_thinking'
        if "thinking" in item and item["thinking"]:
            thinking_text = item["thinking"]
            item["thinking_length"] = count_thinking_tokens(thinking_text, tokenizer)
        elif "with_thinking" in item and "thinking" in item["with_thinking"]:
            thinking_text = item["with_thinking"]["thinking"]
            item["thinking_length"] = count_thinking_tokens(thinking_text, tokenizer)
        else:
            item["thinking_length"] = -1  # Invalid/missing thinking

    return responses


def select_examples_by_length(responses, n_short=100, n_long=100):
    """Select examples based on thinking length, using ThinkEdit's approach."""

    # Filter valid responses (ThinkEdit uses thinking_length != -1)
    valid_responses = [
        item
        for item in responses
        if "thinking_length" in item and item["thinking_length"] != -1
    ]

    print(f"Found {len(valid_responses)} responses with valid thinking")

    # ThinkEdit's filtering approach: hard thresholds
    # Long thinking: > 1000 tokens, Short thinking: < 100 tokens
    long_thinking_examples = [
        ex for ex in valid_responses if ex["thinking_length"] > 1000
    ]
    short_thinking_examples = [
        ex for ex in valid_responses if ex["thinking_length"] < 100
    ]

    print(f"Found {len(long_thinking_examples)} long thinking examples (>1000 tokens)")
    print(f"Found {len(short_thinking_examples)} short thinking examples (<100 tokens)")

    # Use available examples (may be less than requested n_short/n_long)
    short_examples = (
        short_thinking_examples[:n_short]
        if len(short_thinking_examples) >= n_short
        else short_thinking_examples
    )
    long_examples = (
        long_thinking_examples[:n_long]
        if len(long_thinking_examples) >= n_long
        else long_thinking_examples
    )

    if len(short_examples) < n_short:
        print(
            f"Warning: Only {len(short_examples)} short examples available, requested {n_short}"
        )
    if len(long_examples) < n_long:
        print(
            f"Warning: Only {len(long_examples)} long examples available, requested {n_long}"
        )

    # Ensure we have at least some examples
    if len(short_examples) == 0 or len(long_examples) == 0:
        raise ValueError(
            f"Insufficient examples: {len(short_examples)} short, {len(long_examples)} long. Need at least 1 of each."
        )

    # Print selected examples stats
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


class ImprovedActivationExtractor:
    """Improved activation extractor following ThinkEdit approach exactly."""

    def __init__(self, model, component="attn"):
        self.model = model
        self.component = component
        self.activations = []  # Changed to list to match ThinkEdit
        self.hooks = []
        self.setup_hooks()

    def setup_hooks(self):
        """Setup hooks for the specified component - matching ThinkEdit exactly."""
        if self.component == "attn":
            # For attention: hook post_attention_layernorm and capture input[0] (residual stream)
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
            # For MLP: we'll use hidden_states approach during forward pass
            # This will be handled differently in get_activations_for_example
            pass

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = []

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def get_activations_for_example(model, tokenizer, extractor, question, thinking):
    """Get activations for a single example - matching ThinkEdit approach exactly."""

    if extractor.component == "attn":
        # ThinkEdit attention approach: use hooks on post_attention_layernorm
        # Create the prompt in ThinkEdit format with Unicode characters
        prompt_start = f"<｜User｜>{question}<｜Assistant｜>"
        prompt_full = f"<｜User｜>{question}<｜Assistant｜>{thinking}"

        # Get token positions
        toks_start = tokenizer(prompt_start).input_ids
        start = len(toks_start)
        toks_full = tokenizer(prompt_full).input_ids
        end = len(toks_full)

        # Tokenize for model input
        toks = tokenizer(prompt_full, return_tensors="pt")
        toks = {k: v.to(model.device) for k, v in toks.items()}

        # Clear previous activations
        extractor.clear_activations()

        # Forward pass to collect activations
        with torch.no_grad():
            _ = model(**toks)

        # Process activations: stack all layers, slice thinking portion, mean over tokens
        # Shape: [num_layers, batch_size, seq_len, hidden_size] -> [num_layers, hidden_size]
        stacked_activations = torch.stack(extractor.activations, dim=0)[
            :, :, start - 1 : end - 1, :
        ]
        mean_activations = stacked_activations.mean(
            dim=2
        ).cpu()  # Average over thinking tokens

        return mean_activations.squeeze(
            1
        )  # Remove batch dimension: [num_layers, hidden_size]

    elif extractor.component == "mlp":
        # ThinkEdit MLP approach: use output_hidden_states
        prompt_start = f"<｜User｜>{question}<｜Assistant｜>"
        prompt_full = f"<｜User｜>{question}<｜Assistant｜>{thinking}"

        # Get token positions
        toks_start = tokenizer(prompt_start).input_ids
        start = len(toks_start)
        toks_full = tokenizer(prompt_full).input_ids
        end = len(toks_full)

        # Tokenize for model input
        toks = tokenizer(prompt_full, return_tensors="pt")
        toks = {k: v.to(model.device) for k, v in toks.items()}

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(**toks, output_hidden_states=True)
            residual_outputs = outputs.hidden_states[1:]  # Skip embedding layer

        # Process like ThinkEdit: stack layers, slice thinking portion, mean over tokens
        stacked_activations = torch.stack(residual_outputs, dim=0)[
            :, :, start - 1 : end - 1, :
        ]
        mean_activations = stacked_activations.mean(
            dim=2
        ).cpu()  # Average over thinking tokens

        return mean_activations.squeeze(
            1
        )  # Remove batch dimension: [num_layers, hidden_size]

    else:
        raise ValueError(f"Unsupported component: {extractor.component}")


def extract_directions(
    model, tokenizer, short_examples, long_examples, component="attn"
):
    """Extract direction vectors by contrasting short vs long thinking examples - ThinkEdit style."""

    print(f"Extracting directions for {component} component...")

    # Setup activation extractor
    extractor = ImprovedActivationExtractor(model, component)

    # Storage for activations - now storing tensors directly
    short_activations = []
    long_activations = []

    # Process short thinking examples
    print(f"Processing {len(short_examples)} short thinking examples...")
    for i, example in enumerate(tqdm(short_examples)):
        try:
            question = example["question"]
            # Handle both formats: direct 'thinking' field or nested 'with_thinking'
            if "thinking" in example and example["thinking"]:
                thinking = example["thinking"]
            elif "with_thinking" in example and "thinking" in example["with_thinking"]:
                thinking = example["with_thinking"]["thinking"]
            else:
                continue  # Skip if no thinking found

            # Get activations - now returns [num_layers, hidden_size]
            activations = get_activations_for_example(
                model, tokenizer, extractor, question, thinking
            )
            short_activations.append(activations)

            # Memory cleanup
            if i % 20 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing short example {i}: {e}")
            continue

    # Process long thinking examples
    print(f"Processing {len(long_examples)} long thinking examples...")
    for i, example in enumerate(tqdm(long_examples)):
        try:
            question = example["question"]
            # Handle both formats: direct 'thinking' field or nested 'with_thinking'
            if "thinking" in example and example["thinking"]:
                thinking = example["thinking"]
            elif "with_thinking" in example and "thinking" in example["with_thinking"]:
                thinking = example["with_thinking"]["thinking"]
            else:
                continue  # Skip if no thinking found

            # Get activations - now returns [num_layers, hidden_size]
            activations = get_activations_for_example(
                model, tokenizer, extractor, question, thinking
            )
            long_activations.append(activations)

            # Memory cleanup
            if i % 20 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing long example {i}: {e}")
            continue

    # Compute mean activations and directions - ThinkEdit style
    print("Computing direction vectors...")

    # Check if we have sufficient examples
    if len(short_activations) == 0:
        raise ValueError(
            "No short thinking examples processed successfully. Cannot compute direction."
        )
    if len(long_activations) == 0:
        raise ValueError(
            "No long thinking examples processed successfully. Cannot compute direction."
        )

    print(
        f"Successfully processed {len(short_activations)} short and {len(long_activations)} long examples"
    )

    # Stack all examples: [num_examples, num_layers, hidden_size]
    short_stack = torch.stack(short_activations, dim=0)
    long_stack = torch.stack(long_activations, dim=0)

    # Compute means across examples: [num_layers, hidden_size]
    mean_embedding_short = short_stack.mean(dim=0)
    mean_embedding_long = long_stack.mean(dim=0)

    # Direction: long - short (NO NORMALIZATION like ThinkEdit)
    thinking_length_direction = mean_embedding_long - mean_embedding_short

    # Debug info for first few layers
    for layer_idx in range(min(3, thinking_length_direction.shape[0])):
        direction_layer = thinking_length_direction[layer_idx]
        print(
            f"  Layer {layer_idx}: short_mean={mean_embedding_short[layer_idx].mean().item():.6f}, "
            f"long_mean={mean_embedding_long[layer_idx].mean().item():.6f}, "
            f"direction_mean={direction_layer.mean().item():.6f}, "
            f"norm={torch.norm(direction_layer).item():.6f}"
        )

    # Cleanup
    extractor.remove_hooks()
    del short_activations, long_activations
    torch.cuda.empty_cache()

    print(f"Extracted direction tensor with shape: {thinking_length_direction.shape}")
    return thinking_length_direction


def save_directions(directions, output_dir, model_name, component):
    """Save the extracted directions."""
    os.makedirs(output_dir, exist_ok=True)

    model_short_name = model_name.split("/")[-1]
    filename = f"{model_short_name}_thinking_length_direction_gsm8k_{component}.pt"
    filepath = os.path.join(output_dir, filename)

    torch.save(directions, filepath)
    print(f"Directions saved to: {filepath}")

    return filepath


def visualize_directions(directions, output_dir, model_name, component):
    """Visualize direction magnitudes across layers."""
    # directions is now a tensor of shape [num_layers, hidden_size]
    num_layers = directions.shape[0]
    layer_indices = list(range(num_layers))
    direction_norms = [torch.norm(directions[i]).item() for i in range(num_layers)]

    plt.figure(figsize=(12, 6))
    plt.plot(layer_indices, direction_norms, "o-", linewidth=2, markersize=6)
    plt.xlabel("Layer Index")
    plt.ylabel("Direction Magnitude")
    plt.title(
        f"Reasoning Length Direction Magnitudes ({component.upper()}) - {model_name}"
    )
    plt.grid(True, alpha=0.3)

    # Save plot
    model_short_name = model_name.split("/")[-1]
    plot_filename = (
        f"{model_short_name}_thinkedit_style_{component}_direction_magnitudes.png"
    )
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {plot_path}")

    plt.show()


def main():
    args = parse_args()

    print("=" * 60)
    print("IMPROVED REASONING LENGTH DIRECTION EXTRACTION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Component: {args.component}")
    print(f"Short examples: {args.n_short}")
    print(f"Long examples: {args.n_long}")
    print(f"Device: {args.device}")

    # Load responses
    print(f"\nLoading responses from: {args.responses_file}")
    with open(args.responses_file, "r") as f:
        responses = json.load(f)
    print(f"Loaded {len(responses)} responses")

    # Load model first to get tokenizer for length computation
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device
    )
    model.eval()

    # Compute thinking lengths if needed
    if args.recompute_lengths or not any(
        "thinking_length" in item for item in responses
    ):
        print("Computing thinking lengths...")
        responses = compute_thinking_lengths(responses, tokenizer)

    # Select examples
    short_examples, long_examples = select_examples_by_length(
        responses, args.n_short, args.n_long
    )

    # Extract directions
    if args.component == "both":
        components = ["attn", "mlp"]
    else:
        components = [args.component]

    for comp in components:
        print(f"\n{'='*40}")
        print(f"Processing {comp.upper()} component")
        print(f"{'='*40}")

        directions = extract_directions(
            model, tokenizer, short_examples, long_examples, comp
        )

        # Save directions
        save_directions(directions, args.output_dir, args.model, comp)

        # Visualize directions
        visualize_directions(directions, args.output_dir, args.model, comp)

    print(f"\n{'='*60}")
    print("DIRECTION EXTRACTION COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
