#!/usr/bin/env python3
"""
Find short thinking attention heads for Qwen3 models.

This script identifies attention heads that contribute to overly short reasoning
by analyzing their contributions to the short reasoning direction.

Usage:
    python find_short_thinking_attn_heads_qwen3.py --model Qwen/Qwen3-0.6B
    python find_short_thinking_attn_heads_qwen3.py --model Qwen/Qwen3-0.6B --responses_file custom_responses.json
    python find_short_thinking_attn_heads_qwen3.py --model Qwen/Qwen3-0.6B --directions_file custom_directions.pt
    python find_short_thinking_attn_heads_qwen3.py --model Qwen/Qwen3-0.6B --responses_file custom_responses.json --directions_file custom_directions.pt
"""

import gc
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pickle
import time
import json
import math
import re
import matplotlib.pyplot as plt
from logging_setup import setup_logging, get_logger

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find short thinking attention heads for Qwen3 models"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--responses_file",
        type=str,
        default=None,
        help="Path to responses JSON file (auto-detected from model name if not provided)",
    )
    parser.add_argument(
        "--directions_file",
        type=str,
        default=None,
        help="Path to thinking length directions file (auto-detected from model name if not provided)",
    )
    parser.add_argument(
        "--layer_start", type=int, default=0, help="Start layer for visualization"
    )
    parser.add_argument(
        "--layer_end",
        type=int,
        default=-1,
        help="End layer for visualization (-1 for all layers)",
    )
    parser.add_argument(
        "--short_thinking_threshold",
        type=int,
        default=100,
        help="Threshold for considering thinking as 'short' (in tokens)",
    )
    parser.add_argument(
        "--top_k_heads",
        type=int,
        default=None,
        help="Number of top contributing heads to identify (auto-detected from model size if not provided)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="thinkedit_analysis",
        help="Directory to save analysis results",
    )
    return parser.parse_args()


def get_model_size_info(model_name_or_path):
    """Extract model size information and set appropriate defaults."""
    model_str = str(model_name_or_path).lower()

    # Determine model size and set appropriate defaults following ThinkEdit methodology
    if "0.6b" in model_str or "0.5b" in model_str:
        size_category = "small"
        default_k = 10
        default_threshold = 100
    elif "1.5b" in model_str or "1b" in model_str:
        size_category = "small"
        default_k = 10
        default_threshold = 100
    elif "4b" in model_str or "3b" in model_str:
        size_category = "medium"
        default_k = 20
        default_threshold = 200
    elif "7b" in model_str or "8b" in model_str:
        size_category = "medium"
        default_k = 20
        default_threshold = 200
    elif "14b" in model_str or "13b" in model_str:
        size_category = "large"
        default_k = 40
        default_threshold = 300
    else:
        # Default for unknown sizes
        size_category = "medium"
        default_k = 20
        default_threshold = 200

    return {
        "size_category": size_category,
        "default_k": default_k,
        "default_threshold": default_threshold,
    }


def top_k_head(matrix, k=20, reverse=True):
    """Find top k heads from contribution matrix."""
    flattened = [
        (value, (i, j)) for i, row in enumerate(matrix) for j, value in enumerate(row)
    ]
    return sorted(flattened, key=lambda x: x[0], reverse=reverse)[:k]


def extract_thinking_content(response_text):
    """Extract thinking content from model response."""
    # Look for <think>...</think> tags
    think_pattern = r"<think>(.*?)</think>"
    match = re.search(think_pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def calculate_thinking_length(thinking_text, tokenizer):
    """Calculate thinking length in tokens."""
    if not thinking_text:
        return 0
    tokens = tokenizer.encode(thinking_text)
    return len(tokens)


def main():
    args = parse_args()

    # Set up logging
    logger = get_logger()

    # Get model size information and set defaults
    model_size_info = get_model_size_info(args.model)

    # Set defaults based on model size if not provided
    if args.top_k_heads is None:
        args.top_k_heads = model_size_info["default_k"]
        logger.info(
            f"Auto-detected model size category: {model_size_info['size_category']}"
        )
        logger.info(f"Setting top_k_heads to {args.top_k_heads} based on model size")

    # Suggest threshold if using default and it seems inappropriate for model size
    if (
        args.short_thinking_threshold == 100
        and model_size_info["default_threshold"] != 100
    ):
        logger.info(
            f"Consider using --short_thinking_threshold {model_size_info['default_threshold']} for {model_size_info['size_category']} models"
        )

    # Set up device
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract model name for file naming
    model_name = args.model.replace("/", "_").replace("-", "_")

    # Load responses data
    responses_file = (
        args.responses_file or f"responses/{model_name}_gsm8k_responses.json"
    )

    logger.info(f"Loading responses from {responses_file}")
    if not os.path.exists(responses_file):
        logger.error(f"Responses file not found: {responses_file}")
        if args.responses_file:
            logger.info(
                "Please check that the specified responses file path is correct"
            )
        else:
            logger.info(
                "Please run generate_responses_gsm8k.py first to generate responses, "
                "or specify a custom responses file with --responses_file"
            )
        return

    with open(responses_file, "r") as f:
        responses_data = json.load(f)

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Process responses to find short thinking examples - match ThinkEdit exactly
    logger.info("Processing responses to identify short thinking examples...")
    valid_responses = [
        ex
        for ex in responses_data
        if "with_thinking" in ex and "thinking" in ex["with_thinking"]
    ]

    short_thinking_examples = []
    for example in valid_responses:
        thinking_text = example["with_thinking"]["thinking"]
        thinking_length = calculate_thinking_length(thinking_text, tokenizer)
        if thinking_length > 0 and thinking_length < args.short_thinking_threshold:
            # Store thinking in the same format as ThinkEdit
            example["thinking"] = thinking_text
            example["thinking_length"] = thinking_length
            short_thinking_examples.append(example)

    logger.info(
        f"Found {len(short_thinking_examples)} examples with short thinking (< {args.short_thinking_threshold} tokens)"
    )

    if len(short_thinking_examples) < 10:
        logger.warning(
            f"Very few short thinking examples found. Consider increasing --short_thinking_threshold"
        )

    # Load thinking length direction
    direction_file = (
        args.directions_file
        or f"directions/{model_name}_thinking_length_direction_gsm8k_attn.pt"
    )

    logger.info(f"Loading thinking length direction from {direction_file}")
    if not os.path.exists(direction_file):
        logger.error(f"Direction file not found: {direction_file}")
        if args.directions_file:
            logger.info(
                "Please check that the specified directions file path is correct"
            )
        else:
            logger.info(
                "Please run extract_reasoning_length_direction_improved.py first to extract directions, "
                "or specify a custom directions file with --directions_file"
            )
        return

    thinking_length_direction = torch.load(direction_file, map_location=device)
    thinking_length_direction = thinking_length_direction / torch.norm(
        thinking_length_direction, dim=-1, keepdim=True
    )

    # Get model configuration
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    logger.info(
        f"Model config: {num_layers} layers, {num_heads} heads, {head_dim} head_dim, {hidden_size} hidden_size"
    )

    # Debug: Check actual o_proj dimensions
    sample_layer = model.model.layers[0].self_attn.o_proj
    logger.info(f"Sample o_proj weight shape: {sample_layer.weight.shape}")
    logger.info(f"Expected: [{hidden_size}, {hidden_size}] or [{hidden_size}, ?]")

    # Validate model architecture compatibility
    def validate_architecture():
        """Quick validation that our hook will work with this model architecture."""
        try:
            # Test with a short dummy input
            dummy_input = tokenizer("Test", return_tensors="pt").to(device)

            test_contributions = []

            def test_hook(module, input, output):
                attn_out = input[0].detach()[0, :, :]
                test_contributions.append(attn_out.shape)

            # Register temporary hook on first layer
            hook = model.model.layers[0].self_attn.o_proj.register_forward_hook(
                test_hook
            )

            with torch.no_grad():
                _ = model(**dummy_input)

            hook.remove()

            if test_contributions:
                test_shape = test_contributions[0]
                logger.info(
                    f"Architecture validation: attention output shape = {test_shape}"
                )

                if test_shape[-1] == hidden_size:
                    logger.info(
                        "✓ Standard architecture detected - using direct processing"
                    )
                    return "standard"
                elif test_shape[-1] > hidden_size:
                    logger.info(
                        f"✓ Extended architecture detected - using fallback processing"
                    )
                    logger.info(
                        f"  Will use first {hidden_size} of {test_shape[-1]} dimensions"
                    )
                    return "extended"
                else:
                    logger.warning(
                        f"⚠ Unexpected architecture: attention dim {test_shape[-1]} < hidden_size {hidden_size}"
                    )
                    return "unexpected"
            else:
                logger.warning(
                    "⚠ No attention contributions captured during validation"
                )
                return "unknown"

        except Exception as e:
            logger.warning(f"Architecture validation failed: {e}")
            return "unknown"

    architecture_type = validate_architecture()

    # Set up hooks exactly like ThinkEdit but adapted for Qwen3 architecture
    attn_contribution = []

    def capture_attn_contribution_hook():
        def hook_fn(module, input, output):
            # Qwen3 attention: input to o_proj has shape [batch, seq_len, ?]
            # Let's check the actual attention output before o_proj
            attn_out = input[0].detach()[0, :, :]  # [seq_len, ?]

            # Check if this is the correct shape for attention heads
            if attn_out.size(-1) == hidden_size:
                # Standard case: reshape to [seq_len, num_heads, head_dim]
                attn_out = attn_out.reshape(attn_out.size(0), num_heads, head_dim)

                # Get o_proj weight and reshape it
                o_proj = module.weight.detach().clone()  # [hidden_size, hidden_size]
                o_proj = (
                    o_proj.reshape(hidden_size, num_heads, head_dim)
                    .permute(1, 2, 0)
                    .contiguous()
                )
                attn_contribution.append(torch.einsum("snk,nkh->snh", attn_out, o_proj))
            else:
                # Handle different architectures - Qwen3-4B seems to have different dimensions
                actual_dim = attn_out.size(-1)
                logger.warning(
                    f"Unexpected attention output shape: {attn_out.shape}, expected [..., {hidden_size}]"
                )

                # For Qwen3-4B, we expect the attention part to be in the first hidden_size dimensions
                if actual_dim > hidden_size:
                    # Take first hidden_size dimensions (assuming it's [attn_out, other])
                    attn_out = attn_out[:, :hidden_size]
                    attn_out = attn_out.reshape(attn_out.size(0), num_heads, head_dim)

                    # Get o_proj weight - it should match the full attention output dimension
                    o_proj = module.weight.detach().clone()  # [hidden_size, actual_dim]

                    # Take only the part that corresponds to the attention output
                    if o_proj.size(1) >= hidden_size:
                        o_proj = o_proj[:, :hidden_size]  # [hidden_size, hidden_size]

                    o_proj = (
                        o_proj.reshape(hidden_size, num_heads, head_dim)
                        .permute(1, 2, 0)
                        .contiguous()
                    )
                    attn_contribution.append(
                        torch.einsum("snk,nkh->snh", attn_out, o_proj)
                    )
                else:
                    # If actual_dim < hidden_size, something is wrong - skip this layer
                    logger.error(
                        f"Attention output dimension {actual_dim} is smaller than expected {hidden_size}"
                    )
                    # Add a zero tensor to maintain layer count
                    zero_contrib = torch.zeros(
                        attn_out.size(0), num_heads, hidden_size, device=attn_out.device
                    )
                    attn_contribution.append(zero_contrib)

        return hook_fn

    # Register hooks exactly like ThinkEdit
    for layer in model.model.layers:
        layer.self_attn.o_proj.register_forward_hook(capture_attn_contribution_hook())

    # Analyze short thinking examples - exactly like ThinkEdit
    logger.info("Analyzing attention head contributions...")
    avg_contribution = np.zeros((num_layers, num_heads))

    for i, example in enumerate(short_thinking_examples):
        if i % 10 == 0:
            logger.info(f"Processing example {i+1}/{len(short_thinking_examples)}")

        # Follow ThinkEdit format exactly
        toks = tokenizer(f"<|User|>{example['question']}<|Assistant|>").input_ids
        start = len(toks)
        toks = tokenizer(
            f"<|User|>{example['question']}<|Assistant|>{example['thinking']}"
        ).input_ids
        end = len(toks)
        toks = tokenizer(
            f"<|User|>{example['question']}<|Assistant|>{example['thinking']}",
            return_tensors="pt",
        )

        with torch.no_grad():
            _ = model(
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
            )

            # Check if we got any contributions
            if not attn_contribution:
                logger.error(
                    f"No attention contributions captured for example {i+1}. Skipping."
                )
                continue

            if len(attn_contribution) != num_layers:
                logger.warning(
                    f"Expected {num_layers} attention contributions, got {len(attn_contribution)}. Padding with zeros."
                )
                # Pad with zeros if we're missing layers
                while len(attn_contribution) < num_layers:
                    seq_len = toks["input_ids"].size(1)
                    zero_contrib = torch.zeros(seq_len, num_heads, hidden_size)
                    attn_contribution.append(zero_contrib)

            attn_mean_contributions = [
                tensor[start - 1 : end - 1, :, :]
                .mean(dim=0)
                .cpu()  # Move to CPU immediately
                for tensor in attn_contribution
            ]
            all_head_contributions = torch.stack(attn_mean_contributions, dim=0)

            # Fix ThinkEdit's einsum - they have a dimensional error
            # thinking_length_direction[:, 0] has shape [num_layers] so we need 'ijl,i->ij'
            dot_products = torch.einsum(
                "ijl,i->ij",
                all_head_contributions.float(),
                -thinking_length_direction[:, 0].cpu().float(),
            )
            avg_contribution += dot_products.numpy()

        # Clear like ThinkEdit
        attn_contribution = []

        # More aggressive memory cleanup for large models
        if i % 5 == 0:  # More frequent cleanup
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            gc.collect()

    # Normalize contributions
    avg_contribution = np.asarray(avg_contribution) / len(short_thinking_examples)

    # Determine layer range for visualization
    layer_start = max(0, args.layer_start)
    layer_end = num_layers if args.layer_end == -1 else min(num_layers, args.layer_end)

    # Find top contributing heads
    contribution_subset = avg_contribution[layer_start:layer_end, :]
    top_k_contributions = top_k_head(contribution_subset, k=args.top_k_heads)

    # Adjust layer indices back to global coordinates
    global_top_heads = [
        (c[0], (c[1][0] + layer_start, c[1][1])) for c in top_k_contributions
    ]
    top_head_coordinates = [(c[1][0], c[1][1]) for c in global_top_heads]

    logger.info(f"Top {args.top_k_heads} short thinking heads:")
    for contribution, (layer, head) in global_top_heads:
        logger.info(f"  Layer {layer}, Head {head}: {contribution:.4f}")

    # Save results
    results = {
        "model": args.model,
        "short_thinking_threshold": args.short_thinking_threshold,
        "num_short_examples": len(short_thinking_examples),
        "avg_contribution_matrix": avg_contribution.tolist(),
        "top_heads": top_head_coordinates,
        "top_contributions": [c[0] for c in global_top_heads],
        "layer_range": [layer_start, layer_end],
        "model_config": {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
        },
    }

    results_file = os.path.join(
        args.output_dir, f"{model_name}_short_thinking_heads_analysis.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Create heatmap visualization
    max_abs_value = np.abs(contribution_subset).max()

    plt.figure(figsize=(14, 10))
    plt.imshow(
        contribution_subset,
        cmap="coolwarm",
        aspect="auto",
        vmin=-max_abs_value,
        vmax=max_abs_value,
    )
    plt.colorbar(label="Average short thinking contribution")
    plt.title(
        f"{args.model}: Short Thinking Head Contributions\n"
        f"(Layers {layer_start}-{layer_end}, {len(short_thinking_examples)} short examples)"
    )
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")

    # Add head and layer labels
    plt.xticks(
        ticks=np.arange(num_heads)[:: max(1, num_heads // 10)],
        labels=[f"H{i}" for i in range(num_heads)][:: max(1, num_heads // 10)],
        fontsize=8,
        rotation=45,
    )
    plt.yticks(
        ticks=np.arange(layer_end - layer_start)[
            :: max(1, (layer_end - layer_start) // 10)
        ],
        labels=[f"L{i+layer_start}" for i in range(layer_end - layer_start)][
            :: max(1, (layer_end - layer_start) // 10)
        ],
        fontsize=8,
    )

    # Mark top heads
    for contribution, (layer, head) in global_top_heads[:5]:  # Mark top 5
        if layer_start <= layer < layer_end:
            plt.scatter(
                head,
                layer - layer_start,
                color="yellow",
                s=100,
                marker="*",
                edgecolors="black",
                linewidth=1,
                zorder=5,
            )

    plt.tight_layout()
    heatmap_file = os.path.join(
        args.output_dir, f"{model_name}_short_thinking_heads_heatmap.png"
    )
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved to {heatmap_file}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"SHORT THINKING HEAD ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Short thinking threshold: {args.short_thinking_threshold} tokens")
    print(f"Examples analyzed: {len(short_thinking_examples)}")
    print(f"Layer range: {layer_start}-{layer_end}")
    print(f"\nTop {args.top_k_heads} Short Thinking Heads:")
    print("-" * 40)
    for i, (contribution, (layer, head)) in enumerate(global_top_heads):
        print(f"{i+1:2d}. Layer {layer:2d}, Head {head:2d}: {contribution:7.4f}")
    print("\nFiles saved:")
    print(f"  - Analysis: {results_file}")
    print(f"  - Heatmap: {heatmap_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
