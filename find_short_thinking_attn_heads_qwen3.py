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
        default=10,
        help="Number of top contributing heads to identify",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="thinkedit_analysis",
        help="Directory to save analysis results",
    )
    return parser.parse_args()


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

    # Set up device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

    # Process responses to find short thinking examples
    logger.info("Processing responses to identify short thinking examples...")
    short_thinking_examples = []

    for example in responses_data:
        if "with_thinking" in example and "thinking" in example["with_thinking"]:
            thinking_text = example["with_thinking"]["thinking"]
            thinking_length = calculate_thinking_length(thinking_text, tokenizer)

            if thinking_length > 0 and thinking_length < args.short_thinking_threshold:
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

    thinking_length_direction = torch.load(direction_file).to(device)
    thinking_length_direction = thinking_length_direction / torch.norm(
        thinking_length_direction, dim=-1, keepdim=True
    )

    # Get model configuration
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", hidden_size // num_heads)

    logger.info(
        f"Model config: {num_layers} layers, {num_heads} heads, {head_dim} head_dim, {hidden_size} hidden_size"
    )

    # Set up hooks to capture attention contributions
    attn_contribution = []

    def capture_attn_contribution_hook():
        def hook_fn(module, input, output):
            # Debug: print tensor shapes to understand the actual input format
            # Get the input tensor to o_proj (this should be the attention output)
            if len(input) > 0:
                attn_input = input[0].detach()  # This is the input to o_proj

                # Debug logging for first few calls
                if len(attn_contribution) < 3:  # Only log for first few layers
                    logger.info(f"Debug - o_proj input shape: {attn_input.shape}")
                    logger.info(f"Debug - o_proj weight shape: {module.weight.shape}")
                    # Calculate what the input dimension should be
                    expected_input_dim = module.weight.shape[
                        1
                    ]  # input features to o_proj
                    logger.info(
                        f"Debug - o_proj expected input dim: {expected_input_dim}"
                    )

                # Use the actual input dimension from the o_proj layer
                o_proj_input_dim = module.weight.shape[1]

                # Handle different possible input shapes
                if len(attn_input.shape) == 3:  # [batch_size, seq_len, input_dim]
                    batch_size, seq_len, actual_input_dim = attn_input.shape
                    if batch_size == 1:  # Single batch
                        attn_out = attn_input[0, :, :]  # [seq_len, input_dim]
                    else:
                        logger.warning(
                            f"Unexpected batch size: {batch_size}, using first batch"
                        )
                        attn_out = attn_input[0, :, :]  # [seq_len, input_dim]
                elif len(attn_input.shape) == 2:  # [seq_len, input_dim]
                    attn_out = attn_input
                    actual_input_dim = attn_input.shape[-1]
                else:
                    logger.error(f"Unexpected input shape: {attn_input.shape}")
                    return

                # Verify the input dimension matches what o_proj expects
                if actual_input_dim != o_proj_input_dim:
                    logger.error(
                        f"Input dim mismatch: o_proj expects {o_proj_input_dim}, got {actual_input_dim}"
                    )
                    return

                # The input dimension should be num_heads * head_dim
                if actual_input_dim != num_heads * head_dim:
                    logger.warning(
                        f"Input dim {actual_input_dim} != num_heads * head_dim ({num_heads * head_dim})"
                    )
                    # Adjust our understanding - use the actual input dimension
                    effective_head_dim = actual_input_dim // num_heads
                    logger.info(f"Using effective head_dim: {effective_head_dim}")
                else:
                    effective_head_dim = head_dim

                # Now reshape to separate heads using the effective head dimension
                seq_len = attn_out.shape[0]
                try:
                    attn_out = attn_out.reshape(
                        seq_len, num_heads, effective_head_dim
                    )  # [seq_len, num_heads, effective_head_dim]
                except RuntimeError as e:
                    logger.error(f"Reshape failed: {e}")
                    logger.error(
                        f"Tensor shape: {attn_out.shape}, trying to reshape to [{seq_len}, {num_heads}, {effective_head_dim}]"
                    )
                    logger.error(
                        f"Expected total elements: {seq_len * num_heads * effective_head_dim}, actual: {attn_out.numel()}"
                    )
                    return

                # Get o_proj weights and reshape
                o_proj = module.weight.detach().clone()
                # o_proj shape should be [hidden_size, num_heads * effective_head_dim]
                o_proj = (
                    o_proj.reshape(hidden_size, num_heads, effective_head_dim)
                    .permute(1, 2, 0)
                    .contiguous()
                )
                # [num_heads, effective_head_dim, hidden_size]

                # Calculate per-head contribution: [seq_len, num_heads, hidden_size]
                contribution = torch.einsum("snk,nkh->snh", attn_out, o_proj)
                attn_contribution.append(contribution)
            else:
                logger.error("No input found in hook")

        return hook_fn

    # Register hooks
    hooks = []
    for layer in model.model.layers:
        hook = layer.self_attn.o_proj.register_forward_hook(
            capture_attn_contribution_hook()
        )
        hooks.append(hook)

    # Analyze short thinking examples
    logger.info("Analyzing attention head contributions...")
    avg_contribution = np.zeros((num_layers, num_heads))

    for i, example in enumerate(short_thinking_examples):
        if i % 10 == 0:
            logger.info(f"Processing example {i+1}/{len(short_thinking_examples)}")

        question = example["question"]
        thinking_text = example["with_thinking"]["thinking"]

        # Create input prompt
        prompt = f"<|User|>{question}<|Assistant|>"
        input_tokens = tokenizer(prompt).input_ids
        start_pos = len(input_tokens)

        # Add thinking content
        full_prompt = f"<|User|>{question}<|Assistant|><think>{thinking_text}</think>"
        full_tokens = tokenizer(full_prompt, return_tensors="pt")
        end_pos = len(full_tokens["input_ids"][0])

        # Forward pass
        with torch.no_grad():
            _ = model(
                input_ids=full_tokens["input_ids"].to(device),
                attention_mask=full_tokens["attention_mask"].to(device),
            )

            # Calculate mean contributions for thinking tokens
            thinking_contributions = []
            for layer_idx, tensor in enumerate(attn_contribution):
                # Average over thinking token positions
                layer_contribution = tensor[start_pos - 1 : end_pos - 1, :, :].mean(
                    dim=0
                )  # [num_heads, hidden_size]
                thinking_contributions.append(layer_contribution)

            all_head_contributions = torch.stack(
                thinking_contributions, dim=0
            )  # [num_layers, num_heads, hidden_size]

            # Debug: check thinking direction shape
            if i == 0:  # Only log for first example
                logger.info(
                    f"Debug - thinking_length_direction shape: {thinking_length_direction.shape}"
                )
                logger.info(
                    f"Debug - all_head_contributions shape: {all_head_contributions.shape}"
                )

            # Calculate dot product with negative thinking direction (since we want short thinking)
            # Handle different shapes of thinking_length_direction
            if len(thinking_length_direction.shape) == 2:
                # Shape: [num_layers, hidden_size] or [num_layers, 1]
                if thinking_length_direction.shape[1] == 1:
                    # Case: [num_layers, 1] - squeeze to [num_layers]
                    direction = -thinking_length_direction[:, 0].float()
                    dot_products = torch.einsum(
                        "ijl,il->ij",
                        all_head_contributions.float(),
                        direction.unsqueeze(1),
                    )
                else:
                    # Case: [num_layers, hidden_size]
                    direction = -thinking_length_direction.float()
                    dot_products = torch.einsum(
                        "ijl,il->ij", all_head_contributions.float(), direction
                    )
            elif len(thinking_length_direction.shape) == 1:
                # Shape: [hidden_size] - broadcast to all layers
                direction = -thinking_length_direction.float()
                dot_products = torch.einsum(
                    "ijl,l->ij", all_head_contributions.float(), direction
                )
            else:
                logger.error(
                    f"Unexpected thinking_length_direction shape: {thinking_length_direction.shape}"
                )
                continue

            avg_contribution += dot_products.cpu().numpy()

        # Clear for next iteration
        attn_contribution = []

        # Memory cleanup
        if i % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Normalize contributions
    avg_contribution = avg_contribution / len(short_thinking_examples)

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
