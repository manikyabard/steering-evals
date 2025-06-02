#!/usr/bin/env python3
"""
Find short thinking attention heads in Qwen3 models.

Usage:
    python find_short_thinking_attn_heads_qwen3.py --model Qwen/Qwen3-0.6B --responses-file custom_responses.json
    python find_short_thinking_attn_heads_qwen3.py --model Qwen/Qwen3-0.6B --directions-file custom_directions.pt
    python find_short_thinking_attn_heads_qwen3.py --model Qwen/Qwen3-0.6B --responses-file custom_responses.json --directions-file custom_directions.pt
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
    parser = argparse.ArgumentParser(description="Find short thinking attention heads")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--responses-file",
        type=str,
        default=None,
        help="Path to responses JSON file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--directions-file",
        type=str,
        default=None,
        help="Path to directions file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--layer-start", type=int, default=0, help="Start layer for visualization"
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=None,
        help="End layer for visualization (default: all layers)",
    )
    parser.add_argument(
        "--short-thinking-threshold",
        type=int,
        default=None,
        help="Threshold for short thinking in tokens (auto-determined if not provided)",
    )
    parser.add_argument(
        "--top-k-heads",
        type=int,
        default=None,
        help="Number of top heads to identify (auto-determined based on model size)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="thinkedit_analysis",
        help="Directory to save analysis results",
    )
    return parser.parse_args()


def get_model_size_info(model_name_or_path):
    """Extract model size information and set appropriate defaults."""
    model_str = str(model_name_or_path).lower()

    # Determine model size and set appropriate defaults following ThinkEdit methodology
    # Check larger sizes first to avoid substring matching issues
    if "70b" in model_str or "72b" in model_str:
        size_category = "xxxlarge"
        default_k = 60
        default_threshold = 500
    elif "32b" in model_str or "30b" in model_str:
        size_category = "xxlarge"
        default_k = 50
        default_threshold = 400
    elif "14b" in model_str or "13b" in model_str:
        size_category = "xlarge"
        default_k = 40
        default_threshold = 300
    elif "7b" in model_str or "8b" in model_str:
        size_category = "large"
        default_k = 30
        default_threshold = 250
    elif "4b" in model_str:
        size_category = "medium"
        default_k = 20
        default_threshold = 200
    elif "3b" in model_str:
        size_category = "medium"
        default_k = 15
        default_threshold = 150
    elif "1.5b" in model_str or "1b" in model_str:
        size_category = "small"
        default_k = 10
        default_threshold = 100
    elif "0.6b" in model_str or "0.5b" in model_str:
        size_category = "small"
        default_k = 10
        default_threshold = 100
    else:
        # Default for unknown sizes - be conservative
        size_category = "unknown"
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
            f"Consider using --short-thinking-threshold {model_size_info['default_threshold']} for {model_size_info['size_category']} models"
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
                "or specify a custom responses file with --responses-file"
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
            f"Very few short thinking examples found. Consider increasing --short-thinking-threshold"
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
                "or specify a custom directions file with --directions-file"
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
    # Use actual head_dim from config, not calculated from hidden_size
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // num_heads)

    # Calculate actual attention dimension (may differ from hidden_size)
    actual_attn_dim = num_heads * head_dim

    logger.info(f"Model architecture analysis:")
    logger.info(f"  Layers: {num_layers}")
    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Attention heads: {num_heads}")
    logger.info(
        f"  Head dimension: {head_dim} ({'from config' if hasattr(model.config, 'head_dim') else 'calculated'})"
    )
    logger.info(f"  Total attention dimension: {actual_attn_dim}")
    logger.info(f"  Size category: {model_size_info['size_category']}")

    # Debug: Check actual o_proj dimensions
    sample_layer = model.model.layers[0].self_attn.o_proj
    o_proj_shape = sample_layer.weight.shape
    logger.info(f"  Sample o_proj weight shape: {o_proj_shape}")

    # Validate architecture expectations
    expected_o_proj = (hidden_size, actual_attn_dim)
    if o_proj_shape == expected_o_proj:
        logger.info(f"  ✓ o_proj shape matches expected {expected_o_proj}")
    else:
        logger.warning(
            f"  ⚠ o_proj shape {o_proj_shape} differs from expected {expected_o_proj}"
        )
        logger.info(f"  → Will attempt adaptive processing")

    # Check for potential issues
    if actual_attn_dim == hidden_size:
        logger.info(f"  → Standard architecture: attention_dim == hidden_size")
    elif actual_attn_dim > hidden_size:
        logger.info(
            f"  → Compressed architecture: attention_dim ({actual_attn_dim}) > hidden_size ({hidden_size})"
        )
    else:
        logger.info(
            f"  → Expanded architecture: attention_dim ({actual_attn_dim}) < hidden_size ({hidden_size})"
        )

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

                if test_shape[-1] == actual_attn_dim:
                    logger.info(
                        "✓ Standard architecture detected - using direct processing"
                    )
                    return "standard"
                elif test_shape[-1] == hidden_size:
                    logger.info(
                        f"✓ Legacy architecture detected - using legacy processing"
                    )
                    return "legacy"
                elif test_shape[-1] > actual_attn_dim:
                    logger.info(
                        f"✓ Extended architecture detected - using salvage processing"
                    )
                    logger.info(
                        f"  Will use first {actual_attn_dim} of {test_shape[-1]} dimensions"
                    )
                    return "extended"
                else:
                    logger.warning(
                        f"⚠ Unexpected architecture: attention dim {test_shape[-1]} < expected {actual_attn_dim}"
                    )
                    return "unexpected"

        except Exception as e:
            logger.warning(f"Architecture validation failed: {e}")
            return "unknown"

    architecture_type = validate_architecture()

    # Set up hooks exactly like ThinkEdit but adapted for Qwen3 architecture
    attn_contribution = []

    # Track processing statistics
    processing_stats = {"standard_processing": 0, "fallback_processing": 0, "errors": 0}

    def capture_attn_contribution_hook():
        def hook_fn(module, input, output):
            # Qwen3 attention: input to o_proj has shape [batch, seq_len, actual_attn_dim]
            attn_out = input[0].detach()[0, :, :]  # [seq_len, actual_attn_dim]

            # Check if this matches our expected attention dimension
            if attn_out.size(-1) == actual_attn_dim:
                # Expected case: reshape using actual head dimensions
                attn_out = attn_out.reshape(attn_out.size(0), num_heads, head_dim)

                # Get o_proj weight: should be [hidden_size, actual_attn_dim]
                o_proj = (
                    module.weight.detach().clone()
                )  # [hidden_size, actual_attn_dim]

                # Reshape o_proj to work with individual heads
                # o_proj maps from [num_heads, head_dim] -> [hidden_size]
                o_proj = (
                    o_proj.reshape(hidden_size, num_heads, head_dim)
                    .permute(1, 2, 0)
                    .contiguous()  # [num_heads, head_dim, hidden_size]
                )

                # Calculate contribution: [seq_len, num_heads, head_dim] @ [num_heads, head_dim, hidden_size]
                # -> [seq_len, num_heads, hidden_size]
                attn_contribution.append(torch.einsum("snk,nkh->snh", attn_out, o_proj))
                processing_stats["standard_processing"] += 1

            elif attn_out.size(-1) == hidden_size:
                # Legacy case: some models might have attn_out = hidden_size
                # Use calculated head_dim = hidden_size // num_heads
                legacy_head_dim = hidden_size // num_heads
                attn_out = attn_out.reshape(
                    attn_out.size(0), num_heads, legacy_head_dim
                )

                # o_proj should be square matrix
                o_proj = module.weight.detach().clone()  # [hidden_size, hidden_size]
                o_proj = (
                    o_proj.reshape(hidden_size, num_heads, legacy_head_dim)
                    .permute(1, 2, 0)
                    .contiguous()
                )
                attn_contribution.append(torch.einsum("snk,nkh->snh", attn_out, o_proj))
                processing_stats["fallback_processing"] += 1

            else:
                # Unexpected case
                actual_dim = attn_out.size(-1)
                logger.warning(
                    f"Unexpected attention output shape: {attn_out.shape}, expected [..., {actual_attn_dim}] or [..., {hidden_size}]"
                )

                # Try to salvage by using whatever dimensions we have
                if actual_dim >= actual_attn_dim:
                    # Take the expected number of dimensions
                    attn_out = attn_out[:, :actual_attn_dim]
                    attn_out = attn_out.reshape(attn_out.size(0), num_heads, head_dim)

                    # Get corresponding part of o_proj
                    o_proj = module.weight.detach().clone()
                    if o_proj.size(1) >= actual_attn_dim:
                        o_proj = o_proj[:, :actual_attn_dim]

                    o_proj = (
                        o_proj.reshape(hidden_size, num_heads, head_dim)
                        .permute(1, 2, 0)
                        .contiguous()
                    )
                    attn_contribution.append(
                        torch.einsum("snk,nkh->snh", attn_out, o_proj)
                    )
                    processing_stats["fallback_processing"] += 1
                else:
                    # Can't salvage - use zero tensor
                    logger.error(
                        f"Cannot process attention dimension {actual_dim}, too small"
                    )
                    zero_contrib = torch.zeros(
                        attn_out.size(0), num_heads, hidden_size, device=attn_out.device
                    )
                    attn_contribution.append(zero_contrib)
                    processing_stats["errors"] += 1

        return hook_fn

    # Register hooks exactly like ThinkEdit
    for layer in model.model.layers:
        layer.self_attn.o_proj.register_forward_hook(capture_attn_contribution_hook())

    # Analyze short thinking examples - exactly like ThinkEdit
    logger.info("Analyzing attention head contributions...")
    avg_contribution = np.zeros((num_layers, num_heads))

    for i, example in enumerate(short_thinking_examples):
        if i % 5 == 0:  # More frequent updates
            logger.info(f"Processing example {i+1}/{len(short_thinking_examples)}")
            if i > 0:  # Show stats after first few examples
                total_processed = sum(processing_stats.values())
                logger.info(
                    f"  Hook stats so far: {processing_stats['fallback_processing']} fallback, {processing_stats['standard_processing']} standard, {processing_stats['errors']} errors"
                )

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

    # Log processing statistics
    total_hooks_called = sum(processing_stats.values())
    logger.info(f"Hook processing statistics:")
    logger.info(f"  Total hook calls: {total_hooks_called}")
    logger.info(
        f"  Standard processing (attn_dim={actual_attn_dim}): {processing_stats['standard_processing']}"
    )
    logger.info(
        f"  Legacy processing (attn_dim={hidden_size}): {processing_stats['fallback_processing']}"
    )
    logger.info(f"  Errors: {processing_stats['errors']}")
    logger.info(f"  Expected total: {len(short_thinking_examples) * num_layers}")

    # Validate processing worked correctly
    if total_hooks_called != len(short_thinking_examples) * num_layers:
        logger.warning(
            f"Hook call count mismatch! Expected {len(short_thinking_examples) * num_layers}, got {total_hooks_called}"
        )

    if processing_stats["errors"] > 0:
        error_rate = processing_stats["errors"] / total_hooks_called * 100
        logger.warning(f"Error rate: {error_rate:.1f}% - results may be incomplete")

    # Architecture compatibility summary
    if processing_stats["standard_processing"] > 0:
        logger.info(f"  → Architecture: Standard (attention_dim={actual_attn_dim})")
    elif processing_stats["fallback_processing"] > 0:
        logger.info(f"  → Architecture: Legacy fallback (attention_dim={hidden_size})")
    else:
        logger.warning(f"  → Architecture: Unknown or problematic")

    # Determine layer range for visualization
    layer_start = max(0, args.layer_start)
    layer_end = num_layers if args.layer_end is None else min(num_layers, args.layer_end)

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
        "model_size_category": model_size_info["size_category"],
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
            "actual_attn_dim": actual_attn_dim,
            "head_dim_source": (
                "config" if hasattr(model.config, "head_dim") else "calculated"
            ),
            "architecture_type": (
                "standard"
                if actual_attn_dim == hidden_size
                else "compressed" if actual_attn_dim > hidden_size else "expanded"
            ),
            "o_proj_shape": list(sample_layer.weight.shape),
        },
        "processing_stats": processing_stats,
        "architecture_validation": {
            "expected_o_proj_shape": list(expected_o_proj),
            "actual_o_proj_shape": list(o_proj_shape),
            "shapes_match": o_proj_shape == expected_o_proj,
            "architecture_detected": architecture_type,
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
