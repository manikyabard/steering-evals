#!/usr/bin/env python3
"""
Create ThinkEdit models for Qwen3 by editing attention head weights.

This script performs weight editing on identified short thinking attention heads
to mitigate overly short reasoning in Qwen3 models.

Usage:
    python get_thinkedit_qwen3_models.py --model Qwen/Qwen3-0.6B
    python get_thinkedit_qwen3_models.py --model Qwen/Qwen3-0.6B --directions_file custom_directions.pt

Prerequisites:
    1. Run find_short_thinking_attn_heads_qwen3.py to identify target heads
    2. Ensure thinking length directions are available
"""

import os
import argparse
import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from logging_setup import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Create ThinkEdit models for Qwen3")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--intervention_weight",
        type=float,
        default=1.0,
        help="Intervention strength (higher = stronger editing)",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="thinkedit_analysis",
        help="Directory containing head analysis results",
    )
    parser.add_argument(
        "--directions_file",
        type=str,
        default=None,
        help="Path to thinking length directions file (auto-detected from model name if not provided)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="thinkedit_models",
        help="Directory to save edited models",
    )
    parser.add_argument(
        "--top_k_heads",
        type=int,
        default=None,
        help="Number of top heads to edit (default: use all identified heads)",
    )
    parser.add_argument(
        "--manual_heads",
        type=str,
        default=None,
        help="Manually specify heads as comma-separated pairs: 'layer1,head1;layer2,head2'",
    )
    parser.add_argument(
        "--save_local",
        action="store_true",
        default=True,
        help="Save model locally (default: True)",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_model_name",
        type=str,
        default=None,
        help="Model name for HuggingFace Hub (auto-generated if not provided)",
    )
    return parser.parse_args()


def remove_projection_along_direction(W_o, thinking_direction, intervention_weight=1.0):
    """
    Remove projection along thinking direction from weight matrix.

    Args:
        W_o: Weight matrix [head_dim, hidden_size]
        thinking_direction: Direction to remove [hidden_size]
        intervention_weight: Strength of intervention

    Returns:
        Modified weight matrix
    """
    # Normalize direction
    v_normalized = thinking_direction / torch.norm(thinking_direction)

    # Calculate projection: outer product of (W_o @ v) and v
    projection = torch.outer(torch.matmul(W_o, v_normalized), v_normalized)

    # Remove projection
    W_o_modified = W_o - intervention_weight * projection

    # Calculate projection magnitudes for logging
    projection_before = torch.norm(torch.matmul(W_o, thinking_direction))
    projection_after = torch.norm(torch.matmul(W_o_modified, thinking_direction))

    return W_o_modified, projection_before.item(), projection_after.item()


def load_head_analysis(analysis_dir, model_name):
    """Load head analysis results."""
    analysis_file = os.path.join(
        analysis_dir, f"{model_name}_short_thinking_heads_analysis.json"
    )

    if not os.path.exists(analysis_file):
        raise FileNotFoundError(
            f"Head analysis file not found: {analysis_file}\n"
            f"Please run find_short_thinking_attn_heads_qwen3.py first"
        )

    with open(analysis_file, "r") as f:
        analysis = json.load(f)

    return analysis


def parse_manual_heads(manual_heads_str):
    """Parse manually specified heads."""
    if not manual_heads_str:
        return []

    heads = []
    pairs = manual_heads_str.split(";")
    for pair in pairs:
        layer, head = map(int, pair.split(","))
        heads.append((layer, head))
    return heads


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

    # Get model configuration
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    hidden_size = model.config.hidden_size

    logger.info(
        f"Model config: {num_heads} heads, {head_dim} head_dim, {hidden_size} hidden_size"
    )

    # Determine heads to edit
    if args.manual_heads:
        heads_to_edit = parse_manual_heads(args.manual_heads)
        logger.info(f"Using manually specified heads: {heads_to_edit}")
    else:
        # Load from analysis
        analysis = load_head_analysis(args.analysis_dir, model_name)
        heads_to_edit = analysis["top_heads"]

        # Limit to top_k if specified
        if args.top_k_heads is not None:
            heads_to_edit = heads_to_edit[: args.top_k_heads]

        logger.info(f"Using {len(heads_to_edit)} heads from analysis")

    if not heads_to_edit:
        logger.error("No heads to edit found!")
        return

    logger.info(f"Heads to edit: {heads_to_edit}")

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
                "Please run extract_reasoning_length_direction_improved.py first, "
                "or specify a custom directions file with --directions_file"
            )
        return

    thinking_length_direction = torch.load(direction_file).to(device)
    thinking_length_direction = thinking_length_direction / torch.norm(
        thinking_length_direction, dim=-1, keepdim=True
    )

    # Use negative direction (we want to reduce short thinking)
    thinking_length_direction = -thinking_length_direction

    # Perform weight editing
    logger.info(
        f"Performing weight editing with intervention strength: {args.intervention_weight}"
    )

    edit_log = []
    total_projection_reduction = 0.0

    for layer_idx, head_idx in heads_to_edit:
        logger.info(f"Editing Layer {layer_idx}, Head {head_idx}")

        # Calculate slice indices for this head
        start_idx = head_idx * head_dim
        end_idx = (head_idx + 1) * head_dim

        # Get current o_proj weight
        o_proj_weight = (
            model.model.layers[layer_idx].self_attn.o_proj.weight.detach().clone()
        )
        W_o = o_proj_weight[:, start_idx:end_idx].T.float()  # [head_dim, hidden_size]

        # Apply intervention
        W_o_modified, proj_before, proj_after = remove_projection_along_direction(
            W_o,
            thinking_length_direction[layer_idx, 0].float(),
            args.intervention_weight,
        )

        # Update model weights
        o_proj_weight[:, start_idx:end_idx] = W_o_modified.T.to(torch.bfloat16)
        model.model.layers[layer_idx].self_attn.o_proj.weight = torch.nn.Parameter(
            o_proj_weight
        )

        # Log the change
        projection_reduction = proj_before - proj_after
        total_projection_reduction += projection_reduction

        edit_info = {
            "layer": layer_idx,
            "head": head_idx,
            "projection_before": proj_before,
            "projection_after": proj_after,
            "projection_reduction": projection_reduction,
            "reduction_percentage": (
                (projection_reduction / proj_before * 100) if proj_before > 0 else 0
            ),
        }
        edit_log.append(edit_info)

        logger.info(
            f"  Projection: {proj_before:.4f} -> {proj_after:.4f} "
            f"(reduction: {projection_reduction:.4f}, {edit_info['reduction_percentage']:.1f}%)"
        )

    logger.info(f"Total projection reduction: {total_projection_reduction:.4f}")

    # Save the edited model
    if args.save_local:
        save_dir = os.path.join(args.output_dir, f"ThinkEdit-{model_name}")
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Saving edited model to: {save_dir}")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Save edit metadata
        metadata = {
            "base_model": args.model,
            "intervention_weight": args.intervention_weight,
            "edited_heads": heads_to_edit,
            "edit_log": edit_log,
            "total_projection_reduction": total_projection_reduction,
            "model_config": {
                "num_heads": num_heads,
                "head_dim": head_dim,
                "hidden_size": hidden_size,
            },
        }

        metadata_file = os.path.join(save_dir, "thinkedit_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Edit metadata saved to: {metadata_file}")

    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        if args.hub_model_name:
            hub_name = args.hub_model_name
        else:
            hub_name = f"ThinkEdit-{model_name}"

        logger.info(f"Pushing model to HuggingFace Hub: {hub_name}")
        try:
            model.push_to_hub(hub_name)
            tokenizer.push_to_hub(hub_name)
            logger.info(f"Successfully pushed to Hub: {hub_name}")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("THINKEDIT MODEL CREATION SUMMARY")
    print("=" * 80)
    print(f"Base model: {args.model}")
    print(f"Intervention strength: {args.intervention_weight}")
    print(f"Heads edited: {len(heads_to_edit)}")
    print(f"Total projection reduction: {total_projection_reduction:.4f}")
    print(f"\nEdited heads:")
    print("-" * 40)
    for edit in edit_log:
        print(
            f"  Layer {edit['layer']:2d}, Head {edit['head']:2d}: "
            f"{edit['projection_reduction']:6.4f} ({edit['reduction_percentage']:5.1f}% reduction)"
        )

    if args.save_local:
        print(f"\nModel saved to: {save_dir}")
    if args.push_to_hub:
        print(f"Model pushed to Hub: {hub_name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
