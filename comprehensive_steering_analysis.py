#!/usr/bin/env python3
"""
Comprehensive Steering Analysis

This script tests a wider range of alpha values to understand the relationship
between steering magnitude and reasoning length.
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from logging_setup import setup_logging, get_logger


def count_thinking_length(response):
    """Count words in thinking text."""
    if isinstance(response, dict) and "thinking" in response:
        thinking_text = response["thinking"]
    else:
        thinking_text = str(response)

    if not thinking_text or thinking_text.strip() == "":
        return 0

    # Split by whitespace and count non-empty tokens
    words = thinking_text.strip().split()
    meaningful_words = [w for w in words if len(w.strip()) > 0]
    return len(meaningful_words)


def apply_steering_layers(model, directions, alpha=0.0, component="attn"):
    """Apply steering layers to the model - simplified version."""
    from steer_reasoning_length import SteeringAttentionLayer, SteeringMLPLayer

    steering_layers = []

    # Apply steering to attention layers
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

    return steering_layers


def remove_steering_layers(steering_layers):
    """Remove steering layers from the model."""
    for layer in steering_layers:
        layer.restore_original()


def generate_with_steering(
    model, tokenizer, question, alpha=0.0, directions=None, component="attn"
):
    """Generate a response with steering applied."""
    if directions is None:
        raise ValueError("Directions must be provided for steering")

    # Create the prompt
    prompt = f"Solve this math problem step by step, and put your final answer within \\boxed{{}}:\n{question}"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Apply steering
    steering_layers = apply_steering_layers(model, directions, alpha, component)

    # Generate with steering applied
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
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
            # Extract thinking content
            thinking_content = tokenizer.decode(
                output_ids[:think_end_index], skip_special_tokens=True
            ).strip()
            if thinking_content.startswith("<think>"):
                thinking_content = thinking_content[len("<think>") :].strip()

            content = tokenizer.decode(
                output_ids[think_end_index + 1 :], skip_special_tokens=True
            ).strip()
            return {"thinking": thinking_content, "response": content}
    except ValueError:
        pass

    # If no thinking token found, return everything as response
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return {"thinking": "", "response": content}


def test_alpha_range():
    """Test a comprehensive range of alpha values."""

    # Comprehensive alpha values to test
    alpha_values = [
        -0.12,
        -0.10,
        -0.08,
        -0.06,
        -0.04,
        -0.02,
        0.0,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
    ]

    # Test question from GSM8K
    test_question = "Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, calculate Allen's age 10 years from now."

    # Setup
    model_name = "Qwen/Qwen3-0.6B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    direction_file = "directions/Qwen3-0.6B_reasoning_length_direction_gsm8k_attn.pt"

    logger = get_logger()
    logger.info("Loading model and directions...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map=device
    )

    if not os.path.exists(direction_file):
        logger.error(f"Direction file not found: {direction_file}")
        logger.error("Please run direction extraction first.")
        return []

    directions = torch.load(direction_file, map_location=device)
    logger.info(f"Loaded directions for {len(directions)} layers")

    # Results storage
    results = []

    print(f"\nTesting {len(alpha_values)} alpha values...")

    for alpha in tqdm(alpha_values, desc="Testing alphas"):
        try:
            # Generate response with steering
            response = generate_with_steering(
                model,
                tokenizer,
                test_question,
                alpha=alpha,
                directions=directions,
                component="attn",
            )

            # Count thinking words
            thinking_words = count_thinking_length(response)

            # Store result
            result = {
                "alpha": alpha,
                "thinking_words": thinking_words,
                "full_response": response,
            }
            results.append(result)

            print(f"α = {alpha:5.2f}: {thinking_words:4d} words")

        except Exception as e:
            print(f"Error with alpha {alpha}: {e}")
            continue

        # Cleanup CUDA memory
        torch.cuda.empty_cache()

    return results


def analyze_results(results):
    """Analyze the comprehensive results."""

    alphas = [r["alpha"] for r in results]
    word_counts = [r["thinking_words"] for r in results]

    print("\n" + "=" * 60)
    print("COMPREHENSIVE STEERING ANALYSIS RESULTS")
    print("=" * 60)

    # Print table
    print("\n| Alpha  | Thinking Words | Change from Baseline |")
    print("|--------|----------------|---------------------|")

    baseline_idx = next(i for i, r in enumerate(results) if r["alpha"] == 0.0)
    baseline_words = results[baseline_idx]["thinking_words"]

    for result in results:
        alpha = result["alpha"]
        words = result["thinking_words"]
        change = words - baseline_words
        change_sign = "+" if change > 0 else ""
        print(f"| {alpha:6.2f} | {words:13d} | {change_sign}{change:18d} |")

    # Statistical analysis
    print(f"\nStatistical Analysis:")
    print(f"Baseline (α=0.0): {baseline_words} words")
    print(f"Range: {min(word_counts)} - {max(word_counts)} words")
    print(f"Standard deviation: {np.std(word_counts):.1f} words")

    # Find patterns
    negative_alphas = [
        (r["alpha"], r["thinking_words"]) for r in results if r["alpha"] < 0
    ]
    positive_alphas = [
        (r["alpha"], r["thinking_words"]) for r in results if r["alpha"] > 0
    ]

    print(f"\nPattern Analysis:")
    print(
        f"Negative α range: {min(w for _, w in negative_alphas)} - {max(w for _, w in negative_alphas)} words"
    )
    print(
        f"Positive α range: {min(w for _, w in positive_alphas)} - {max(w for _, w in positive_alphas)} words"
    )

    # Check for U-shape or other patterns
    abs_alphas = [abs(alpha) for alpha in alphas]

    # Correlation between |alpha| and word count
    correlation = np.corrcoef(abs_alphas, word_counts)[0, 1]
    print(f"Correlation between |α| and word count: {correlation:.3f}")

    return alphas, word_counts


def plot_results(
    alphas, word_counts, output_path="comprehensive_steering_analysis.png"
):
    """Create visualizations of the results."""

    plt.figure(figsize=(14, 10))

    # Main plot: Alpha vs Word Count
    plt.subplot(2, 2, 1)
    plt.plot(alphas, word_counts, "o-", linewidth=2, markersize=8, color="blue")
    plt.axhline(
        y=word_counts[alphas.index(0.0)],
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Baseline",
    )
    plt.xlabel("Alpha Value")
    plt.ylabel("Thinking Words")
    plt.title("Alpha vs Thinking Length")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot: |Alpha| vs Word Count
    plt.subplot(2, 2, 2)
    abs_alphas = [abs(alpha) for alpha in alphas]
    plt.plot(abs_alphas, word_counts, "o-", linewidth=2, markersize=8, color="green")
    plt.xlabel("|Alpha| (Absolute Value)")
    plt.ylabel("Thinking Words")
    plt.title("|Alpha| vs Thinking Length")
    plt.grid(True, alpha=0.3)

    # Plot: Positive vs Negative Alpha comparison
    plt.subplot(2, 2, 3)
    neg_alphas = [
        (alpha, words) for alpha, words in zip(alphas, word_counts) if alpha < 0
    ]
    pos_alphas = [
        (alpha, words) for alpha, words in zip(alphas, word_counts) if alpha > 0
    ]
    baseline_words = word_counts[alphas.index(0.0)]

    if neg_alphas:
        neg_abs, neg_words = zip(*[(abs(alpha), words) for alpha, words in neg_alphas])
        plt.plot(
            neg_abs,
            neg_words,
            "o-",
            linewidth=2,
            markersize=6,
            color="red",
            label="Negative α",
        )

    if pos_alphas:
        pos_abs, pos_words = zip(*[(alpha, words) for alpha, words in pos_alphas])
        plt.plot(
            pos_abs,
            pos_words,
            "o-",
            linewidth=2,
            markersize=6,
            color="blue",
            label="Positive α",
        )

    plt.axhline(
        y=baseline_words, color="black", linestyle="--", alpha=0.7, label="Baseline"
    )
    plt.xlabel("|Alpha|")
    plt.ylabel("Thinking Words")
    plt.title("Positive vs Negative Alpha Effects")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot: Change from baseline
    plt.subplot(2, 2, 4)
    baseline_words = word_counts[alphas.index(0.0)]
    changes = [words - baseline_words for words in word_counts]
    colors = [
        "red" if alpha < 0 else "blue" if alpha > 0 else "black" for alpha in alphas
    ]
    plt.bar(range(len(alphas)), changes, color=colors, alpha=0.7)
    plt.xlabel("Alpha Index")
    plt.ylabel("Change from Baseline (words)")
    plt.title("Change from Baseline by Alpha")
    plt.xticks(range(len(alphas)), [f"{alpha:.2f}" for alpha in alphas], rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")
    plt.show()


def save_results(results, output_path="comprehensive_steering_results.json"):
    """Save results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    """Run comprehensive steering analysis."""
    # Setup logging
    logger = setup_logging("comprehensive_steering_analysis")

    logger.info("=" * 60)
    logger.info("COMPREHENSIVE STEERING ANALYSIS")
    logger.info("=" * 60)
    logger.info("Testing relationship between alpha values and reasoning length")
    logger.info(
        "This will help us understand if the pattern is U-shaped, linear, or other"
    )

    # Run tests
    results = test_alpha_range()

    if not results:
        logger.error("No results obtained. Check your setup.")
        return

    # Analyze results
    alphas, word_counts = analyze_results(results)

    # Visualize results
    plot_results(alphas, word_counts)

    # Save results
    save_results(results)

    # Interpretation
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info("=" * 60)

    baseline_idx = next(i for i, r in enumerate(results) if r["alpha"] == 0.0)
    baseline_words = results[baseline_idx]["thinking_words"]

    # Check if U-shaped
    extreme_alphas = [r for r in results if abs(r["alpha"]) >= 0.08]
    moderate_alphas = [r for r in results if 0.02 <= abs(r["alpha"]) <= 0.06]

    if extreme_alphas and moderate_alphas:
        extreme_avg = np.mean([r["thinking_words"] for r in extreme_alphas])
        moderate_avg = np.mean([r["thinking_words"] for r in moderate_alphas])

        print(f"Extreme |α| ≥ 0.08: {extreme_avg:.0f} words (average)")
        print(f"Moderate 0.02 ≤ |α| ≤ 0.06: {moderate_avg:.0f} words (average)")
        print(f"Baseline α = 0.0: {baseline_words} words")

        if extreme_avg > moderate_avg and extreme_avg > baseline_words:
            print("\n✓ CONFIRMED: U-shaped relationship")
            print("  - Large |α| values increase reasoning length")
            print("  - Small |α| values decrease reasoning length")
            print("  - This suggests magnitude-based reasoning activation")
        else:
            print("\n? Complex relationship detected")
            print("  - Pattern is not simply U-shaped")
            print("  - Further investigation needed")


if __name__ == "__main__":
    main()
