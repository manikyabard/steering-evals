#!/usr/bin/env python3
"""
Evaluate and compare ThinkEdit models against original Qwen3 models.

Usage:
    python evaluate_thinkedit_qwen3.py --original-model Qwen/Qwen3-0.6B --edited-model thinkedit_models/ThinkEdit-Qwen_Qwen3_0.6B
"""

import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import concurrent.futures
from logging_setup import setup_logging, get_logger
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ThinkEdit Qwen3 models")
    parser.add_argument(
        "--original-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Original model name or path",
    )
    parser.add_argument(
        "--edited-model",
        type=str,
        default=None,
        help="Edited model path (auto-detected if not provided)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math", "custom"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32768,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Temperature for sampling"
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k for sampling")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="thinkedit_evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use-sglang",
        action="store_true",
        help="Use SGLang server for evaluation (faster)",
    )
    parser.add_argument(
        "--sglang-port", type=int, default=30000, help="SGLang server port"
    )
    return parser.parse_args()


def extract_thinking_and_response(text):
    """Extract thinking content and final response from model output."""
    # Look for <think>...</think> pattern
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, text, re.DOTALL)

    if think_match:
        thinking = think_match.group(1).strip()
        # Remove thinking part to get response
        response = re.sub(think_pattern, "", text, flags=re.DOTALL).strip()
    else:
        thinking = ""
        response = text.strip()

    return thinking, response


def calculate_thinking_length(thinking_text, tokenizer):
    """Calculate thinking length in tokens."""
    if not thinking_text:
        return 0
    try:
        tokens = tokenizer.encode(thinking_text)
        return len(tokens)
    except:
        return 0


def extract_answer_from_response(response):
    """Extract numerical answer from model response."""
    # Look for boxed answer
    boxed_pattern = r"\\boxed{([^}]+)}"
    boxed_match = re.search(boxed_pattern, response)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Look for #### pattern (GSM8K style)
    hash_pattern = r"####\s*([^\n]+)"
    hash_match = re.search(hash_pattern, response)
    if hash_match:
        return hash_match.group(1).strip()

    # Look for final number in response
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", response)
    if numbers:
        return numbers[-1]

    return None


def check_answer_correctness(predicted, ground_truth):
    """Check if predicted answer matches ground truth."""
    if predicted is None:
        return False

    try:
        # Extract numerical values
        pred_num = float(re.sub(r"[^\d.]", "", str(predicted)))
        gt_num = float(re.sub(r"[^\d.]", "", str(ground_truth)))
        return abs(pred_num - gt_num) < 1e-6
    except:
        # Fallback to string comparison
        return str(predicted).strip().lower() == str(ground_truth).strip().lower()


def load_evaluation_dataset(dataset_name, num_samples):
    """Load evaluation dataset."""
    if dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")["test"]
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        return dataset
    elif dataset_name == "math":
        dataset = load_dataset("hendrycks/competition_math")["test"]
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        return dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def generate_with_model(model, tokenizer, prompt, args):
    """Generate response using local model."""
    # Format prompt for thinking
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
    )
    return response


def generate_with_sglang(prompt, model_name, args):
    """Generate response using SGLang server."""
    url = f"http://localhost:{args.sglang_port}/v1/chat/completions"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"SGLang generation failed: {e}")
        return ""


def evaluate_model(model_path, dataset, args, model_name_suffix=""):
    """Evaluate a single model on the dataset."""
    logger = get_logger(__name__)
    logger.info(f"Evaluating model: {model_path}")

    results = []

    if args.use_sglang:
        # Use SGLang for faster evaluation
        for i, example in enumerate(
            tqdm(dataset, desc=f"Evaluating {model_name_suffix}")
        ):
            if i >= args.num_samples:
                break

            prompt = example["question"]
            ground_truth = example.get("answer", "")

            response = generate_with_sglang(prompt, model_path, args)
            thinking, final_response = extract_thinking_and_response(response)

            predicted_answer = extract_answer_from_response(final_response)
            is_correct = check_answer_correctness(predicted_answer, ground_truth)

            results.append(
                {
                    "question": prompt,
                    "ground_truth": ground_truth,
                    "full_response": response,
                    "thinking": thinking,
                    "final_response": final_response,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "thinking_length": len(thinking.split()) if thinking else 0,
                }
            )
    else:
        # Load model locally
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = (
            AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
            .to(device)
            .eval()
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        tokenizer.pad_token = tokenizer.eos_token

        for i, example in enumerate(
            tqdm(dataset, desc=f"Evaluating {model_name_suffix}")
        ):
            if i >= args.num_samples:
                break

            prompt = example["question"]
            ground_truth = example.get("answer", "")

            response = generate_with_model(model, tokenizer, prompt, args)
            thinking, final_response = extract_thinking_and_response(response)

            predicted_answer = extract_answer_from_response(final_response)
            is_correct = check_answer_correctness(predicted_answer, ground_truth)
            thinking_length_tokens = calculate_thinking_length(thinking, tokenizer)

            results.append(
                {
                    "question": prompt,
                    "ground_truth": ground_truth,
                    "full_response": response,
                    "thinking": thinking,
                    "final_response": final_response,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "thinking_length": len(thinking.split()) if thinking else 0,
                    "thinking_length_tokens": thinking_length_tokens,
                }
            )

        # Clean up
        del model
        torch.cuda.empty_cache()

    return results


def analyze_results(original_results, edited_results, output_dir):
    """Analyze and compare results between original and edited models."""
    logger = get_logger(__name__)

    # Calculate metrics
    original_accuracy = np.mean([r["is_correct"] for r in original_results])
    edited_accuracy = np.mean([r["is_correct"] for r in edited_results])

    original_thinking_lengths = [r["thinking_length"] for r in original_results]
    edited_thinking_lengths = [r["thinking_length"] for r in edited_results]

    original_avg_thinking = np.mean(original_thinking_lengths)
    edited_avg_thinking = np.mean(edited_thinking_lengths)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy comparison
    axes[0, 0].bar(
        ["Original", "ThinkEdit"],
        [original_accuracy, edited_accuracy],
        color=["skyblue", "lightcoral"],
    )
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Model Accuracy Comparison")
    axes[0, 0].set_ylim(0, 1)

    # Add accuracy values on bars
    for i, v in enumerate([original_accuracy, edited_accuracy]):
        axes[0, 0].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

    # Thinking length distribution
    axes[0, 1].hist(
        original_thinking_lengths, bins=30, alpha=0.7, label="Original", color="skyblue"
    )
    axes[0, 1].hist(
        edited_thinking_lengths,
        bins=30,
        alpha=0.7,
        label="ThinkEdit",
        color="lightcoral",
    )
    axes[0, 1].set_xlabel("Thinking Length (words)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Thinking Length Distribution")
    axes[0, 1].legend()

    # Average thinking length comparison
    axes[1, 0].bar(
        ["Original", "ThinkEdit"],
        [original_avg_thinking, edited_avg_thinking],
        color=["skyblue", "lightcoral"],
    )
    axes[1, 0].set_ylabel("Average Thinking Length (words)")
    axes[1, 0].set_title("Average Thinking Length Comparison")

    # Add values on bars
    for i, v in enumerate([original_avg_thinking, edited_avg_thinking]):
        axes[1, 0].text(
            i,
            v + max(original_avg_thinking, edited_avg_thinking) * 0.01,
            f"{v:.1f}",
            ha="center",
            va="bottom",
        )

    # Thinking length vs accuracy scatter
    original_correct = [
        r["thinking_length"] for r in original_results if r["is_correct"]
    ]
    original_incorrect = [
        r["thinking_length"] for r in original_results if not r["is_correct"]
    ]
    edited_correct = [r["thinking_length"] for r in edited_results if r["is_correct"]]
    edited_incorrect = [
        r["thinking_length"] for r in edited_results if not r["is_correct"]
    ]

    axes[1, 1].scatter(
        original_correct,
        [1] * len(original_correct),
        alpha=0.6,
        color="blue",
        label="Original Correct",
        s=20,
    )
    axes[1, 1].scatter(
        original_incorrect,
        [0.9] * len(original_incorrect),
        alpha=0.6,
        color="lightblue",
        label="Original Incorrect",
        s=20,
    )
    axes[1, 1].scatter(
        edited_correct,
        [0.6] * len(edited_correct),
        alpha=0.6,
        color="red",
        label="ThinkEdit Correct",
        s=20,
    )
    axes[1, 1].scatter(
        edited_incorrect,
        [0.5] * len(edited_incorrect),
        alpha=0.6,
        color="lightcoral",
        label="ThinkEdit Incorrect",
        s=20,
    )

    axes[1, 1].set_xlabel("Thinking Length (words)")
    axes[1, 1].set_ylabel("Model & Correctness")
    axes[1, 1].set_title("Thinking Length vs Correctness")
    axes[1, 1].set_yticks([0.5, 0.6, 0.9, 1.0])
    axes[1, 1].set_yticklabels(["Edit-Wrong", "Edit-Right", "Orig-Wrong", "Orig-Right"])
    axes[1, 1].legend()

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "evaluation_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    # Summary statistics
    summary = {
        "original_accuracy": original_accuracy,
        "edited_accuracy": edited_accuracy,
        "accuracy_change": edited_accuracy - original_accuracy,
        "original_avg_thinking_length": original_avg_thinking,
        "edited_avg_thinking_length": edited_avg_thinking,
        "thinking_length_change": edited_avg_thinking - original_avg_thinking,
        "thinking_length_reduction_percentage": (
            (
                (original_avg_thinking - edited_avg_thinking)
                / original_avg_thinking
                * 100
            )
            if original_avg_thinking > 0
            else 0
        ),
        "num_samples": len(original_results),
    }

    logger.info(f"Evaluation Summary:")
    logger.info(f"  Original Accuracy: {original_accuracy:.3f}")
    logger.info(f"  ThinkEdit Accuracy: {edited_accuracy:.3f}")
    logger.info(f"  Accuracy Change: {summary['accuracy_change']:+.3f}")
    logger.info(f"  Original Avg Thinking Length: {original_avg_thinking:.1f}")
    logger.info(f"  ThinkEdit Avg Thinking Length: {edited_avg_thinking:.1f}")
    logger.info(
        f"  Thinking Length Reduction: {summary['thinking_length_reduction_percentage']:.1f}%"
    )

    return summary, plot_file


def main():
    args = parse_args()

    # Set up logging
    logger = get_logger(__name__)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect edited model if not provided
    if args.edited_model is None:
        model_name = args.original_model.replace("/", "_").replace("-", "_")
        args.edited_model = f"thinkedit_models/ThinkEdit-{model_name}"

    if not os.path.exists(args.edited_model):
        logger.error(f"Edited model not found: {args.edited_model}")
        logger.info("Please run get_thinkedit_qwen3_models.py first")
        return

    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    dataset = load_evaluation_dataset(args.dataset, args.num_samples)
    logger.info(f"Loaded {len(dataset)} examples")

    # Evaluate original model
    logger.info("Evaluating original model...")
    original_results = evaluate_model(args.original_model, dataset, args, "Original")

    # Evaluate edited model
    logger.info("Evaluating ThinkEdit model...")
    edited_results = evaluate_model(args.edited_model, dataset, args, "ThinkEdit")

    # Save detailed results
    results_data = {
        "original_model": args.original_model,
        "edited_model": args.edited_model,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "original_results": original_results,
        "edited_results": edited_results,
    }

    results_file = os.path.join(args.output_dir, "detailed_results.json")
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"Detailed results saved to: {results_file}")

    # Analyze and compare results
    summary, plot_file = analyze_results(
        original_results, edited_results, args.output_dir
    )

    # Save summary
    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")

    # Print final summary
    print("\n" + "=" * 80)
    print("THINKEDIT EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Original Model: {args.original_model}")
    print(f"ThinkEdit Model: {args.edited_model}")
    print("-" * 80)
    print(f"Original Accuracy:     {summary['original_accuracy']:.3f}")
    print(f"ThinkEdit Accuracy:    {summary['edited_accuracy']:.3f}")
    print(f"Accuracy Change:       {summary['accuracy_change']:+.3f}")
    print("-" * 80)
    print(f"Original Thinking Len: {summary['original_avg_thinking_length']:.1f} words")
    print(f"ThinkEdit Thinking Len:{summary['edited_avg_thinking_length']:.1f} words")
    print(
        f"Reduction:             {summary['thinking_length_reduction_percentage']:.1f}%"
    )
    print("-" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
