#!/usr/bin/env python3
"""
Validate ThinkEdit Models

This script compares the original model with its ThinkEdit edited version
to verify that the attention head editing is working as expected.

Usage:
    python validate_thinkedit_models.py --original_model Qwen/Qwen3-0.6B --thinkedit_model thinkedit_models/ThinkEdit-Qwen_Qwen3_0.6B
    python validate_thinkedit_models.py --original_model ../qwen3_4b/ --thinkedit_model thinkedit_models/ThinkEdit-qwen3_4b
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from logging_setup import setup_logging, get_logger
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate ThinkEdit model effectiveness"
    )
    parser.add_argument(
        "--original_model", type=str, required=True, help="Path to original model"
    )
    parser.add_argument(
        "--thinkedit_model",
        type=str,
        required=True,
        help="Path to ThinkEdit edited model",
    )
    parser.add_argument(
        "--num_questions", type=int, default=20, help="Number of questions to test"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation_results",
        help="Directory to save validation results",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4096, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p for nucleus sampling"
    )
    parser.add_argument("--top_k", type=int, default=20, help="Top-k for sampling")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu, mps)",
    )
    parser.add_argument(
        "--save_examples", action="store_true", help="Save detailed example comparisons"
    )
    return parser.parse_args()


def get_device(device_arg):
    """Determine the best device to use."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def load_models_and_tokenizer(original_path, thinkedit_path, device):
    """Load both original and ThinkEdit models."""
    logger = get_logger()

    logger.info(f"Loading original model from: {original_path}")
    original_model = AutoModelForCausalLM.from_pretrained(
        original_path, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    logger.info(f"Loading ThinkEdit model from: {thinkedit_path}")
    thinkedit_model = AutoModelForCausalLM.from_pretrained(
        thinkedit_path, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    # Use tokenizer from original model (should be the same)
    tokenizer = AutoTokenizer.from_pretrained(original_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Models loaded on device: {device}")

    return original_model, thinkedit_model, tokenizer


def load_thinkedit_metadata(thinkedit_path):
    """Load ThinkEdit metadata if available."""
    metadata_file = os.path.join(thinkedit_path, "thinkedit_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            return json.load(f)
    return None


def generate_response(
    model,
    tokenizer,
    question,
    max_new_tokens=4096,
    temperature=0.7,
    top_p=0.95,
    top_k=20,
):
    """Generate response for a single question."""
    prompt = f"Solve this math problem step by step, and put your final answer within \\boxed{{}}:\n{question}"

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()

    # Parse thinking and response
    try:
        think_end_token = tokenizer.encode("</think>", add_special_tokens=False)[-1]
        think_end_index = (
            output_ids.index(think_end_token) if think_end_token in output_ids else -1
        )

        if think_end_index != -1:
            thinking_content = tokenizer.decode(
                output_ids[:think_end_index], skip_special_tokens=True
            ).strip()
            if thinking_content.startswith("<think>"):
                thinking_content = thinking_content[len("<think>") :].strip()

            response_content = tokenizer.decode(
                output_ids[think_end_index + 1 :], skip_special_tokens=True
            ).strip()

            return {"thinking": thinking_content, "response": response_content}
    except ValueError:
        pass

    # If no thinking tags found, treat everything as response
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return {"thinking": "", "response": content}


def calculate_thinking_metrics(thinking_text):
    """Calculate metrics for thinking content."""
    if not thinking_text:
        return {"word_count": 0, "char_count": 0, "line_count": 0, "reasoning_steps": 0}

    words = thinking_text.split()
    lines = thinking_text.split("\n")

    # Count reasoning steps (simple heuristic)
    reasoning_indicators = [
        "first",
        "second",
        "third",
        "next",
        "then",
        "so",
        "therefore",
        "step",
        "now",
        "let's",
        "we need",
        "i need",
        "because",
    ]
    reasoning_steps = sum(1 for word in words if word.lower() in reasoning_indicators)

    return {
        "word_count": len(words),
        "char_count": len(thinking_text),
        "line_count": len([line for line in lines if line.strip()]),
        "reasoning_steps": reasoning_steps,
    }


def extract_final_answer(response_text):
    """Extract final answer from response."""
    # Look for boxed answer
    boxed_pattern = r"\\boxed\{([^}]*)\}"
    match = re.search(boxed_pattern, response_text)
    if match:
        return match.group(1).strip()

    # Look for other answer patterns
    answer_patterns = [
        r"(?:final answer|answer|result|solution) is:?\s*([^\n.]+)",
        r"= ([0-9]+(?:\.[0-9]+)?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*$",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return "Not found"


def is_correct_answer(predicted, expected):
    """Check if predicted answer matches expected answer."""
    # Extract numerical value from GSM8K answer format
    if "####" in expected:
        expected = expected.split("####")[1].strip()

    # Simple numerical comparison
    try:
        pred_num = float(re.sub(r"[^\d.-]", "", predicted))
        exp_num = float(re.sub(r"[^\d.-]", "", expected))
        return abs(pred_num - exp_num) < 1e-6
    except:
        return predicted.lower().strip() == expected.lower().strip()


def compare_models(original_model, thinkedit_model, tokenizer, test_questions, args):
    """Compare original and ThinkEdit models on test questions."""
    logger = get_logger()
    results = []

    logger.info(f"Comparing models on {len(test_questions)} questions...")

    for i, example in enumerate(tqdm(test_questions, desc="Generating responses")):
        question = example["question"]
        expected_answer = example["answer"]

        logger.info(f"Processing question {i+1}/{len(test_questions)}")

        # Generate responses from both models
        torch.manual_seed(args.seed + i)  # Consistent seed per question

        original_response = generate_response(
            original_model,
            tokenizer,
            question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        torch.manual_seed(args.seed + i)  # Same seed for fair comparison

        thinkedit_response = generate_response(
            thinkedit_model,
            tokenizer,
            question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        # Calculate metrics
        original_metrics = calculate_thinking_metrics(original_response["thinking"])
        thinkedit_metrics = calculate_thinking_metrics(thinkedit_response["thinking"])

        # Extract and check answers
        original_answer = extract_final_answer(original_response["response"])
        thinkedit_answer = extract_final_answer(thinkedit_response["response"])

        original_correct = is_correct_answer(original_answer, expected_answer)
        thinkedit_correct = is_correct_answer(thinkedit_answer, expected_answer)

        result = {
            "question_id": i,
            "question": question,
            "expected_answer": expected_answer,
            "original": {
                "thinking": original_response["thinking"],
                "response": original_response["response"],
                "metrics": original_metrics,
                "extracted_answer": original_answer,
                "correct": original_correct,
            },
            "thinkedit": {
                "thinking": thinkedit_response["thinking"],
                "response": thinkedit_response["response"],
                "metrics": thinkedit_metrics,
                "extracted_answer": thinkedit_answer,
                "correct": thinkedit_correct,
            },
        }

        results.append(result)

        # Log progress
        orig_words = original_metrics["word_count"]
        edit_words = thinkedit_metrics["word_count"]
        logger.info(
            f"  Original thinking: {orig_words} words, ThinkEdit: {edit_words} words"
        )

        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def analyze_results(results, metadata=None):
    """Analyze comparison results and generate statistics."""
    logger = get_logger()

    original_thinking_words = [r["original"]["metrics"]["word_count"] for r in results]
    thinkedit_thinking_words = [
        r["thinkedit"]["metrics"]["word_count"] for r in results
    ]

    original_accuracy = np.mean([r["original"]["correct"] for r in results])
    thinkedit_accuracy = np.mean([r["thinkedit"]["correct"] for r in results])

    original_avg_words = np.mean(original_thinking_words)
    thinkedit_avg_words = np.mean(thinkedit_thinking_words)

    word_count_change = thinkedit_avg_words - original_avg_words
    word_count_change_pct = (
        (word_count_change / original_avg_words * 100) if original_avg_words > 0 else 0
    )

    # Count examples where ThinkEdit actually increased thinking length
    longer_thinking_count = sum(
        1
        for r in results
        if r["thinkedit"]["metrics"]["word_count"]
        > r["original"]["metrics"]["word_count"]
    )
    longer_thinking_pct = longer_thinking_count / len(results) * 100

    analysis = {
        "summary": {
            "total_questions": len(results),
            "original_avg_thinking_words": original_avg_words,
            "thinkedit_avg_thinking_words": thinkedit_avg_words,
            "word_count_change": word_count_change,
            "word_count_change_percentage": word_count_change_pct,
            "original_accuracy": original_accuracy,
            "thinkedit_accuracy": thinkedit_accuracy,
            "accuracy_change": thinkedit_accuracy - original_accuracy,
            "longer_thinking_examples": longer_thinking_count,
            "longer_thinking_percentage": longer_thinking_pct,
        },
        "distributions": {
            "original_thinking_words": original_thinking_words,
            "thinkedit_thinking_words": thinkedit_thinking_words,
        },
    }

    if metadata:
        analysis["metadata"] = {
            "edited_heads": metadata.get("edited_heads", []),
            "total_projection_reduction": metadata.get("total_projection_reduction", 0),
            "intervention_weight": metadata.get("intervention_weight", 0),
            "num_edited_heads": len(metadata.get("edited_heads", [])),
        }

    # Log key findings
    logger.info("=" * 80)
    logger.info("THINKEDIT VALIDATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Questions tested: {len(results)}")
    logger.info(f"Average thinking length:")
    logger.info(f"  Original model: {original_avg_words:.1f} words")
    logger.info(f"  ThinkEdit model: {thinkedit_avg_words:.1f} words")
    logger.info(
        f"  Change: {word_count_change:+.1f} words ({word_count_change_pct:+.1f}%)"
    )
    logger.info(f"Accuracy:")
    logger.info(f"  Original model: {original_accuracy:.1%}")
    logger.info(f"  ThinkEdit model: {thinkedit_accuracy:.1%}")
    logger.info(f"  Change: {thinkedit_accuracy - original_accuracy:+.1%}")
    logger.info(
        f"Examples with longer thinking in ThinkEdit: {longer_thinking_count}/{len(results)} ({longer_thinking_pct:.1f}%)"
    )

    if metadata:
        logger.info(f"Model modifications:")
        logger.info(f"  Heads edited: {len(metadata.get('edited_heads', []))}")
        logger.info(f"  Intervention weight: {metadata.get('intervention_weight', 0)}")
        logger.info(
            f"  Total projection reduction: {metadata.get('total_projection_reduction', 0):.4f}"
        )

    return analysis


def create_visualizations(analysis, output_dir):
    """Create visualization plots."""
    orig_words = analysis["distributions"]["original_thinking_words"]
    edit_words = analysis["distributions"]["thinkedit_thinking_words"]

    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Histogram comparison
    ax1.hist(orig_words, bins=20, alpha=0.7, label="Original", color="blue")
    ax1.hist(edit_words, bins=20, alpha=0.7, label="ThinkEdit", color="red")
    ax1.set_xlabel("Thinking Length (words)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Thinking Lengths")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot
    ax2.scatter(orig_words, edit_words, alpha=0.6, color="purple")
    ax2.plot(
        [0, max(max(orig_words), max(edit_words))],
        [0, max(max(orig_words), max(edit_words))],
        "k--",
        alpha=0.5,
        label="y=x line",
    )
    ax2.set_xlabel("Original Thinking Length (words)")
    ax2.set_ylabel("ThinkEdit Thinking Length (words)")
    ax2.set_title("Thinking Length Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Box plot comparison
    ax3.boxplot([orig_words, edit_words], labels=["Original", "ThinkEdit"])
    ax3.set_ylabel("Thinking Length (words)")
    ax3.set_title("Thinking Length Distribution Comparison")
    ax3.grid(True, alpha=0.3)

    # Summary statistics
    summary = analysis["summary"]
    ax4.axis("off")

    summary_text = f"""
Summary Statistics:

Questions tested: {summary['total_questions']}

Average thinking length:
• Original: {summary['original_avg_thinking_words']:.1f} words
• ThinkEdit: {summary['thinkedit_avg_thinking_words']:.1f} words
• Change: {summary['word_count_change']:+.1f} words ({summary['word_count_change_percentage']:+.1f}%)

Accuracy:
• Original: {summary['original_accuracy']:.1%}
• ThinkEdit: {summary['thinkedit_accuracy']:.1%}  
• Change: {summary['accuracy_change']:+.1%}

ThinkEdit effectiveness:
• {summary['longer_thinking_examples']}/{summary['total_questions']} examples showed longer thinking
• {summary['longer_thinking_percentage']:.1f}% improvement rate
"""

    if "metadata" in analysis:
        meta = analysis["metadata"]
        summary_text += f"""
Model modifications:
• Heads edited: {meta['num_edited_heads']}
• Intervention weight: {meta['intervention_weight']}
• Projection reduction: {meta['total_projection_reduction']:.4f}
"""

    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(output_dir, "thinkedit_validation.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger = get_logger()
    logger.info(f"Visualization saved to: {plot_file}")


def save_detailed_examples(results, output_dir, num_examples=5):
    """Save detailed examples for inspection."""
    logger = get_logger()

    # Select interesting examples: mix of successful and unsuccessful cases
    sorted_by_improvement = sorted(
        results,
        key=lambda r: r["thinkedit"]["metrics"]["word_count"]
        - r["original"]["metrics"]["word_count"],
        reverse=True,
    )

    examples_to_save = (
        sorted_by_improvement[: num_examples // 2]  # Best improvements
        + sorted_by_improvement[-num_examples // 2 :]  # Worst/no improvements
    )

    examples_file = os.path.join(output_dir, "detailed_examples.json")

    with open(examples_file, "w") as f:
        json.dump(examples_to_save, f, indent=2)

    logger.info(f"Detailed examples saved to: {examples_file}")

    # Also create a readable text file
    text_file = os.path.join(output_dir, "detailed_examples.txt")
    with open(text_file, "w") as f:
        f.write("THINKEDIT MODEL VALIDATION - DETAILED EXAMPLES\n")
        f.write("=" * 80 + "\n\n")

        for i, example in enumerate(examples_to_save):
            orig_words = example["original"]["metrics"]["word_count"]
            edit_words = example["thinkedit"]["metrics"]["word_count"]
            improvement = edit_words - orig_words

            f.write(f"EXAMPLE {i+1} (Question ID: {example['question_id']})\n")
            f.write(
                f"Improvement: {improvement:+d} words ({orig_words} → {edit_words})\n"
            )
            f.write("-" * 40 + "\n")
            f.write(f"QUESTION: {example['question']}\n\n")
            f.write(f"EXPECTED ANSWER: {example['expected_answer']}\n\n")

            f.write("ORIGINAL MODEL:\n")
            f.write(
                f"Thinking ({orig_words} words): {example['original']['thinking'][:500]}{'...' if len(example['original']['thinking']) > 500 else ''}\n"
            )
            f.write(f"Response: {example['original']['response']}\n")
            f.write(f"Extracted Answer: {example['original']['extracted_answer']}\n")
            f.write(f"Correct: {example['original']['correct']}\n\n")

            f.write("THINKEDIT MODEL:\n")
            f.write(
                f"Thinking ({edit_words} words): {example['thinkedit']['thinking'][:500]}{'...' if len(example['thinkedit']['thinking']) > 500 else ''}\n"
            )
            f.write(f"Response: {example['thinkedit']['response']}\n")
            f.write(f"Extracted Answer: {example['thinkedit']['extracted_answer']}\n")
            f.write(f"Correct: {example['thinkedit']['correct']}\n")
            f.write("\n" + "=" * 80 + "\n\n")

    logger.info(f"Readable examples saved to: {text_file}")


def main():
    args = parse_args()

    # Set up logging and device
    logger = setup_logging("validate_thinkedit")
    device = get_device(args.device)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    original_model, thinkedit_model, tokenizer = load_models_and_tokenizer(
        args.original_model, args.thinkedit_model, device
    )

    # Load ThinkEdit metadata
    metadata = load_thinkedit_metadata(args.thinkedit_model)
    if metadata:
        logger.info(
            f"Loaded ThinkEdit metadata: {len(metadata.get('edited_heads', []))} heads edited"
        )
    else:
        logger.warning("No ThinkEdit metadata found")

    # Load test questions
    logger.info(f"Loading {args.num_questions} test questions from GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{args.num_questions}]")

    # Compare models
    results = compare_models(original_model, thinkedit_model, tokenizer, dataset, args)

    # Analyze results
    analysis = analyze_results(results, metadata)

    # Save results
    results_file = os.path.join(args.output_dir, "validation_results.json")
    with open(results_file, "w") as f:
        json.dump({"analysis": analysis, "detailed_results": results}, f, indent=2)

    analysis_file = os.path.join(args.output_dir, "validation_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)

    # Create visualizations
    create_visualizations(analysis, args.output_dir)

    # Save detailed examples if requested
    if args.save_examples:
        save_detailed_examples(results, args.output_dir)

    logger.info(f"Validation complete! Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
