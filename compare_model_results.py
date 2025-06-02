#!/usr/bin/env python3
"""
Compare Model Results

This script compares validation results from two models by reading their JSON output files.
Use this after running validate_single_model_sglang.py on both original and ThinkEdit models.

Usage:
    python compare_model_results.py --original-results validation_results/original_results.json --thinkedit-results validation_results/thinkedit_results.json
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from logging_setup import setup_logging, get_logger


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json_with_numpy_support(data, filepath):
    """Save data to JSON with numpy type conversion."""
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        else:
            return obj
    
    # Convert the entire data structure
    converted_data = convert_to_serializable(data)
    
    with open(filepath, 'w') as f:
        json.dump(converted_data, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original vs ThinkEdit model results")
    parser.add_argument(
        "--original-results",
        type=str,
        required=True,
        help="Path to original model results JSON file",
    )
    parser.add_argument(
        "--thinkedit-results",
        type=str,
        required=True,
        help="Path to ThinkEdit model results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Directory to save comparison results",
    )
    return parser.parse_args()


def load_results(file_path):
    """Load results from JSON file."""
    logger = get_logger()
    logger.info(f"Loading results from: {file_path}")
    
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded {len(results)} results")
    return results


def compare_results(original_results, thinkedit_results):
    """Compare the two sets of results."""
    logger = get_logger()
    
    # Create dictionaries indexed by question_id for proper matching
    original_dict = {r["question_id"]: r for r in original_results}
    thinkedit_dict = {r["question_id"]: r for r in thinkedit_results}
    
    # Get common question IDs
    common_ids = set(original_dict.keys()) & set(thinkedit_dict.keys())
    
    if len(common_ids) != len(original_results) or len(common_ids) != len(thinkedit_results):
        logger.warning(f"Question count mismatch. Original: {len(original_results)}, ThinkEdit: {len(thinkedit_results)}, Common: {len(common_ids)}")
        logger.warning("Proceeding with common questions only")
    
    # Extract metrics for common questions only
    original_thinking_lengths = []
    thinkedit_thinking_lengths = []
    original_correct = 0
    thinkedit_correct = 0
    
    for question_id in common_ids:
        orig = original_dict[question_id]
        edit = thinkedit_dict[question_id]
        
        original_thinking_lengths.append(orig["metrics"]["word_count"])
        thinkedit_thinking_lengths.append(edit["metrics"]["word_count"])
        
        if orig["correct"]:
            original_correct += 1
        if edit["correct"]:
            thinkedit_correct += 1
    
    total_questions = len(common_ids)
    
    # Calculate statistics
    original_stats = {
        "accuracy": original_correct / total_questions,
        "avg_thinking_length": np.mean(original_thinking_lengths),
        "median_thinking_length": np.median(original_thinking_lengths),
        "std_thinking_length": np.std(original_thinking_lengths),
        "min_thinking_length": np.min(original_thinking_lengths),
        "max_thinking_length": np.max(original_thinking_lengths)
    }
    
    thinkedit_stats = {
        "accuracy": thinkedit_correct / total_questions,
        "avg_thinking_length": np.mean(thinkedit_thinking_lengths),
        "median_thinking_length": np.median(thinkedit_thinking_lengths),
        "std_thinking_length": np.std(thinkedit_thinking_lengths),
        "min_thinking_length": np.min(thinkedit_thinking_lengths),
        "max_thinking_length": np.max(thinkedit_thinking_lengths)
    }
    
    # Calculate changes (ensure same ordering for comparison)
    thinking_length_changes = []
    for question_id in sorted(common_ids):
        orig_length = original_dict[question_id]["metrics"]["word_count"]
        edit_length = thinkedit_dict[question_id]["metrics"]["word_count"]
        thinking_length_changes.append(edit_length - orig_length)
    
    comparison = {
        "total_questions": total_questions,
        "original_stats": original_stats,
        "thinkedit_stats": thinkedit_stats,
        "changes": {
            "accuracy_change": thinkedit_stats["accuracy"] - original_stats["accuracy"],
            "avg_thinking_length_change": thinkedit_stats["avg_thinking_length"] - original_stats["avg_thinking_length"],
            "avg_thinking_length_change_pct": ((thinkedit_stats["avg_thinking_length"] - original_stats["avg_thinking_length"]) / original_stats["avg_thinking_length"] * 100) if original_stats["avg_thinking_length"] > 0 else 0,
            "questions_with_longer_thinking": sum(1 for change in thinking_length_changes if change > 0),
            "questions_with_shorter_thinking": sum(1 for change in thinking_length_changes if change < 0),
            "questions_with_same_thinking": sum(1 for change in thinking_length_changes if change == 0),
            "thinking_length_changes": thinking_length_changes
        }
    }
    
    # Log summary
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total questions: {total_questions}")
    logger.info("")
    logger.info("ACCURACY:")
    logger.info(f"  Original model: {original_stats['accuracy']:.2%}")
    logger.info(f"  ThinkEdit model: {thinkedit_stats['accuracy']:.2%}")
    logger.info(f"  Change: {comparison['changes']['accuracy_change']:+.2%}")
    logger.info("")
    logger.info("THINKING LENGTH:")
    logger.info(f"  Original avg: {original_stats['avg_thinking_length']:.1f} words")
    logger.info(f"  ThinkEdit avg: {thinkedit_stats['avg_thinking_length']:.1f} words")
    logger.info(f"  Change: {comparison['changes']['avg_thinking_length_change']:+.1f} words ({comparison['changes']['avg_thinking_length_change_pct']:+.1f}%)")
    logger.info("")
    logger.info("THINKING LENGTH DISTRIBUTION:")
    logger.info(f"  Questions with longer thinking: {comparison['changes']['questions_with_longer_thinking']}/{total_questions}")
    logger.info(f"  Questions with shorter thinking: {comparison['changes']['questions_with_shorter_thinking']}/{total_questions}")
    logger.info(f"  Questions with same thinking: {comparison['changes']['questions_with_same_thinking']}/{total_questions}")
    
    return comparison


def create_detailed_comparison(original_results, thinkedit_results):
    """Create detailed question-by-question comparison."""
    detailed_comparison = []
    
    # Create dictionaries indexed by question_id for efficient lookup
    original_dict = {r["question_id"]: r for r in original_results}
    thinkedit_dict = {r["question_id"]: r for r in thinkedit_results}
    
    # Get all question IDs that exist in both datasets
    common_ids = set(original_dict.keys()) & set(thinkedit_dict.keys())
    
    if len(common_ids) != len(original_results) or len(common_ids) != len(thinkedit_results):
        logger = get_logger()
        logger.warning(f"Not all questions match between files. Original: {len(original_results)}, ThinkEdit: {len(thinkedit_results)}, Common: {len(common_ids)}")
    
    # Sort IDs for consistent ordering
    sorted_ids = sorted(common_ids)
    
    for question_id in sorted_ids:
        orig = original_dict[question_id]
        edit = thinkedit_dict[question_id]
        
        # Double-check that questions match
        if orig["question"] != edit["question"]:
            raise ValueError(f"Question content mismatch for ID {question_id}: questions don't match between files")
        
        comparison_item = {
            "question_id": question_id,
            "question": orig["question"],
            "expected_answer": orig["expected_answer"],
            "original": {
                "thinking_length": orig["metrics"]["word_count"],
                "correct": orig["correct"],
                "extracted_answer": orig["extracted_answer"],
                "thinking": orig["thinking"][:200] + "..." if len(orig["thinking"]) > 200 else orig["thinking"]
            },
            "thinkedit": {
                "thinking_length": edit["metrics"]["word_count"],
                "correct": edit["correct"],
                "extracted_answer": edit["extracted_answer"],
                "thinking": edit["thinking"][:200] + "..." if len(edit["thinking"]) > 200 else edit["thinking"]
            },
            "changes": {
                "thinking_length_change": edit["metrics"]["word_count"] - orig["metrics"]["word_count"],
                "accuracy_change": edit["correct"] - orig["correct"]
            }
        }
        
        detailed_comparison.append(comparison_item)
    
    return detailed_comparison


def create_visualizations(comparison, output_dir):
    """Create comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    original_lengths = [comparison["original_stats"]["avg_thinking_length"]] * comparison["total_questions"]
    thinkedit_lengths = [comparison["thinkedit_stats"]["avg_thinking_length"]] * comparison["total_questions"]
    
    # Get individual lengths for distribution plots
    changes = comparison["changes"]["thinking_length_changes"]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Thinking length change histogram
    ax1.hist(changes, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No change')
    ax1.set_xlabel("Thinking Length Change (words)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Thinking Length Changes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy comparison bar chart
    models = ['Original', 'ThinkEdit']
    accuracies = [comparison["original_stats"]["accuracy"], comparison["thinkedit_stats"]["accuracy"]]
    colors = ['blue', 'red']
    
    bars = ax2.bar(models, accuracies, color=colors, alpha=0.7)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Comparison")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom')
    
    # Thinking length comparison bar chart
    avg_lengths = [comparison["original_stats"]["avg_thinking_length"], 
                   comparison["thinkedit_stats"]["avg_thinking_length"]]
    
    bars = ax3.bar(models, avg_lengths, color=colors, alpha=0.7)
    ax3.set_ylabel("Average Thinking Length (words)")
    ax3.set_title("Average Thinking Length Comparison")
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, length in zip(bars, avg_lengths):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(avg_lengths)*0.01,
                f'{length:.1f}', ha='center', va='bottom')
    
    # Summary statistics
    ax4.axis('off')
    
    summary_text = f"""
COMPARISON SUMMARY

Total Questions: {comparison['total_questions']}

ACCURACY:
• Original: {comparison['original_stats']['accuracy']:.1%}
• ThinkEdit: {comparison['thinkedit_stats']['accuracy']:.1%}
• Change: {comparison['changes']['accuracy_change']:+.1%}

THINKING LENGTH:
• Original avg: {comparison['original_stats']['avg_thinking_length']:.1f} words
• ThinkEdit avg: {comparison['thinkedit_stats']['avg_thinking_length']:.1f} words  
• Change: {comparison['changes']['avg_thinking_length_change']:+.1f} words
• Change %: {comparison['changes']['avg_thinking_length_change_pct']:+.1f}%

EFFECTIVENESS:
• Longer thinking: {comparison['changes']['questions_with_longer_thinking']}/{comparison['total_questions']}
• Shorter thinking: {comparison['changes']['questions_with_shorter_thinking']}/{comparison['total_questions']}
• Same thinking: {comparison['changes']['questions_with_same_thinking']}/{comparison['total_questions']}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger = get_logger()
    logger.info(f"Visualization saved to: {plot_file}")


def save_results(comparison, detailed_comparison, output_dir):
    """Save comparison results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full comparison using numpy-safe JSON
    save_json_with_numpy_support(comparison, os.path.join(output_dir, "comparison_analysis.json"))
    
    # Save detailed comparison using numpy-safe JSON
    save_json_with_numpy_support(detailed_comparison, os.path.join(output_dir, "detailed_comparison.json"))
    
    # Save human-readable summary
    with open(os.path.join(output_dir, "comparison_summary.txt"), "w") as f:
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total questions tested: {comparison['total_questions']}\n\n")
        
        f.write("ACCURACY COMPARISON:\n")
        f.write(f"Original model: {comparison['original_stats']['accuracy']:.2%}\n")
        f.write(f"ThinkEdit model: {comparison['thinkedit_stats']['accuracy']:.2%}\n")
        f.write(f"Change: {comparison['changes']['accuracy_change']:+.2%}\n\n")
        
        f.write("THINKING LENGTH COMPARISON:\n")
        f.write(f"Original avg: {comparison['original_stats']['avg_thinking_length']:.1f} words\n")
        f.write(f"ThinkEdit avg: {comparison['thinkedit_stats']['avg_thinking_length']:.1f} words\n")
        f.write(f"Change: {comparison['changes']['avg_thinking_length_change']:+.1f} words ({comparison['changes']['avg_thinking_length_change_pct']:+.1f}%)\n\n")
        
        f.write("EFFECTIVENESS BREAKDOWN:\n")
        f.write(f"Questions with longer thinking: {comparison['changes']['questions_with_longer_thinking']}\n")
        f.write(f"Questions with shorter thinking: {comparison['changes']['questions_with_shorter_thinking']}\n")
        f.write(f"Questions with same thinking: {comparison['changes']['questions_with_same_thinking']}\n\n")
        
        f.write("QUESTION-BY-QUESTION DETAILS:\n")
        f.write("-" * 40 + "\n")
        
        for item in detailed_comparison:
            f.write(f"\nQuestion {item['question_id'] + 1}:\n")
            f.write(f"  Thinking change: {item['changes']['thinking_length_change']:+d} words\n")
            f.write(f"  Original correct: {item['original']['correct']}\n")
            f.write(f"  ThinkEdit correct: {item['thinkedit']['correct']}\n")
            f.write(f"  Question: {item['question'][:100]}{'...' if len(item['question']) > 100 else ''}\n")


def main():
    """Main function."""
    args = parse_args()
    logger = setup_logging("compare_model_results")
    
    # Load results from both models
    original_results = load_results(args.original_results)
    thinkedit_results = load_results(args.thinkedit_results)
    
    # Validate that we have the same number of results
    if len(original_results) != len(thinkedit_results):
        logger.error(f"Mismatch in number of results: {len(original_results)} vs {len(thinkedit_results)}")
        return
    
    # Compare results
    comparison = compare_results(original_results, thinkedit_results)
    
    # Create detailed comparison
    detailed_comparison = create_detailed_comparison(original_results, thinkedit_results)
    
    # Create visualizations
    create_visualizations(comparison, args.output_dir)
    
    # Save results
    save_results(comparison, detailed_comparison, args.output_dir)
    
    logger.info(f"Comparison complete! Results saved to {args.output_dir}")
    
    # Print final assessment
    changes = comparison["changes"]
    if changes["avg_thinking_length_change"] > 0 and changes["accuracy_change"] >= 0:
        logger.info("✅ ThinkEdit model shows improvement: longer thinking with maintained/better accuracy")
    elif changes["avg_thinking_length_change"] > 0:
        logger.info("⚠️  ThinkEdit model shows longer thinking but with accuracy trade-off")
    else:
        logger.info("❌ ThinkEdit model does not show expected improvement")


if __name__ == "__main__":
    main() 