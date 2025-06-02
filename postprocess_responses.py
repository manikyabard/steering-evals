#!/usr/bin/env python3
"""
Post-process GSM8K response files to add improved answer parsing.

Usage:
    python postprocess_responses.py --input-file responses/Qwen3-0.6B_gsm8k_responses.json --output-file responses/Qwen3-0.6B_gsm8k_responses_processed.json
"""

import os
import json
import argparse
import re
import numpy as np
from tqdm import tqdm
from logging_setup import setup_logging, get_logger

# GSM8K answer parsing patterns
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def parse_args():
    parser = argparse.ArgumentParser(description="Postprocess GSM8K responses")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input JSON file with model responses",
    )
    parser.add_argument(
        "--output-file", type=str, help="Output file for processed responses"
    )
    parser.add_argument(
        "--backup-original",
        action="store_true",
        help="Create backup of original file before processing",
    )
    parser.add_argument(
        "--overwrite-input",
        action="store_true",
        help="Overwrite input file with processed responses",
    )
    return parser.parse_args()


def extract_answer_hf(completion):
    """Extract answer using GSM8K format (#### answer)."""
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            return eval(match_str)
        except:
            return INVALID_ANS
    else:
        return INVALID_ANS


def extract_answer_fallback(completion):
    """Fallback: extract last number from completion."""
    try:
        # Look for boxed answer first
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(boxed_pattern, completion)
        if match:
            boxed_content = match.group(1).strip()
            # Try to extract number from boxed content
            numbers = re.findall(r"-?\d+(?:\.\d+)?", boxed_content)
            if numbers:
                return eval(numbers[-1])
        
        # Fallback to last number in text
        numbers = re.findall(r"-?\d+(?:\.\d+)?", completion)
        if numbers:
            return eval(numbers[-1])
        return INVALID_ANS
    except:
        return INVALID_ANS


def extract_answer(completion):
    """Extract answer from completion using multiple strategies."""
    # Try GSM8K format first
    answer = extract_answer_hf(completion)
    if answer != INVALID_ANS:
        return answer
    
    # Try fallback method
    return extract_answer_fallback(completion)


def is_correct(completion, ground_truth_answer):
    """Check if completion answer matches ground truth."""
    try:
        # Extract ground truth
        gold = extract_answer_hf(ground_truth_answer)
        if gold == INVALID_ANS:
            # Try to extract from ground truth using fallback
            gold = extract_answer_fallback(ground_truth_answer)
        
        if gold == INVALID_ANS:
            return False
        
        # Extract predicted answer
        predicted = extract_answer(completion)
        
        if predicted == INVALID_ANS:
            return False
        
        # Compare (handle floating point comparison)
        if isinstance(predicted, float) or isinstance(gold, float):
            return abs(predicted - gold) < 1e-6
        else:
            return predicted == gold
            
    except Exception as e:
        return False


def calculate_thinking_metrics(thinking_text):
    """Calculate metrics for thinking content."""
    if not thinking_text:
        return {"word_count": 0, "char_count": 0, "line_count": 0, "reasoning_steps": 0}

    words = thinking_text.split()
    lines = thinking_text.split("\n")

    # Count reasoning steps (simple heuristic)
    reasoning_indicators = [
        "first", "second", "third", "next", "then", "so", "therefore", 
        "step", "now", "let's", "we need", "i need"
    ]
    
    reasoning_steps = sum(1 for word in words if word.lower() in reasoning_indicators)

    return {
        "word_count": len(words),
        "char_count": len(thinking_text),
        "line_count": len(lines),
        "reasoning_steps": reasoning_steps
    }


def process_response_entry(entry):
    """Process a single response entry to add new fields."""
    processed_entry = entry.copy()
    
    # Extract data from with_thinking section
    with_thinking = entry.get("with_thinking", {})
    response_text = with_thinking.get("response", "")
    thinking_text = with_thinking.get("thinking", "")
    ground_truth = entry.get("answer", "")
    
    # Check if already processed (has extracted_answer field)
    if "extracted_answer" in with_thinking:
        return processed_entry
    
    # Extract answer using improved methods
    extracted_answer = extract_answer(response_text)
    
    # Check correctness
    correct = is_correct(response_text, ground_truth)
    
    # Calculate thinking metrics
    thinking_metrics = calculate_thinking_metrics(thinking_text)
    
    # Add new fields to with_thinking section
    processed_entry["with_thinking"]["extracted_answer"] = extracted_answer
    processed_entry["with_thinking"]["correct"] = correct
    processed_entry["with_thinking"]["thinking_metrics"] = thinking_metrics
    
    # Add full_text if not present
    if "full_text" not in processed_entry["with_thinking"]:
        processed_entry["with_thinking"]["full_text"] = response_text
    
    return processed_entry


def analyze_processed_responses(responses):
    """Analyze the processed responses and return statistics."""
    total_responses = len(responses)
    
    if total_responses == 0:
        return {"error": "No responses to analyze"}
    
    # Count correct responses
    correct_count = sum(1 for r in responses 
                       if r.get("with_thinking", {}).get("correct", False))
    accuracy = correct_count / total_responses
    
    # Analyze thinking lengths
    thinking_lengths = []
    for r in responses:
        thinking_metrics = r.get("with_thinking", {}).get("thinking_metrics", {})
        thinking_lengths.append(thinking_metrics.get("word_count", 0))
    
    # Response lengths (for comparison)
    response_lengths = []
    for r in responses:
        response = r.get("with_thinking", {}).get("response", "")
        response_lengths.append(len(response.split()))
    
    analysis = {
        "total_responses": total_responses,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "thinking_stats": {
            "avg_length": float(np.mean(thinking_lengths)),
            "median_length": float(np.median(thinking_lengths)),
            "std_length": float(np.std(thinking_lengths)),
            "min_length": int(np.min(thinking_lengths)),
            "max_length": int(np.max(thinking_lengths))
        },
        "response_stats": {
            "avg_length": float(np.mean(response_lengths)),
            "median_length": float(np.median(response_lengths)),
            "std_length": float(np.std(response_lengths)),
            "min_length": int(np.min(response_lengths)),
            "max_length": int(np.max(response_lengths))
        }
    }
    
    return analysis


def main():
    """Main processing function."""
    args = parse_args()
    logger = setup_logging("postprocess_responses")
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"Input file does not exist: {args.input_file}")
        return
    
    # Determine output file
    if args.overwrite_input:
        output_file = args.input_file
        if args.backup_original:
            backup_file = args.input_file + ".backup"
            logger.info(f"Creating backup: {backup_file}")
            os.rename(args.input_file, backup_file)
            args.input_file = backup_file  # Read from backup
    elif args.output_file:
        output_file = args.output_file
    else:
        # Default: add _processed suffix
        base_name = os.path.splitext(args.input_file)[0]
        extension = os.path.splitext(args.input_file)[1]
        output_file = f"{base_name}_processed{extension}"
    
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {output_file}")
    
    # Load responses
    logger.info("Loading response file...")
    try:
        with open(args.input_file, 'r') as f:
            responses = json.load(f)
        logger.info(f"Loaded {len(responses)} responses")
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return
    
    # Process responses
    logger.info("Processing responses...")
    processed_responses = []
    
    for entry in tqdm(responses, desc="Processing responses"):
        try:
            processed_entry = process_response_entry(entry)
            processed_responses.append(processed_entry)
        except Exception as e:
            logger.warning(f"Error processing entry {entry.get('id', 'unknown')}: {e}")
            # Keep original entry if processing fails
            processed_responses.append(entry)
    
    # Analyze results
    logger.info("Analyzing processed responses...")
    analysis = analyze_processed_responses(processed_responses)
    
    # Save processed responses
    logger.info(f"Saving processed responses to {output_file}...")
    try:
        with open(output_file, 'w') as f:
            json.dump(processed_responses, f, indent=2)
        logger.info(f"Successfully saved {len(processed_responses)} processed responses")
    except Exception as e:
        logger.error(f"Error saving output file: {e}")
        return
    
    # Save analysis
    analysis_file = os.path.splitext(output_file)[0] + "_analysis.json"
    logger.info(f"Saving analysis to {analysis_file}...")
    try:
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    except Exception as e:
        logger.warning(f"Error saving analysis file: {e}")
    
    # Print summary
    logger.info("=== PROCESSING SUMMARY ===")
    logger.info(f"Total responses processed: {analysis['total_responses']}")
    logger.info(f"Accuracy: {analysis['accuracy']:.2%} ({analysis['correct_count']}/{analysis['total_responses']})")
    logger.info(f"Average thinking length: {analysis['thinking_stats']['avg_length']:.1f} words")
    logger.info(f"Average response length: {analysis['response_stats']['avg_length']:.1f} words")
    logger.info(f"Thinking length range: {analysis['thinking_stats']['min_length']}-{analysis['thinking_stats']['max_length']} words")
    logger.info(f"Response length range: {analysis['response_stats']['min_length']}-{analysis['response_stats']['max_length']} words")
    
    # Show some examples
    logger.info("\n=== EXAMPLE PROCESSED ENTRIES ===")
    for i, entry in enumerate(processed_responses[:3]):
        with_thinking = entry.get("with_thinking", {})
        logger.info(f"\nExample {i+1} (ID: {entry.get('id', 'unknown')}):")
        logger.info(f"  Question: {entry.get('question', '')[:100]}...")
        logger.info(f"  Extracted answer: {with_thinking.get('extracted_answer', 'N/A')}")
        logger.info(f"  Correct: {with_thinking.get('correct', 'N/A')}")
        logger.info(f"  Thinking length: {with_thinking.get('thinking_metrics', {}).get('word_count', 0)} words")


if __name__ == "__main__":
    main() 