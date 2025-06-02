#!/usr/bin/env python3
"""
Analyze model responses and create comparison tables.

Usage:
    python analyze_response_tables.py --response-files file1.json file2.json --model-names model1 model2
"""

import argparse
import json
import os
import subprocess
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import logging

def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def ensure_processed_file(input_file: str, force_reprocess: bool = False) -> str:
    """Ensure the response file is processed, process it if needed."""
    logger = logging.getLogger(__name__)
    
    # Check if it's already a processed file
    if "_processed" in input_file and os.path.exists(input_file) and not force_reprocess:
        return input_file
    
    # Generate processed file name
    base_name = os.path.splitext(input_file)[0]
    if "_processed" in base_name:
        processed_file = input_file
    else:
        processed_file = f"{base_name}_processed.json"
    
    # Check if processed file already exists
    if os.path.exists(processed_file) and not force_reprocess:
        logger.info(f"Using existing processed file: {processed_file}")
        return processed_file
    
    # Need to process the file
    logger.info(f"Processing response file: {input_file} -> {processed_file}")
    
    if not os.path.exists("postprocess_responses.py"):
        raise FileNotFoundError("postprocess_responses.py not found. Cannot process response file.")
    
    try:
        cmd = [
            "python", "postprocess_responses.py",
            "--input-file", input_file,
            "--output-file", processed_file
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully processed {input_file}")
        return processed_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {input_file}: {e}")
        logger.error(f"stderr: {e.stderr}")
        raise

def load_response_file(file_path: str) -> List[Dict]:
    """Load responses from JSON file."""
    try:
        with open(file_path, 'r') as f:
            responses = json.load(f)
        return responses
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the tokenizer."""
    if not text:
        return 0
    try:
        if tokenizer is not None:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        else:
            # Fallback: approximate token count (1 token ≈ 4 characters for most models)
            return len(text) // 4
    except Exception:
        # Fallback: approximate token count
        return len(text) // 4

def extract_thinking_text(with_thinking: dict) -> str:
    """Extract thinking text from various possible formats."""
    # Try direct thinking field first
    thinking = with_thinking.get("thinking", "")
    if thinking:
        return thinking
    
    # Try extracting from full_text if available
    full_text = with_thinking.get("full_text", "")
    if full_text and "<think>" in full_text and "</think>" in full_text:
        # Extract thinking from <think> tags
        start = full_text.find("<think>") + 7
        end = full_text.find("</think>")
        if start > 6 and end > start:
            return full_text[start:end].strip()
    
    return thinking

def analyze_responses(responses: List[Dict], tokenizer, model_name: str) -> Dict:
    """Analyze responses and calculate metrics."""
    logger = logging.getLogger(__name__)
    
    # Filter valid responses
    valid_responses = []
    for resp in responses:
        if 'with_thinking' in resp and isinstance(resp['with_thinking'], dict):
            thinking = resp['with_thinking']
            if 'correct' in thinking:  # Must have correctness info
                valid_responses.append(resp)
    
    logger.info(f"Model {model_name}: {len(valid_responses)} valid responses out of {len(responses)}")
    
    if not valid_responses:
        return {
            'model_name': model_name,
            'total_responses': 0,
            'overall_accuracy': 0.0,
            'correct_count': 0,
            'percentile_analysis': {},
            'thinking_stats': {'avg_tokens': 0, 'median_tokens': 0, 'min_tokens': 0, 'max_tokens': 0, 'std_tokens': 0}
        }
    
    # Calculate overall accuracy
    correct_count = sum(1 for resp in valid_responses 
                       if resp['with_thinking'].get('correct', False))
    overall_accuracy = correct_count / len(valid_responses)
    
    # Calculate thinking lengths in tokens
    thinking_data = []
    for resp in valid_responses:
        with_thinking = resp['with_thinking']
        
        # Extract thinking text using multiple strategies
        thinking_text = extract_thinking_text(with_thinking)
        
        # Count tokens
        thinking_tokens = count_tokens(thinking_text, tokenizer)
        is_correct = with_thinking.get('correct', False)
        
        thinking_data.append({
            'id': resp.get('id', 0),
            'thinking_tokens': thinking_tokens,
            'thinking_text': thinking_text,
            'correct': is_correct,
            'question': resp.get('question', ''),
            'response': with_thinking.get('response', ''),
            'extracted_answer': with_thinking.get('extracted_answer', 'N/A')
        })
    
    # Sort by thinking length (shortest first)
    thinking_data.sort(key=lambda x: x['thinking_tokens'])
    
    # Calculate percentile metrics
    total_count = len(thinking_data)
    percentiles = [5, 10, 20]
    percentile_analysis = {}
    
    for p in percentiles:
        # Get top p% shortest responses
        cutoff_idx = max(1, int(total_count * p / 100))
        shortest_responses = thinking_data[:cutoff_idx]
        
        # Calculate metrics for this percentile
        correct_in_percentile = sum(1 for r in shortest_responses if r['correct'])
        accuracy_in_percentile = correct_in_percentile / len(shortest_responses) if shortest_responses else 0
        avg_length_in_percentile = np.mean([r['thinking_tokens'] for r in shortest_responses]) if shortest_responses else 0
        
        percentile_analysis[f'top_{p}%'] = {
            'count': len(shortest_responses),
            'accuracy': accuracy_in_percentile,
            'avg_tokens': avg_length_in_percentile,
            'max_tokens': max([r['thinking_tokens'] for r in shortest_responses]) if shortest_responses else 0,
            'min_tokens': min([r['thinking_tokens'] for r in shortest_responses]) if shortest_responses else 0,
            'examples': shortest_responses[:3]  # Store first 3 examples for inspection
        }
    
    # Overall thinking length statistics
    all_thinking_lengths = [d['thinking_tokens'] for d in thinking_data]
    
    return {
        'model_name': model_name,
        'total_responses': len(valid_responses),
        'overall_accuracy': overall_accuracy,
        'correct_count': correct_count,
        'percentile_analysis': percentile_analysis,
        'thinking_stats': {
            'avg_tokens': np.mean(all_thinking_lengths),
            'median_tokens': np.median(all_thinking_lengths),
            'min_tokens': min(all_thinking_lengths),
            'max_tokens': max(all_thinking_lengths),
            'std_tokens': np.std(all_thinking_lengths)
        }
    }

def create_summary_tables(analysis_results: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create summary tables for the analysis results."""
    
    # Table 1: Overall Accuracy
    overall_data = []
    for result in analysis_results:
        overall_data.append({
            'Model': result['model_name'],
            'Total Responses': result['total_responses'],
            'Overall Accuracy (%)': f"{result['overall_accuracy'] * 100:.2f}",
            'Correct Count': result['correct_count'],
            'Avg Thinking Length (tokens)': f"{result['thinking_stats']['avg_tokens']:.1f}"
        })
    
    overall_table = pd.DataFrame(overall_data)
    
    # Table 2: Accuracy by Percentiles
    accuracy_data = []
    for result in analysis_results:
        row = {'Model': result['model_name']}
        for percentile in ['top_5%', 'top_10%', 'top_20%']:
            if percentile in result['percentile_analysis']:
                accuracy = result['percentile_analysis'][percentile]['accuracy'] * 100
                row[f'{percentile.replace("top_", "").replace("%", "")}% Shortest Accuracy (%)'] = f"{accuracy:.2f}"
            else:
                row[f'{percentile.replace("top_", "").replace("%", "")}% Shortest Accuracy (%)'] = "N/A"
        accuracy_data.append(row)
    
    accuracy_table = pd.DataFrame(accuracy_data)
    
    # Table 3: Average Reasoning Length by Percentiles
    length_data = []
    for result in analysis_results:
        row = {'Model': result['model_name']}
        for percentile in ['top_5%', 'top_10%', 'top_20%']:
            if percentile in result['percentile_analysis']:
                p_data = result['percentile_analysis'][percentile]
                avg_tokens = p_data['avg_tokens']
                max_tokens = p_data['max_tokens']
                min_tokens = p_data['min_tokens']
                count = p_data['count']
                
                p_label = percentile.replace("top_", "").replace("%", "")
                row[f'{p_label}% Avg Length (tokens)'] = f"{avg_tokens:.1f}"
                row[f'{p_label}% Max Length (tokens)'] = f"{max_tokens}"
                row[f'{p_label}% Min Length (tokens)'] = f"{min_tokens}"
                row[f'{p_label}% Count'] = count
            else:
                p_label = percentile.replace("top_", "").replace("%", "")
                row[f'{p_label}% Avg Length (tokens)'] = "N/A"
                row[f'{p_label}% Max Length (tokens)'] = "N/A"
                row[f'{p_label}% Min Length (tokens)'] = "N/A"
                row[f'{p_label}% Count'] = "N/A"
        length_data.append(row)
    
    length_table = pd.DataFrame(length_data)
    
    return overall_table, accuracy_table, length_table

def save_detailed_analysis(analysis_results: List[Dict], output_dir: str):
    """Save detailed analysis results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for result in analysis_results:
        model_name = result['model_name']
        
        # Save detailed analysis
        output_file = os.path.join(output_dir, f"{model_name}_detailed_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Save examples from each percentile
        for percentile, data in result['percentile_analysis'].items():
            if 'examples' in data:
                examples_file = os.path.join(output_dir, f"{model_name}_{percentile}_examples.json")
                with open(examples_file, 'w') as f:
                    json.dump(data['examples'], f, indent=2)

def print_tables(overall_table: pd.DataFrame, accuracy_table: pd.DataFrame, length_table: pd.DataFrame):
    """Print formatted tables to console."""
    print("\n" + "="*100)
    print("OVERALL ACCURACY TABLE")
    print("="*100)
    print(overall_table.to_string(index=False))
    
    print("\n" + "="*100)
    print("ACCURACY OF SHORTEST REASONING RESPONSES")
    print("="*100)
    print(accuracy_table.to_string(index=False))
    
    print("\n" + "="*100)
    print("AVERAGE REASONING LENGTH FOR SHORTEST RESPONSES")
    print("="*100)
    print(length_table.to_string(index=False))

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze response accuracy tables")
    parser.add_argument(
        "--response-files",
        nargs="+",
        required=True,
        help="Path(s) to response JSON files"
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        help="Model names for the response files (same order as response_files)"
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer model for length calculation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save summary tables as CSV files"
    )
    parser.add_argument(
        "--analysis-type",
        choices=["accuracy", "thinking_length", "both"],
        default="both",
        help="Type of analysis to perform"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing even if processed files exist"
    )
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    level = logging.DEBUG
    logger = setup_logging(level)
    
    # Validate inputs
    if len(args.response_files) < 1:
        raise ValueError("At least one response file must be provided")
    
    # Check if files exist
    for file_path in args.response_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Response file not found: {file_path}")
    
    # Ensure all files are processed
    processed_files = []
    for file_path in args.response_files:
        try:
            processed_file = ensure_processed_file(file_path, args.force_reprocess)
            processed_files.append(processed_file)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            raise
    
    # Generate model names if not provided
    if args.model_names:
        if len(args.model_names) != len(processed_files):
            raise ValueError("Number of model names must match number of response files")
        model_names = args.model_names
    else:
        model_names = []
        for f in processed_files:
            basename = os.path.splitext(os.path.basename(f))[0]
            # Clean up the name
            basename = basename.replace("_processed", "").replace("_gsm8k_responses", "")
            model_names.append(basename)
    
    logger.info(f"Analyzing {len(processed_files)} response files...")
    logger.info(f"Model names: {model_names}")
    logger.info(f"Processed files: {processed_files}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load tokenizer {args.tokenizer_model}: {e}")
        logger.warning("Using approximate token counting...")
        tokenizer = None
    
    # Analyze each response file
    analysis_results = []
    for file_path, model_name in zip(processed_files, model_names):
        logger.info(f"Analyzing {model_name} from {file_path}...")
        
        responses = load_response_file(file_path)
        analysis = analyze_responses(responses, tokenizer, model_name)
        analysis_results.append(analysis)
        
        logger.info(f"  Total responses: {analysis['total_responses']}")
        logger.info(f"  Overall accuracy: {analysis['overall_accuracy']:.2%}")
        logger.info(f"  Avg thinking length: {analysis['thinking_stats']['avg_tokens']:.1f} tokens")
    
    # Create summary tables
    logger.info("Creating summary tables...")
    overall_table, accuracy_table, length_table = create_summary_tables(analysis_results)
    
    # Print tables
    print_tables(overall_table, accuracy_table, length_table)
    
    # Save detailed analysis
    logger.info(f"Saving detailed analysis to {args.output_dir}...")
    save_detailed_analysis(analysis_results, args.output_dir)
    
    # Save CSV files if requested
    if args.save_csv:
        os.makedirs(args.output_dir, exist_ok=True)
        
        overall_table.to_csv(os.path.join(args.output_dir, "overall_accuracy.csv"), index=False)
        accuracy_table.to_csv(os.path.join(args.output_dir, "shortest_responses_accuracy.csv"), index=False)
        length_table.to_csv(os.path.join(args.output_dir, "shortest_responses_length.csv"), index=False)
        
        logger.info("Tables saved as CSV files")
    
    # Summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    for result in analysis_results:
        print(f"\n{result['model_name']}:")
        print(f"  Total responses: {result['total_responses']}")
        print(f"  Overall accuracy: {result['overall_accuracy']:.2%}")
        print(f"  Average thinking length: {result['thinking_stats']['avg_tokens']:.1f} tokens")
        print(f"  Thinking length range: {result['thinking_stats']['min_tokens']}-{result['thinking_stats']['max_tokens']} tokens")
        
        # Show percentile details
        for percentile, data in result['percentile_analysis'].items():
            print(f"  {percentile}: {data['count']} responses, {data['accuracy']:.2%} accuracy, {data['avg_tokens']:.1f} avg tokens")

if __name__ == "__main__":
    main() 