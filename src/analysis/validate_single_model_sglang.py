#!/usr/bin/env python3
"""
Validate Single Model using SGLang

This script tests a single model using SGLang server to generate responses
and analyze thinking patterns. It selects questions with the shortest responses
from a previously generated response file.

Usage:
    # Start server:
    python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port 30000 --reasoning-parser qwen3
    
    # Run validation:
    python validate_single_model_sglang.py --port 30000 --model-name original --num-questions 6 --response-file responses/Qwen3-0.6B_gsm8k_responses.json
"""

import os
import json
import argparse
import requests
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from logging_setup import setup_logging, get_logger
import re
import concurrent.futures

# GSM8K answer parsing patterns
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate single model using SGLang")
    parser.add_argument("--port", type=int, default=30000, help="SGLang server port")
    parser.add_argument(
        "--server-host", type=str, default="localhost", help="Server host"
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="Name for this model (original/thinkedit)"
    )
    parser.add_argument(
        "--num-questions", type=int, default=100, help="Number of questions to evaluate"
    )
    parser.add_argument(
        "--response-file", 
        type=str, 
        help="Optional: Use responses from existing file instead of generating new ones"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=32768, help="Max tokens to generate"
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
        default="validation_results",
        help="Directory to save results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def connect_to_sglang(server_url):
    """Test connection to the SGLang server."""
    logger = get_logger()
    logger.info(f"Testing connection to SGLang server at {server_url}...")

    try:
        response = requests.get(f"{server_url}/health")
        response.raise_for_status()
        logger.info(f"Successfully connected to SGLang server at {server_url}")
        return True
    except Exception as e:
        logger.error(f"Error connecting to SGLang server: {e}")
        return False


def load_and_select_shortest_questions(response_file_path, num_questions):
    """Load response file and select questions with shortest responses."""
    logger = get_logger()
    logger.info(f"Loading response file: {response_file_path}")
    
    with open(response_file_path, 'r') as f:
        responses = json.load(f)
    
    logger.info(f"Loaded {len(responses)} responses from file")
    
    # Calculate response lengths for each question
    question_lengths = []
    for resp in responses:
        # Use the response text length (without thinking)
        response_text = resp.get("with_thinking", {}).get("response", "")
        response_length = len(response_text.split())
        
        question_lengths.append({
            "question": resp["question"],
            "answer": resp["answer"],
            "response_length": response_length,
            "id": resp.get("id", len(question_lengths))
        })
    
    # Sort by response length (shortest first)
    question_lengths.sort(key=lambda x: x["response_length"])
    
    # Select the k shortest
    selected_questions = question_lengths[:num_questions]
    
    logger.info(f"Selected {len(selected_questions)} questions with shortest responses")
    logger.info(f"Response length range: {selected_questions[0]['response_length']} - {selected_questions[-1]['response_length']} words")
    
    return [{"question": q["question"], "answer": q["answer"]} for q in selected_questions]


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


def generate_response_sglang(server_url, tokenizer, question, args):
    """Generate response using SGLang server."""
    prompt = f"Solve this math problem step by step, and put your final answer within \\boxed{{}}:\n{question}"

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    generate_payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
    }

    try:
        response = requests.post(f"{server_url}/generate", json=generate_payload)
        response.raise_for_status()
        result = response.json()
        generated_text = result["text"]

        # Use separate_reasoning to extract thinking and response
        reasoning_payload = {"text": generated_text, "reasoning_parser": "qwen3"}
        reasoning_response = requests.post(
            f"{server_url}/separate_reasoning", json=reasoning_payload
        )
        reasoning_response.raise_for_status()
        reasoning_result = reasoning_response.json()

        return {
            "thinking": reasoning_result.get("reasoning_text", ""),
            "response": reasoning_result.get("text", generated_text),
            "full_text": generated_text
        }
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error in generation: {e}")
        return {
            "thinking": "",
            "response": f"Error: {str(e)}",
            "full_text": f"Error: {str(e)}"
        }
    except Exception as e:
        print(f"Error in generation: {e}")
        return {
            "thinking": "",
            "response": f"Error: {str(e)}",
            "full_text": f"Error: {str(e)}"
        }


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


def process_batch(server_url, tokenizer, questions_batch, args, start_idx):
    """Process a batch of questions in parallel."""
    logger = get_logger()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(questions_batch)) as executor:
        future_to_idx = {}
        
        for i, question_data in enumerate(questions_batch):
            question = question_data["question"]
            expected_answer = question_data["answer"]
            future = executor.submit(generate_response_sglang, server_url, tokenizer, question, args)
            future_to_idx[future] = (i, question, expected_answer)
        
        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(questions_batch),
            desc=f"Batch starting at {start_idx}",
        ):
            i, question, expected_answer = future_to_idx[future]
            try:
                result = future.result()
                
                # Calculate metrics
                metrics = calculate_thinking_metrics(result["thinking"])
                
                # Extract answer and check correctness using improved methods
                extracted_answer = extract_answer(result["response"])
                correct = is_correct(result["response"], expected_answer)
                
                result_data = {
                    "question_id": start_idx + i,
                    "question": question,
                    "expected_answer": expected_answer,
                    "thinking": result["thinking"],
                    "response": result["response"],
                    "full_text": result["full_text"],
                    "extracted_answer": extracted_answer,
                    "correct": correct,
                    "metrics": metrics
                }
                
                results.append(result_data)
                
                logger.info(f"Question {start_idx + i + 1}: Thinking={metrics['word_count']} words, "
                           f"Answer={extracted_answer}, Correct={correct}")
                
            except Exception as e:
                logger.error(f"Error processing question {start_idx + i}: {e}")
                result_data = {
                    "question_id": start_idx + i,
                    "question": question,
                    "expected_answer": expected_answer,
                    "thinking": "",
                    "response": f"Error: {str(e)}",
                    "full_text": f"Error: {str(e)}",
                    "extracted_answer": INVALID_ANS,
                    "correct": False,
                    "metrics": {"word_count": 0, "char_count": 0, "line_count": 0, "reasoning_steps": 0},
                    "error": str(e)
                }
                results.append(result_data)
    
    return results


def test_model(server_url, tokenizer, test_questions, args):
    """Test model on test questions using batching."""
    logger = get_logger()
    logger.info(f"Testing model on {len(test_questions)} questions with batching (batch_size={args.batch_size})...")
    
    all_results = []
    
    # Process in batches
    for i in range(0, len(test_questions), args.batch_size):
        end_idx = min(i + args.batch_size, len(test_questions))
        batch = test_questions[i:end_idx]
        
        logger.info(f"Processing batch {i//args.batch_size + 1}/{(len(test_questions) + args.batch_size - 1)//args.batch_size}: questions {i+1}-{end_idx}")
        
        batch_results = process_batch(server_url, tokenizer, batch, args, i)
        all_results.extend(batch_results)
    
    return all_results


def analyze_results(results, model_name):
    """Analyze the test results."""
    logger = get_logger()
    
    # Collect metrics
    thinking_lengths = [r["metrics"]["word_count"] for r in results]
    correct_count = sum(1 for r in results if r["correct"])
    
    # Calculate statistics
    analysis = {
        "model_name": model_name,
        "total_questions": len(results),
        "accuracy": correct_count / len(results),
        "thinking_stats": {
            "avg_length": float(np.mean(thinking_lengths)),
            "median_length": float(np.median(thinking_lengths)),
            "std_length": float(np.std(thinking_lengths)),
            "min_length": int(np.min(thinking_lengths)),
            "max_length": int(np.max(thinking_lengths)),
            "lengths": [int(x) for x in thinking_lengths]
        }
    }
    
    # Log summary
    logger.info(f"=== {model_name.upper()} MODEL RESULTS ===")
    logger.info(f"Total questions: {analysis['total_questions']}")
    logger.info(f"Accuracy: {analysis['accuracy']:.2%}")
    logger.info(f"Average thinking length: {analysis['thinking_stats']['avg_length']:.1f} words")
    logger.info(f"Median thinking length: {analysis['thinking_stats']['median_length']:.1f} words")
    logger.info(f"Thinking length range: {analysis['thinking_stats']['min_length']}-{analysis['thinking_stats']['max_length']} words")
    
    return analysis


def save_results(results, analysis, output_dir, model_name):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    with open(os.path.join(output_dir, f"{model_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save analysis summary
    with open(os.path.join(output_dir, f"{model_name}_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Save human-readable summary
    with open(os.path.join(output_dir, f"{model_name}_summary.txt"), "w") as f:
        f.write(f"{model_name.upper()} Model Validation Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total questions tested: {analysis['total_questions']}\n")
        f.write(f"Accuracy: {analysis['accuracy']:.2%}\n")
        f.write(f"Average thinking length: {analysis['thinking_stats']['avg_length']:.1f} words\n")
        f.write(f"Median thinking length: {analysis['thinking_stats']['median_length']:.1f} words\n")
        f.write(f"Thinking length range: {analysis['thinking_stats']['min_length']}-{analysis['thinking_stats']['max_length']} words\n\n")
        
        f.write("INDIVIDUAL RESULTS:\n")
        for i, result in enumerate(results):
            f.write(f"\nQuestion {i+1}:\n")
            f.write(f"  Thinking length: {result['metrics']['word_count']} words\n")
            f.write(f"  Correct: {result['correct']}\n")
            f.write(f"  Expected: {result['expected_answer']}\n")
            f.write(f"  Extracted: {result['extracted_answer']}\n")


def main():
    """Main function."""
    args = parse_args()
    logger = setup_logging(f"validate_single_model_{args.model_name}")
    
    # Setup server URL
    server_url = f"http://{args.server_host}:{args.port}"
    
    # Test connection
    logger.info("Testing server connection...")
    if not connect_to_sglang(server_url):
        logger.error(f"Cannot connect to model server at {server_url}")
        return
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    
    # Load and select questions with shortest responses
    test_questions = load_and_select_shortest_questions(args.response_file, args.num_questions)
    
    # Test model
    results = test_model(server_url, tokenizer, test_questions, args)
    
    # Analyze results
    analysis = analyze_results(results, args.model_name)
    
    # Save results
    save_results(results, analysis, args.output_dir, args.model_name)
    
    logger.info(f"Validation complete! Results saved to {args.output_dir}")
    
    # Print some example responses
    logger.info("\n=== EXAMPLE RESPONSES ===")
    for i, result in enumerate(results[:2]):  # Show first 2 examples
        logger.info(f"\nExample {i+1}:")
        logger.info(f"Question: {result['question']}")
        logger.info(f"Thinking ({result['metrics']['word_count']} words): {result['thinking'][:200]}...")
        logger.info(f"Response: {result['response']}")
        logger.info(f"Expected: {result['expected_answer']}")
        logger.info(f"Extracted: {result['extracted_answer']}")
        logger.info(f"Correct: {result['correct']}")


if __name__ == "__main__":
    main() 