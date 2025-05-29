#!/usr/bin/env python3
"""
Steering Reasoning Length

This script applies extracted reasoning length direction during model generation
to control the length of reasoning in the model's responses.

We can steer the model toward longer or shorter reasoning by adjusting the
steering strength (alpha parameter). This allows us to:
1. Control the verbosity of the model's reasoning
2. Observe how reasoning length affects accuracy
3. Find optimal reasoning length for different tasks
"""

# %% [markdown]
# # Steering Reasoning Length
#
# This notebook demonstrates how to apply reasoning length direction vectors to control
# model reasoning verbosity. We'll walk through the complete process step by step,
# from loading direction vectors to evaluating steering effectiveness.
#
# ## Overview
# The process involves:
# 1. Loading pre-extracted direction vectors
# 2. Setting up steering mechanisms (hooks)
# 3. Generating responses with different steering strengths
# 4. Evaluating the effect on reasoning length and accuracy
# 5. Visualizing the results

# %% Setup and imports
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

# %% [markdown]
# ## Configuration and Arguments
#
# First, let's set up the configuration parameters for our steering experiments.


# %% Command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Steer model reasoning length")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--directions_dir",
        type=str,
        default="directions",
        help="Directory with saved direction vectors",
    )
    parser.add_argument(
        "--directions_file",
        type=str,
        default=None,
        help="Direct path to directions file (overrides directions_dir and auto-naming)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="steering_results",
        help="Directory to save steering results",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--direction_weights",
        nargs="+",
        type=float,
        default=[-0.08, 0, 0.08],
        help="Direction weights to apply (alpha values)",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["attn", "mlp", "both"],
        default="attn",
        help="Which component to steer (attention, MLP, or both)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--top_k",
        type=float,
        default=20,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing multiple examples simultaneously",
    )
    parser.add_argument(
        "--use_efficient_mode",
        action="store_true",
        help="Use efficient batch processing mode for faster evaluation",
    )
    parser.add_argument(
        "--low_memory_mode",
        action="store_true",
        help="Enable low memory optimizations (smaller batch sizes, more frequent cleanup)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
        help="Device to use for computation (e.g., cuda:0, cpu)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing intermediate results if available",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start fresh, ignoring any existing intermediate results",
    )
    return parser.parse_args()


# %% Interactive configuration (for notebook use)
# Uncomment and modify this cell when running interactively in a notebook
"""
# Test configuration for interactive development
test_args = type('Args', (), {
    'model': "Qwen/Qwen3-0.6B",
    'directions_dir': "directions",
    'directions_file': None,
    'output_dir': "steering_results",
    'num_samples': 3,  # Small number for testing
    'direction_weights': [-0.08, 0.0, 0.08],
    'component': "attn",
    'max_new_tokens': 32768,
    'temperature': 0.6,
    'top_p': 0.95,
    'top_k': 20,
    'batch_size': 2,
    'use_efficient_mode': True,
    'low_memory_mode': False,
    'seed': 42,
    'device': "cuda:0" if torch.cuda.is_available() else "cpu",
    'resume': True,
    'no_resume': False
})()

# Use test_args instead of parse_args() when running interactively
# args = test_args
"""

# %% [markdown]
# ## Step 1: Direction Vector Loading
#
# Let's start by loading the pre-extracted direction vectors that control reasoning length.


# %% Direction loading functions
def load_directions(directions_dir, model_name, component, directions_file=None):
    """Load direction vectors from file."""
    if directions_file:
        directions_path = directions_file
    else:
        model_short_name = model_name.split("/")[-1]
        directions_path = os.path.join(
            directions_dir,
            f"{model_short_name}_thinking_length_direction_gsm8k_{component}.pt",
        )

    if not os.path.exists(directions_path):
        raise FileNotFoundError(f"Directions file not found: {directions_path}")

    directions = torch.load(directions_path)
    print(f"✓ Loaded directions from: {directions_path}")
    print(f"  Shape: {directions.shape}")
    print(f"  Number of layers: {directions.shape[0]}")
    print(f"  Hidden dimension: {directions.shape[1]}")

    return directions, directions_path


# %% Test directions loading (uncomment to test interactively)
"""
def test_directions_loading():
    \"\"\"Test loading direction vectors.\"\"\"
    try:
        print("Testing directions loading...")
        
        directions, filepath = load_directions(
            "directions", 
            "Qwen/Qwen3-0.6B", 
            "attn"
        )
        
        print(f"✓ Successfully loaded directions")
        print(f"  Direction magnitude range: {directions.abs().min().item():.6f} to {directions.abs().max().item():.6f}")
        
        # Show some statistics
        layer_norms = [torch.norm(directions[i]).item() for i in range(min(3, directions.shape[0]))]
        print(f"  First 3 layer norms: {layer_norms}")
        
    except Exception as e:
        print(f"Directions loading test failed: {e}")
        print("Make sure to run extract_reasoning_length_direction_improved.py first")

# Uncomment to test:
# test_directions_loading()
"""

# %% [markdown]
# ## Step 2: Steering Mechanism Setup
#
# Now we'll set up the steering mechanism using forward hooks.


# %% Steering mechanism functions
def apply_steering_layers(model, directions, alpha=0.0, component="attn"):
    """Apply steering layers to the model - ThinkEdit style."""
    if component == "attn":

        def adjust_residual_hook(layer_idx):
            def hook_fn(module, input, output):
                return (
                    output[0] + alpha * directions[layer_idx].to(model.device),
                ) + output[1:]

            return hook_fn

        hooks = []
        for i, layer in enumerate(model.model.layers):
            if i < len(directions):
                hook = layer.self_attn.register_forward_hook(adjust_residual_hook(i))
                hooks.append(hook)
        return hooks

    elif component == "mlp":

        def adjust_residual_hook(layer_idx):
            def hook_fn(module, input, output):
                return output + alpha * directions[layer_idx].to(model.device)

            return hook_fn

        hooks = []
        for i, layer in enumerate(model.model.layers):
            if i < len(directions):
                hook = layer.mlp.register_forward_hook(adjust_residual_hook(i))
                hooks.append(hook)
        return hooks

    else:
        raise ValueError(f"Unsupported component: {component}")


def remove_steering_layers(hooks):
    """Remove steering hooks from the model."""
    for hook in hooks:
        hook.remove()


# %% Test steering mechanism (uncomment to test interactively)
"""
def test_steering_mechanism():
    \"\"\"Test steering mechanism setup and removal.\"\"\"
    try:
        print("Testing steering mechanism...")
        
        # Load model and directions
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", torch_dtype="auto", device_map="cpu"
        )
        
        directions, _ = load_directions("directions", "Qwen/Qwen3-0.6B", "attn")
        
        # Test hook application
        hooks = apply_steering_layers(model, directions, alpha=0.1, component="attn")
        print(f"✓ Applied {len(hooks)} steering hooks")
        
        # Test hook removal
        remove_steering_layers(hooks)
        print("✓ Successfully removed steering hooks")
        
    except Exception as e:
        print(f"Steering mechanism test failed: {e}")

# Uncomment to test:
# test_steering_mechanism()
"""

# %% [markdown]
# ## Step 3: Single Example Generation with Steering
#
# Let's implement the core functionality for generating responses with steering applied.


# %% Single example generation with steering
def generate_single_with_steering(
    model,
    tokenizer,
    question,
    alpha=0.0,
    directions=None,
    component="attn",
    max_new_tokens=32768,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
):
    """Generate a response for a single question with steering applied."""
    if directions is None:
        raise ValueError("Directions must be provided for steering")

    prompt = f"Solve this math problem step by step, and put your final answer within \\boxed{{}}:\n{question}"

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    hooks = apply_steering_layers(model, directions, alpha, component)

    try:
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

        # Parse thinking content
        try:
            think_end_token = tokenizer.encode("</think>", add_special_tokens=False)[-1]
            think_end_index = (
                output_ids.index(think_end_token)
                if think_end_token in output_ids
                else -1
            )

            if think_end_index != -1:
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

        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return {"thinking": "", "response": content}

    finally:
        remove_steering_layers(hooks)
        del inputs, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# %% Test single steering (uncomment to test interactively)
"""
def test_single_steering():
    \"\"\"Test steering on a single example.\"\"\"
    try:
        print("Testing single example steering...")
        
        # Load model and directions
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", torch_dtype="auto", device_map="cpu"  # Use CPU for testing
        )
        
        directions, _ = load_directions("directions", "Qwen/Qwen3-0.6B", "attn")
        
        test_question = "If there are 5 apples and 3 are eaten, how many remain?"
        
        print(f"\\nTesting steering on: {test_question}")
        
        for alpha in [-0.1, 0.0, 0.1]:
            print(f"\\n--- Testing α = {alpha} ---")
            response = generate_single_with_steering(
                model, tokenizer, test_question,
                alpha=alpha, directions=directions, component="attn",
                max_new_tokens=1024  # Shorter for testing
            )
            
            thinking_length = len(response["thinking"].split())
            print(f"Thinking length: {thinking_length} words")
            print(f"Response: {response['response'][:100]}...")
            
    except Exception as e:
        print(f"Single steering test failed: {e}")

# Uncomment to test:
# test_single_steering()
"""

# %% [markdown]
# ## Step 4: Data Loading and Evaluation Functions
#
# Let's set up functions to load test data and evaluate the steering results.


# %% Data loading and evaluation functions
def load_test_data(num_samples=10, seed=42):
    """Load and prepare the GSM8K test dataset."""
    test_data = load_dataset("openai/gsm8k", "main", split="test[:200]")
    return test_data


def calculate_metrics(response, alpha):
    """Calculate metrics for a response."""
    thinking = response["thinking"]
    thinking_length = len(thinking.split()) if thinking else 0
    thinking_chars = len(thinking) if thinking else 0

    return {
        "alpha": alpha,
        "thinking_words": thinking_length,
        "thinking_chars": thinking_chars,
    }


def is_correct(response, answer):
    """Check if the response has the correct answer."""
    expected_answer = extract_answer_from_gsm8k(answer)
    full_response = response["thinking"] + " " + response["response"]

    if "\\boxed{" in full_response:
        try:
            boxed_content = full_response.split("\\boxed{")[1].split("}")[0].strip()
            return expected_answer in boxed_content
        except:
            pass

    return expected_answer in full_response


def extract_answer_from_gsm8k(answer_text):
    """Extract the numerical answer from a GSM8K answer string."""
    if "####" in answer_text:
        return answer_text.split("####")[1].strip()
    return answer_text.strip()


# %% Test data loading (uncomment to test interactively)
"""
def test_data_loading():
    \"\"\"Test loading GSM8K test data.\"\"\"
    try:
        print("Testing data loading...")
        test_data = load_test_data(num_samples=3, seed=42)
        print(f"✓ Loaded {len(test_data)} test examples")
        
        for i, example in enumerate(test_data[:2]):
            expected = extract_answer_from_gsm8k(example['answer'])
            print(f"\\nExample {i+1}:")
            print(f"  Question: {example['question'][:100]}...")
            print(f"  Expected answer: {expected}")
        
    except Exception as e:
        print(f"Data loading test failed: {e}")

# Uncomment to test:
# test_data_loading()
"""

# %% [markdown]
# ## Step 5: Visualization Functions
#
# Let's create functions to visualize the steering results.


# %% Visualization functions
def visualize_results(results, output_dir):
    """Visualize the relationship between steering strength, reasoning length, and accuracy."""
    alpha_values = sorted(set(r["alpha"] for r in results))
    avg_metrics = []

    for alpha in alpha_values:
        alpha_results = [r for r in results if r["alpha"] == alpha]
        avg_thinking_words = np.mean([r["thinking_words"] for r in alpha_results])
        avg_thinking_chars = np.mean([r["thinking_chars"] for r in alpha_results])
        accuracy = np.mean([r["is_correct"] for r in alpha_results])

        avg_metrics.append(
            {
                "alpha": alpha,
                "avg_thinking_words": avg_thinking_words,
                "avg_thinking_chars": avg_thinking_chars,
                "accuracy": accuracy,
            }
        )

    print("\nSteering Results Summary:")
    print("-" * 60)
    print(f"{'Alpha':^10} | {'Words':^12} | {'Characters':^12} | {'Accuracy':^10}")
    print("-" * 60)
    for m in avg_metrics:
        print(
            f"{m['alpha']:^10.2f} | {m['avg_thinking_words']:^12.1f} | {m['avg_thinking_chars']:^12.1f} | {m['accuracy']:^10.1%}"
        )

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlabel("Steering Strength (α)", fontsize=14)
    ax1.set_ylabel("Average Thinking Length (words)", color="blue", fontsize=14)
    ax1.plot(
        [m["alpha"] for m in avg_metrics],
        [m["avg_thinking_words"] for m in avg_metrics],
        "bo-",
        linewidth=3,
        markersize=8,
        label="Thinking Length",
    )
    ax1.tick_params(axis="y", labelcolor="blue", labelsize=12)

    for m in avg_metrics:
        ax1.annotate(
            f"{m['avg_thinking_words']:.1f}",
            (m["alpha"], m["avg_thinking_words"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=11,
            color="blue",
        )

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="red", fontsize=14)
    ax2.plot(
        [m["alpha"] for m in avg_metrics],
        [m["accuracy"] for m in avg_metrics],
        "ro-",
        linewidth=3,
        markersize=8,
        label="Accuracy",
    )
    ax2.tick_params(axis="y", labelcolor="red", labelsize=12)
    ax2.set_ylim([0, 1.1])

    for m in avg_metrics:
        ax2.annotate(
            f"{m['accuracy']:.1%}",
            (m["alpha"], m["accuracy"]),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=11,
            color="red",
        )

    plt.title(
        "Effect of Reasoning Length Steering on Thinking Length and Accuracy",
        fontsize=16,
    )
    fig.tight_layout()

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)

    ax1.grid(True, linestyle="--", alpha=0.7)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "steering_effect.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")

    plt.show()

    return avg_metrics


# %% Test visualization (uncomment to test interactively)
"""
def test_visualization():
    \"\"\"Test visualization with dummy data.\"\"\"
    try:
        print("Testing visualization with dummy data...")
        
        # Create dummy results
        dummy_results = []
        for alpha in [-0.1, 0.0, 0.1]:
            for i in range(5):
                dummy_results.append({
                    "alpha": alpha,
                    "thinking_words": 50 + alpha * 100 + np.random.normal(0, 10),
                    "thinking_chars": 300 + alpha * 600 + np.random.normal(0, 50),
                    "is_correct": np.random.random() > 0.3
                })
        
        avg_metrics = visualize_results(dummy_results, "test_output")
        print("✓ Visualization test completed")
        
    except Exception as e:
        print(f"Visualization test failed: {e}")

# Uncomment to test:
# test_visualization()
"""

# %% [markdown]
# ## Step 6: Batch Processing
#
# For efficiency, we'll implement batch processing capabilities.


# %% Batch processing functions
def generate_batch_with_steering(
    model,
    tokenizer,
    questions,
    alpha=0.0,
    directions=None,
    component="attn",
    max_new_tokens=32768,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    batch_size=4,
):
    """Generate responses for a batch of questions with steering applied."""
    if directions is None:
        raise ValueError("Directions must be provided for steering")

    logger = get_logger()
    all_responses = []

    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}, questions {i+1}-{min(i+batch_size, len(questions))}"
        )

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            prompts = []
            for question in batch_questions:
                prompt = f"Solve this math problem step by step, and put your final answer within \\boxed{{}}:\n{question}"
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                prompts.append(text)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(model.device)

            hooks = apply_steering_layers(model, directions, alpha, component)

            try:
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

                batch_responses = []
                for j, (input_ids, generated) in enumerate(
                    zip(inputs.input_ids, generated_ids)
                ):
                    output_ids = generated[len(input_ids) :].tolist()

                    try:
                        think_end_token = tokenizer.encode(
                            "</think>", add_special_tokens=False
                        )[-1]
                        think_end_index = (
                            output_ids.index(think_end_token)
                            if think_end_token in output_ids
                            else -1
                        )

                        if think_end_index != -1:
                            thinking_content = tokenizer.decode(
                                output_ids[:think_end_index], skip_special_tokens=True
                            ).strip()
                            if thinking_content.startswith("<think>"):
                                thinking_content = thinking_content[
                                    len("<think>") :
                                ].strip()

                            content = tokenizer.decode(
                                output_ids[think_end_index + 1 :],
                                skip_special_tokens=True,
                            ).strip()
                            response = {
                                "thinking": thinking_content,
                                "response": content,
                            }
                        else:
                            content = tokenizer.decode(
                                output_ids, skip_special_tokens=True
                            ).strip()
                            response = {"thinking": "", "response": content}

                        batch_responses.append(response)

                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing response for question {j}: {e}")
                        content = tokenizer.decode(
                            output_ids, skip_special_tokens=True
                        ).strip()
                        batch_responses.append({"thinking": "", "response": content})

                all_responses.extend(batch_responses)
                logger.info(f"Successfully processed batch {i//batch_size + 1}")

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM error in batch {i//batch_size + 1}: {e}")
                logger.info("Falling back to single-question processing for this batch")
                batch_responses = []
                for j, question in enumerate(batch_questions):
                    try:
                        torch.cuda.empty_cache()
                        single_response = generate_single_with_steering(
                            model,
                            tokenizer,
                            question,
                            alpha,
                            directions,
                            component,
                            max_new_tokens,
                            temperature,
                            top_p,
                            top_k,
                        )
                        batch_responses.append(single_response)
                        logger.info(
                            f"Successfully processed question {i+j+1} individually"
                        )
                    except torch.cuda.OutOfMemoryError:
                        logger.error(f"CUDA OOM even with single question {i+j+1}")
                        batch_responses.append(
                            {"thinking": "", "response": "Error: Out of memory"}
                        )

                all_responses.extend(batch_responses)

            finally:
                remove_steering_layers(hooks)

        except Exception as e:
            logger.error(f"Error setting up batch {i//batch_size + 1}: {e}")
            empty_responses = [
                {"thinking": "", "response": f"Setup error: {str(e)}"}
                for _ in batch_questions
            ]
            all_responses.extend(empty_responses)

        finally:
            if "inputs" in locals():
                del inputs
            if "generated_ids" in locals():
                del generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_responses


# %% [markdown]
# ## Step 7: Complete Evaluation Pipeline
#
# This combines everything into a complete evaluation pipeline.


# %% Efficient evaluation functions
def evaluate_steering_batch_efficient(
    model,
    tokenizer,
    test_data,
    directions,
    alpha_values,
    component="attn",
    batch_size=4,
    max_new_tokens=32768,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    output_dir=None,
    model_short_name=None,
    resume_from_existing=True,
):
    """Efficiently evaluate steering across multiple alpha values using batch processing."""
    logger = get_logger()
    all_results = []

    questions = [example["question"] for example in test_data]
    answers = [example["answer"] for example in test_data]

    if output_dir and model_short_name:
        intermediate_file = os.path.join(
            output_dir, f"{model_short_name}_{component}_steering_intermediate.json"
        )

        existing_alphas = set()
        if resume_from_existing and os.path.exists(intermediate_file):
            try:
                with open(intermediate_file, "r") as f:
                    existing_results = json.load(f)
                    all_results.extend(existing_results)
                    existing_alphas = set(r["alpha"] for r in existing_results)
                    logger.info(
                        f"Resuming from existing results. Found {len(existing_results)} previous results for alphas: {sorted(existing_alphas)}"
                    )
            except Exception as e:
                logger.warning(f"Could not load existing intermediate results: {e}")
    else:
        intermediate_file = None

    for alpha in tqdm(alpha_values, desc="Testing steering strengths"):
        if alpha in existing_alphas:
            logger.info(f"Skipping α = {alpha} (already processed)")
            continue

        logger.info(f"Processing α = {alpha}")

        try:
            responses = generate_batch_with_steering(
                model,
                tokenizer,
                questions,
                alpha=alpha,
                directions=directions,
                component=component,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                batch_size=batch_size,
            )

            alpha_results = []
            for i, (response, answer) in enumerate(zip(responses, answers)):
                metrics = calculate_metrics(response, alpha)
                metrics["question_id"] = i
                metrics["is_correct"] = is_correct(response, answer)
                metrics["response"] = response
                alpha_results.append(metrics)
                all_results.append(metrics)

            logger.info(f"Completed α = {alpha}, processed {len(responses)} examples")

            if intermediate_file:
                try:
                    serializable_results = []
                    for r in all_results:
                        result_copy = r.copy()
                        result_copy["response"] = {
                            "thinking": r["response"]["thinking"],
                            "response": r["response"]["response"],
                        }
                        serializable_results.append(result_copy)

                    with open(intermediate_file, "w") as f:
                        json.dump(serializable_results, f, indent=2)
                    logger.info(f"Saved intermediate results to {intermediate_file}")
                except Exception as e:
                    logger.warning(f"Failed to save intermediate results: {e}")

        except Exception as e:
            logger.error(f"Critical error processing α = {alpha}: {e}")
            for i, answer in enumerate(answers):
                metrics = {
                    "alpha": alpha,
                    "thinking_words": 0,
                    "thinking_chars": 0,
                    "question_id": i,
                    "is_correct": False,
                    "response": {
                        "thinking": "",
                        "response": f"Error processing α={alpha}: {str(e)}",
                    },
                }
                all_results.append(metrics)

    results_file = os.path.join(
        output_dir,
        f"{model_short_name}_{component}_steering_results_efficient.json",
    )

    serializable_results = []
    for r in all_results:
        result_copy = r.copy()
        result_copy["response"] = {
            "thinking": r["response"]["thinking"],
            "response": r["response"]["response"],
        }
        serializable_results.append(result_copy)

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    if intermediate_file and os.path.exists(intermediate_file):
        try:
            os.remove(intermediate_file)
            logger.info(f"Cleaned up intermediate file: {intermediate_file}")
        except Exception as e:
            logger.warning(f"Could not remove intermediate file: {e}")

    logger.info("Creating visualizations...")
    avg_metrics = visualize_results(all_results, output_dir)

    logger.info(f"Efficient processing complete! Results saved to {results_file}")
    logger.info(f"Processed {len(all_results)} total examples")
    logger.info(f"Final batch size used: {batch_size}")

    return all_results, avg_metrics


# %% [markdown]
# ## Step 8: Main Processing Functions
#
# These tie everything together into complete pipelines.


# %% Memory efficient main function
def memory_efficient_main(args):
    """Memory-efficient version of the main function with batch processing."""
    logger = setup_logging("steer_reasoning_length_efficient")

    model_short_name = args.model.split("/")[-1]

    if args.directions_file:
        directions_file = args.directions_file
    else:
        directions_file = os.path.join(
            args.directions_dir,
            f"{model_short_name}_thinking_length_direction_gsm8k_{args.component}.pt",
        )

    if not os.path.exists(directions_file):
        logger.error(f"Directions file {directions_file} not found.")
        return None

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device
    )

    logger.info(f"Loading directions from {directions_file}...")
    directions = torch.load(directions_file, map_location=args.device)
    if directions.dim() > 1:
        directions = directions.contiguous()
    logger.info(f"Loaded directions tensor with shape: {directions.shape}")

    logger.info(f"Loading test data...")
    test_data = load_test_data(args.num_samples, args.seed)
    logger.info(f"Loaded {len(test_data)} test examples")

    if hasattr(args, "batch_size"):
        initial_batch_size = args.batch_size
    else:
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            if total_memory > 20e9:
                initial_batch_size = 8
            elif total_memory > 10e9:
                initial_batch_size = 4
            else:
                initial_batch_size = 2
        else:
            initial_batch_size = 2

    logger.info(f"Starting with batch size: {initial_batch_size}")

    current_batch_size = initial_batch_size
    min_batch_size = 1

    logger.info("Starting efficient batch evaluation...")

    resume_from_existing = args.resume and not args.no_resume
    if args.no_resume:
        logger.info("Starting fresh (ignoring existing intermediate results)")
    elif resume_from_existing:
        logger.info("Will resume from existing intermediate results if available")

    while current_batch_size >= min_batch_size:
        try:
            all_results, avg_metrics = evaluate_steering_batch_efficient(
                model,
                tokenizer,
                test_data,
                directions,
                args.direction_weights,
                component=args.component,
                batch_size=current_batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                output_dir=args.output_dir,
                model_short_name=model_short_name,
                resume_from_existing=resume_from_existing,
            )

            logger.info(
                f"Successfully completed evaluation with batch size {current_batch_size}"
            )
            break

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"OOM with batch size {current_batch_size}: {e}")
            current_batch_size = max(1, current_batch_size // 2)
            logger.info(f"Reducing batch size to {current_batch_size} and retrying...")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Non-OOM error during evaluation: {e}")
            return None, None

    if current_batch_size < min_batch_size:
        logger.error("Could not complete evaluation even with minimum batch size")
        return None, None

    return all_results, avg_metrics


# %% Standard main function
def main(args):
    """Main function to run steering experiments."""
    logger = setup_logging("steer_reasoning_length")

    model_short_name = args.model.split("/")[-1]

    if args.directions_file:
        directions_file = args.directions_file
    else:
        directions_file = os.path.join(
            args.directions_dir,
            f"{model_short_name}_thinking_length_direction_gsm8k_{args.component}.pt",
        )

    if not os.path.exists(directions_file):
        logger.error(f"Directions file {directions_file} not found.")
        logger.error(f"Please run extract_reasoning_length_direction.py first.")
        return None

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    logger.info(f"Loading directions from {directions_file}...")
    directions = torch.load(directions_file)
    logger.info(f"Loaded directions tensor with shape: {directions.shape}")

    logger.info(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device
    )
    logger.info(f"Using device: {args.device}")

    logger.info(f"Loading test data...")
    test_data = load_test_data(args.num_samples, args.seed)
    logger.info(f"Loaded {len(test_data)} test examples")

    all_results = []

    for alpha in args.direction_weights:
        logger.info(f"Testing with steering strength α = {alpha}...")

        for i, example in enumerate(
            tqdm(test_data, desc=f"Processing examples (α = {alpha})")
        ):
            question = example["question"]
            answer = example["answer"]

            response = generate_single_with_steering(
                model,
                tokenizer,
                question,
                alpha=alpha,
                directions=directions,
                component=args.component,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            metrics = calculate_metrics(response, alpha)
            metrics["question_id"] = i
            metrics["is_correct"] = is_correct(response, answer)
            metrics["response"] = response

            all_results.append(metrics)

    results_file = os.path.join(
        args.output_dir, f"{model_short_name}_{args.component}_steering_results.json"
    )
    with open(results_file, "w") as f:
        serializable_results = []
        for r in all_results:
            result_copy = r.copy()
            result_copy["response"] = {
                "thinking": r["response"]["thinking"],
                "response": r["response"]["response"],
            }
            serializable_results.append(result_copy)

        json.dump(serializable_results, f, indent=2)

    logger.info("Creating visualizations...")
    avg_metrics = visualize_results(all_results, args.output_dir)

    logger.info(f"Done! Results saved to {results_file}")
    logger.info(f"Processed {len(all_results)} total examples")

    return all_results, avg_metrics


# %% [markdown]
# ## Summary and Next Steps
#
# Congratulations! You've successfully implemented reasoning length steering.
#
# **What we accomplished:**
# 1. Loaded pre-extracted reasoning length direction vectors
# 2. Set up steering mechanisms using forward hooks
# 3. Generated responses with different steering strengths
# 4. Evaluated the effect on reasoning length and accuracy
# 5. Visualized the relationship between steering and performance
#
# **Key insights:**
# - Positive α values typically increase reasoning length
# - Negative α values typically decrease reasoning length
# - There's often a trade-off between reasoning length and accuracy
# - The optimal steering strength depends on the specific task
#
# **Next steps:**
# - Experiment with different steering strengths and components
# - Try steering on different types of reasoning tasks
# - Analyze the relationship between reasoning quality and length
# - Investigate task-specific optimal steering parameters

# %% Script execution
if __name__ == "__main__":
    args = parse_args()

    if args.use_efficient_mode:
        print("Using efficient batch processing mode...")
        results, avg_metrics = memory_efficient_main(args)
    else:
        print("Using standard (single-sample) processing mode...")
        results, avg_metrics = main(args)
