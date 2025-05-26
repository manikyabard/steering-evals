#!/usr/bin/env python3
"""
Improved Reasoning Length Direction Extraction

This script extracts reasoning length directions following the ThinkEdit approach more closely:
1. Uses fixed sample sizes (100 examples each) instead of percentages
2. Computes thinking lengths using word counts
3. Better sample selection and direction computation
4. More thorough debugging and validation

Based on: https://github.com/Trustworthy-ML-Lab/ThinkEdit/blob/main/extract_thinking_length_direction_gsm8k_attn.py
"""

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract reasoning length direction from model (improved version)"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--responses_file",
        type=str,
        default="responses/Qwen3-0.6B_gsm8k_responses.json",
        help="Path to the responses JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="directions",
        help="Directory to save extracted directions",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="attn",
        choices=["attn", "mlp", "both"],
        help="Which component to extract directions for",
    )
    parser.add_argument(
        "--n_short",
        type=int,
        default=100,
        help="Number of short thinking examples to use",
    )
    parser.add_argument(
        "--n_long",
        type=int,
        default=100,
        help="Number of long thinking examples to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation",
    )
    parser.add_argument(
        "--recompute_lengths",
        action="store_true",
        help="Recompute thinking lengths (useful if not already computed)",
    )
    return parser.parse_args()

def count_thinking_words(thinking_text):
    """Count words in thinking text, similar to ThinkEdit approach."""
    if not thinking_text or thinking_text.strip() == "":
        return 0
    
    # Split by whitespace and count non-empty tokens
    words = thinking_text.strip().split()
    # Filter out very short tokens that might be punctuation
    meaningful_words = [w for w in words if len(w.strip()) > 0]
    return len(meaningful_words)

def compute_thinking_lengths(responses):
    """Compute thinking lengths for all responses."""
    print("Computing thinking lengths...")
    
    for item in tqdm(responses, desc="Computing thinking lengths"):
        # Handle both formats: direct 'thinking' field or nested 'with_thinking'
        if "thinking" in item and item["thinking"]:
            thinking_text = item["thinking"]
            item["thinking_length"] = count_thinking_words(thinking_text)
        elif "with_thinking" in item and "thinking" in item["with_thinking"]:
            thinking_text = item["with_thinking"]["thinking"]
            item["thinking_length"] = count_thinking_words(thinking_text)
        else:
            item["thinking_length"] = -1  # Invalid/missing thinking
    
    return responses

def select_examples_by_length(responses, n_short=100, n_long=100):
    """Select examples based on thinking length, similar to ThinkEdit approach."""
    
    # Filter valid responses
    valid_responses = [
        item for item in responses 
        if "thinking_length" in item and item["thinking_length"] > 0
    ]
    
    print(f"Found {len(valid_responses)} responses with valid thinking")
    
    if len(valid_responses) < n_short + n_long:
        print(f"Warning: Only {len(valid_responses)} valid responses, but need {n_short + n_long}")
        print("Will use all available responses")
    
    # Sort by thinking length
    valid_responses.sort(key=lambda x: x["thinking_length"])
    
    # Print distribution info
    thinking_lengths = [item["thinking_length"] for item in valid_responses]
    print(f"Thinking length distribution:")
    print(f"  Min: {min(thinking_lengths)} words")
    print(f"  Max: {max(thinking_lengths)} words")
    print(f"  Mean: {np.mean(thinking_lengths):.1f} words")
    print(f"  Median: {np.median(thinking_lengths):.1f} words")
    
    # Select examples: first n_short for short, last n_long for long
    short_examples = valid_responses[:n_short]
    long_examples = valid_responses[-n_long:]
    
    # Print selected examples stats
    short_lengths = [item["thinking_length"] for item in short_examples]
    long_lengths = [item["thinking_length"] for item in long_examples]
    
    print(f"\nSelected examples:")
    print(f"  Short thinking: {len(short_examples)} examples")
    print(f"    Length range: {min(short_lengths)} - {max(short_lengths)} words")
    print(f"    Average: {np.mean(short_lengths):.1f} words")
    
    print(f"  Long thinking: {len(long_examples)} examples")
    print(f"    Length range: {min(long_lengths)} - {max(long_lengths)} words") 
    print(f"    Average: {np.mean(long_lengths):.1f} words")
    
    print(f"  Separation: {min(long_lengths) - max(short_lengths)} words gap")
    
    return short_examples, long_examples

class ImprovedActivationExtractor:
    """Improved activation extractor following ThinkEdit approach."""
    
    def __init__(self, model, component="attn"):
        self.model = model
        self.component = component
        self.activations = {}
        self.hooks = []
        self.setup_hooks()
    
    def setup_hooks(self):
        """Setup hooks for the specified component."""
        for i, layer in enumerate(self.model.model.layers):
            if self.component in ["attn", "both"]:
                # Hook for attention output
                hook = layer.self_attn.register_forward_hook(
                    lambda module, input, output, layer_idx=i: 
                    self._save_activation(output, f"attn_layer_{layer_idx}")
                )
                self.hooks.append(hook)
            
            if self.component in ["mlp", "both"]:
                # Hook for MLP output
                hook = layer.mlp.register_forward_hook(
                    lambda module, input, output, layer_idx=i: 
                    self._save_activation(output, f"mlp_layer_{layer_idx}")
                )
                self.hooks.append(hook)
    
    def _save_activation(self, output, layer_name):
        """Save activation output."""
        if isinstance(output, tuple):
            # For attention, take the first element (attention output)
            activation = output[0]
        else:
            activation = output
        
        # Store on CPU to save memory
        self.activations[layer_name] = activation.detach().cpu()
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def get_activations_for_example(model, tokenizer, extractor, question, thinking):
    """Get activations for a single example."""
    
    # Create the prompt in the format used during training
    prompt = f"<|User|>{question}<|Assistant|>{thinking}"
    
    # Tokenize the full prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Get token positions for the thinking part
    question_prompt = f"<|User|>{question}<|Assistant|>"
    question_tokens = tokenizer(question_prompt, return_tensors="pt")
    thinking_start = question_tokens["input_ids"].shape[1] - 1  # -1 for 0-indexing
    thinking_end = inputs["input_ids"].shape[1] - 1
    
    # Clear previous activations
    extractor.clear_activations()
    
    # Forward pass to collect activations
    with torch.no_grad():
        model(**inputs)
    
    # Extract activations for the thinking portion and average over tokens
    layer_activations = {}
    for layer_name, activation in extractor.activations.items():
        # activation shape: [batch_size, seq_len, hidden_size]
        thinking_activation = activation[:, thinking_start:thinking_end, :]
        
        # Average over the thinking tokens
        if thinking_activation.shape[1] > 0:
            mean_activation = thinking_activation.mean(dim=1)  # [batch_size, hidden_size]
            layer_activations[layer_name] = mean_activation.squeeze(0)  # [hidden_size]
        else:
            # Handle edge case where thinking is empty
            layer_activations[layer_name] = torch.zeros(activation.shape[-1])
    
    return layer_activations

def extract_directions(model, tokenizer, short_examples, long_examples, component="attn"):
    """Extract direction vectors by contrasting short vs long thinking examples."""
    
    print(f"Extracting directions for {component} component...")
    
    # Setup activation extractor
    extractor = ImprovedActivationExtractor(model, component)
    
    # Storage for activations
    short_activations = defaultdict(list)
    long_activations = defaultdict(list)
    
    # Process short thinking examples
    print(f"Processing {len(short_examples)} short thinking examples...")
    for i, example in enumerate(tqdm(short_examples)):
        try:
            question = example["question"]
            # Handle both formats: direct 'thinking' field or nested 'with_thinking'
            if "thinking" in example and example["thinking"]:
                thinking = example["thinking"]
            elif "with_thinking" in example and "thinking" in example["with_thinking"]:
                thinking = example["with_thinking"]["thinking"]
            else:
                continue  # Skip if no thinking found
            
            # Get activations
            activations = get_activations_for_example(model, tokenizer, extractor, question, thinking)
            
            # Store activations
            for layer_name, activation in activations.items():
                short_activations[layer_name].append(activation)
            
            # Memory cleanup
            if i % 20 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing short example {i}: {e}")
            continue
    
    # Process long thinking examples  
    print(f"Processing {len(long_examples)} long thinking examples...")
    for i, example in enumerate(tqdm(long_examples)):
        try:
            question = example["question"]
            # Handle both formats: direct 'thinking' field or nested 'with_thinking'
            if "thinking" in example and example["thinking"]:
                thinking = example["thinking"]
            elif "with_thinking" in example and "thinking" in example["with_thinking"]:
                thinking = example["with_thinking"]["thinking"]
            else:
                continue  # Skip if no thinking found
            
            # Get activations
            activations = get_activations_for_example(model, tokenizer, extractor, question, thinking)
            
            # Store activations
            for layer_name, activation in activations.items():
                long_activations[layer_name].append(activation)
            
            # Memory cleanup
            if i % 20 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing long example {i}: {e}")
            continue
    
    # Compute mean activations and directions
    print("Computing direction vectors...")
    directions = {}
    
    for layer_name in short_activations:
        if layer_name in long_activations and len(short_activations[layer_name]) > 0 and len(long_activations[layer_name]) > 0:
            
            # Stack and compute means
            short_stack = torch.stack(short_activations[layer_name])
            long_stack = torch.stack(long_activations[layer_name])
            
            short_mean = short_stack.mean(dim=0)
            long_mean = long_stack.mean(dim=0)
            
            # Direction: long - short (positive direction should increase thinking length)
            direction = long_mean - short_mean
            
            # Normalize the direction
            direction_norm = torch.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
            
            directions[layer_name] = direction
            
            # Debug info for first few layers
            layer_idx = int(layer_name.split('_')[-1]) if '_' in layer_name else 0
            if layer_idx < 3:
                print(f"  {layer_name}: short_mean={short_mean.mean().item():.6f}, "
                      f"long_mean={long_mean.mean().item():.6f}, "
                      f"direction_mean={direction.mean().item():.6f}, "
                      f"norm={direction_norm.item():.6f}")
    
    # Cleanup
    extractor.remove_hooks()
    del short_activations, long_activations
    torch.cuda.empty_cache()
    
    print(f"Extracted directions for {len(directions)} layers")
    return directions

def save_directions(directions, output_dir, model_name, component):
    """Save the extracted directions."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_short_name = model_name.split("/")[-1]
    filename = f"{model_short_name}_reasoning_length_direction_gsm8k_{component}_improved.pt"
    filepath = os.path.join(output_dir, filename)
    
    torch.save(directions, filepath)
    print(f"Directions saved to: {filepath}")
    
    return filepath

def visualize_directions(directions, output_dir, model_name, component):
    """Visualize direction magnitudes across layers."""
    layer_names = sorted(directions.keys(), key=lambda x: int(x.split('_')[-1]) if '_' in x else 0)
    layer_indices = [int(name.split('_')[-1]) if '_' in name else 0 for name in layer_names]
    direction_norms = [torch.norm(directions[name]).item() for name in layer_names]
    
    plt.figure(figsize=(12, 6))
    plt.plot(layer_indices, direction_norms, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Layer Index')
    plt.ylabel('Direction Magnitude')
    plt.title(f'Reasoning Length Direction Magnitudes ({component.upper()}) - {model_name}')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    model_short_name = model_name.split("/")[-1]
    plot_filename = f"{model_short_name}_improved_{component}_direction_magnitudes.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_path}")
    
    plt.show()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("IMPROVED REASONING LENGTH DIRECTION EXTRACTION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Component: {args.component}")
    print(f"Short examples: {args.n_short}")
    print(f"Long examples: {args.n_long}")
    print(f"Device: {args.device}")
    
    # Load responses
    print(f"\nLoading responses from: {args.responses_file}")
    with open(args.responses_file, 'r') as f:
        responses = json.load(f)
    print(f"Loaded {len(responses)} responses")
    
    # Compute thinking lengths if needed
    if args.recompute_lengths or not any('thinking_length' in item for item in responses):
        print("Computing thinking lengths...")
        responses = compute_thinking_lengths(responses)
    
    # Select examples
    short_examples, long_examples = select_examples_by_length(responses, args.n_short, args.n_long)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype="auto", 
        device_map=args.device
    )
    model.eval()
    
    # Extract directions
    if args.component == "both":
        components = ["attn", "mlp"] 
    else:
        components = [args.component]
    
    for comp in components:
        print(f"\n{'='*40}")
        print(f"Processing {comp.upper()} component")
        print(f"{'='*40}")
        
        directions = extract_directions(model, tokenizer, short_examples, long_examples, comp)
        
        # Save directions
        save_directions(directions, args.output_dir, args.model, comp)
        
        # Visualize directions
        visualize_directions(directions, args.output_dir, args.model, comp)
    
    print(f"\n{'='*60}")
    print("DIRECTION EXTRACTION COMPLETE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 