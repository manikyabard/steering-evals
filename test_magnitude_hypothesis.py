#!/usr/bin/env python3
"""
Test the Magnitude-Based Steering Hypothesis

This script systematically tests whether the reasoning length steering follows
a magnitude-based pattern: reasoning_length = f(|α|) rather than directional control.

Hypothesis: Both positive and negative alpha values of the same magnitude should 
produce similar reasoning lengths, and larger |α| should produce longer reasoning.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import seaborn as sns

# Import steering functions from our existing script
import sys
sys.path.append('.')
from steer_reasoning_length import (
    apply_steering_layers, 
    remove_steering_layers, 
    count_thinking_length
)

def test_magnitude_hypothesis():
    """Test the magnitude-based steering hypothesis with systematic experiments."""
    
    # Experimental parameters
    model_name = "Qwen/Qwen3-0.6B"
    device = "cuda:0"
    num_samples_per_alpha = 5  # Multiple samples for statistics
    
    # Test symmetric alpha values to test magnitude hypothesis
    alpha_values = [
        -0.12, -0.08, -0.06, -0.04, -0.02, 0.0, 
        0.02, 0.04, 0.06, 0.08, 0.12
    ]
    
    print("Testing Magnitude-Based Steering Hypothesis")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Testing alpha values: {alpha_values}")
    print(f"Samples per alpha: {num_samples_per_alpha}")
    
    # Load model and tokenizer
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map=device
    )
    
    # Load directions
    model_short_name = model_name.split("/")[-1]
    directions_file = f"directions/{model_short_name}_reasoning_length_directions.pt"
    
    if not os.path.exists(directions_file):
        print(f"Error: Directions file {directions_file} not found!")
        return None
        
    directions = torch.load(directions_file)
    print(f"Loaded directions for {len(directions)} layers")
    
    # Test questions (simple ones for consistency)
    test_questions = [
        "If there are 5 apples and 3 are eaten, how many remain?",
        "What is 12 + 8?",
        "A box has 20 marbles. If 5 are red and the rest are blue, how many blue marbles are there?",
        "Sarah has 15 stickers. She gives 4 to her friend. How many does she have left?",
        "If a pizza is cut into 8 slices and 3 slices are eaten, how many slices remain?"
    ]
    
    # Store results
    results = []
    
    # Test each alpha value
    for alpha in tqdm(alpha_values, desc="Testing alpha values"):
        alpha_results = []
        
        # Test multiple samples for this alpha
        for sample_idx in range(num_samples_per_alpha):
            question = test_questions[sample_idx % len(test_questions)]
            
            # Apply steering
            steering_layers = apply_steering_layers(
                model, directions, alpha=alpha, component="attn"
            )
            
            # Generate response
            messages = [{"role": "user", "content": question}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=1000,  # Shorter for faster testing
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode and measure thinking length
            response = tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            thinking_length = count_thinking_length(response)
            
            alpha_results.append({
                'alpha': alpha,
                'abs_alpha': abs(alpha),
                'thinking_length': thinking_length,
                'question': question,
                'sample_idx': sample_idx
            })
            
            # Remove steering
            remove_steering_layers(steering_layers)
            
            # Clear memory
            del output, response
            torch.cuda.empty_cache()
        
        results.extend(alpha_results)
        
        # Print interim results
        avg_length = np.mean([r['thinking_length'] for r in alpha_results])
        std_length = np.std([r['thinking_length'] for r in alpha_results])
        print(f"α = {alpha:+.2f}: {avg_length:.1f} ± {std_length:.1f} words")
    
    return results

def analyze_results(results):
    """Analyze the results to test the magnitude hypothesis."""
    
    print("\n" + "=" * 60)
    print("ANALYSIS: Testing Magnitude-Based Hypothesis")
    print("=" * 60)
    
    # Group results by alpha value
    alpha_stats = {}
    for result in results:
        alpha = result['alpha']
        if alpha not in alpha_stats:
            alpha_stats[alpha] = []
        alpha_stats[alpha].append(result['thinking_length'])
    
    # Calculate statistics for each alpha
    analysis_data = []
    for alpha in sorted(alpha_stats.keys()):
        lengths = alpha_stats[alpha]
        stats = {
            'alpha': alpha,
            'abs_alpha': abs(alpha),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'sample_count': len(lengths)
        }
        analysis_data.append(stats)
    
    # Print detailed results table
    print("\nDetailed Results:")
    print("-" * 80)
    print(f"{'Alpha':^8} | {'|Alpha|':^8} | {'Mean':^8} | {'Std':^8} | {'Min':^6} | {'Max':^6} | {'N':^3}")
    print("-" * 80)
    
    for stats in analysis_data:
        print(f"{stats['alpha']:^+8.2f} | {stats['abs_alpha']:^8.2f} | "
              f"{stats['mean_length']:^8.1f} | {stats['std_length']:^8.1f} | "
              f"{stats['min_length']:^6.0f} | {stats['max_length']:^6.0f} | "
              f"{stats['sample_count']:^3.0f}")
    
    # Test magnitude hypothesis: compare pairs with same |α|
    print("\n" + "-" * 60)
    print("MAGNITUDE HYPOTHESIS TEST")
    print("-" * 60)
    
    magnitude_pairs = {}
    for stats in analysis_data:
        abs_alpha = stats['abs_alpha']
        if abs_alpha > 0:  # Skip α = 0
            if abs_alpha not in magnitude_pairs:
                magnitude_pairs[abs_alpha] = []
            magnitude_pairs[abs_alpha].append(stats)
    
    hypothesis_supported = True
    for abs_alpha in sorted(magnitude_pairs.keys()):
        pairs = magnitude_pairs[abs_alpha]
        if len(pairs) == 2:  # We have both positive and negative
            neg_stats = next(p for p in pairs if p['alpha'] < 0)
            pos_stats = next(p for p in pairs if p['alpha'] > 0)
            
            # Calculate difference in means
            diff = abs(neg_stats['mean_length'] - pos_stats['mean_length'])
            combined_std = np.sqrt(neg_stats['std_length']**2 + pos_stats['std_length']**2)
            
            # Simple significance test (difference > 1 standard deviation)
            is_similar = diff < combined_std
            
            print(f"|α| = {abs_alpha:.2f}: "
                  f"α = {neg_stats['alpha']:+.2f} → {neg_stats['mean_length']:.1f} words, "
                  f"α = {pos_stats['alpha']:+.2f} → {pos_stats['mean_length']:.1f} words "
                  f"(diff = {diff:.1f}, {'✓' if is_similar else '✗'})")
            
            if not is_similar:
                hypothesis_supported = False
    
    # Test if larger |α| leads to longer reasoning
    print("\n" + "-" * 60)
    print("MAGNITUDE-LENGTH RELATIONSHIP TEST")
    print("-" * 60)
    
    # Sort by absolute alpha and check if length generally increases
    analysis_data.sort(key=lambda x: x['abs_alpha'])
    
    length_increases = 0
    total_comparisons = 0
    
    for i in range(1, len(analysis_data)):
        if analysis_data[i]['abs_alpha'] > analysis_data[i-1]['abs_alpha']:
            if analysis_data[i]['mean_length'] > analysis_data[i-1]['mean_length']:
                length_increases += 1
            total_comparisons += 1
    
    increase_rate = length_increases / total_comparisons if total_comparisons > 0 else 0
    
    print(f"Larger |α| leads to longer reasoning in {length_increases}/{total_comparisons} "
          f"({increase_rate:.1%}) of cases")
    
    # Overall hypothesis conclusion
    print("\n" + "=" * 60)
    print("HYPOTHESIS CONCLUSION")
    print("=" * 60)
    
    if hypothesis_supported and increase_rate > 0.6:
        print("✅ MAGNITUDE HYPOTHESIS SUPPORTED")
        print("- Symmetric alpha values produce similar reasoning lengths")
        print("- Larger |α| generally produces longer reasoning")
        print("- This suggests magnitude-based rather than directional steering")
    else:
        print("❌ MAGNITUDE HYPOTHESIS NOT CLEARLY SUPPORTED")
        print("- Results may indicate directional steering or more complex behavior")
        print("- Further investigation needed")
    
    return analysis_data

def create_visualizations(results):
    """Create visualizations to illustrate the magnitude relationship."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Magnitude-Based Steering Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data
    alphas = [r['alpha'] for r in results]
    abs_alphas = [r['abs_alpha'] for r in results]
    lengths = [r['thinking_length'] for r in results]
    
    # Plot 1: Alpha vs Thinking Length (raw relationship)
    ax1.scatter(alphas, lengths, alpha=0.6, s=50)
    ax1.set_xlabel('Alpha Value (α)')
    ax1.set_ylabel('Thinking Length (words)')
    ax1.set_title('Raw Alpha vs Thinking Length')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: |Alpha| vs Thinking Length (magnitude relationship)
    ax2.scatter(abs_alphas, lengths, alpha=0.6, s=50, color='orange')
    
    # Add trend line
    z = np.polyfit(abs_alphas, lengths, 1)
    p = np.poly1d(z)
    ax2.plot(sorted(set(abs_alphas)), p(sorted(set(abs_alphas))), "r--", alpha=0.8)
    
    ax2.set_xlabel('|Alpha| (Magnitude)')
    ax2.set_ylabel('Thinking Length (words)')
    ax2.set_title('Magnitude vs Thinking Length')
    ax2.grid(True, alpha=0.3)
    
    # Calculate correlation
    correlation = np.corrcoef(abs_alphas, lengths)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Box plot by alpha value
    alpha_values = sorted(set(alphas))
    length_by_alpha = [[r['thinking_length'] for r in results if r['alpha'] == a] for a in alpha_values]
    
    ax3.boxplot(length_by_alpha, labels=[f'{a:+.2f}' for a in alpha_values])
    ax3.set_xlabel('Alpha Value')
    ax3.set_ylabel('Thinking Length (words)')
    ax3.set_title('Distribution by Alpha Value')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Symmetric comparison (positive vs negative alphas)
    # Group by magnitude
    magnitude_groups = {}
    for r in results:
        abs_alpha = r['abs_alpha']
        if abs_alpha > 0:  # Skip zero
            if abs_alpha not in magnitude_groups:
                magnitude_groups[abs_alpha] = {'positive': [], 'negative': []}
            if r['alpha'] > 0:
                magnitude_groups[abs_alpha]['positive'].append(r['thinking_length'])
            else:
                magnitude_groups[abs_alpha]['negative'].append(r['thinking_length'])
    
    magnitudes = sorted(magnitude_groups.keys())
    pos_means = [np.mean(magnitude_groups[m]['positive']) if magnitude_groups[m]['positive'] else 0 for m in magnitudes]
    neg_means = [np.mean(magnitude_groups[m]['negative']) if magnitude_groups[m]['negative'] else 0 for m in magnitudes]
    
    x = np.arange(len(magnitudes))
    width = 0.35
    
    ax4.bar(x - width/2, pos_means, width, label='Positive α', alpha=0.8)
    ax4.bar(x + width/2, neg_means, width, label='Negative α', alpha=0.8)
    
    ax4.set_xlabel('|Alpha| Magnitude')
    ax4.set_ylabel('Mean Thinking Length (words)')
    ax4.set_title('Positive vs Negative Alpha Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{m:.2f}' for m in magnitudes])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('steering_results', exist_ok=True)
    plt.savefig('steering_results/magnitude_hypothesis_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: steering_results/magnitude_hypothesis_analysis.png")
    
    plt.show()

def main():
    """Run the complete magnitude hypothesis test."""
    print("Starting Magnitude-Based Steering Hypothesis Test")
    print("This will test whether reasoning length follows |α| rather than α")
    
    # Run the experiment
    results = test_magnitude_hypothesis()
    
    if results is None:
        print("Test failed - missing directions file")
        return
    
    # Analyze results
    analysis_data = analyze_results(results)
    
    # Create visualizations
    create_visualizations(results)
    
    # Save results
    output_file = 'steering_results/magnitude_hypothesis_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'raw_results': results,
            'analysis_summary': analysis_data,
            'experiment_params': {
                'model': 'Qwen/Qwen3-0.6B',
                'alpha_values': [-0.12, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.12],
                'samples_per_alpha': 5
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("\nMagnitude hypothesis test complete!")

if __name__ == "__main__":
    main() 