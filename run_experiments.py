#!/usr/bin/env python3
"""
Main Experiment Runner for Reasoning Length Steering

This script provides a unified interface for running all types of experiments
in the reasoning length steering project, handling path management and
providing clear command-line options.

Authors: Tino Trangia, Teresa Lee, Derrick Yao, Manikya Bardhan
Institution: UC San Diego
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_command(cmd, cwd=None):
    """Run a command and handle errors gracefully."""
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd or project_root, check=True, 
                              capture_output=True, text=True)
        print("✓ Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def generate_responses(args):
    """Run response generation experiment."""
    cmd = [
        sys.executable, "src/generation/generate_responses_gsm8k.py",
        "--model", args.model,
        "--num_samples", str(args.num_samples)
    ]
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    
    return run_command(cmd)

def extract_directions(args):
    """Run direction extraction experiment."""
    responses_file = f"results/responses/{args.model.split('/')[-1]}_gsm8k_responses.json"
    
    cmd = [
        sys.executable, "src/extraction/extract_reasoning_length_direction_improved.py",
        "--model", args.model,
        "--responses-file", responses_file
    ]
    if hasattr(args, 'components') and args.components:
        cmd.extend(["--components"] + args.components)
    
    return run_command(cmd)

def test_steering(args):
    """Run steering experiment."""
    cmd = [
        sys.executable, "src/generation/steer_reasoning_length.py",
        "--model", args.model,
        "--component", getattr(args, 'component', 'attn')
    ]
    if hasattr(args, 'direction_weights') and args.direction_weights:
        cmd.extend(["--direction_weights"] + [str(w) for w in args.direction_weights])
    
    return run_command(cmd)

def run_iphr_evaluation(args):
    """Run IPHR faithfulness evaluation."""
    cmd = [
        "./experiments/run_iphr_experiment.sh",
        "--normal-model", args.normal_model,
        "--num-pairs", str(getattr(args, 'num_pairs', 50))
    ]
    
    if hasattr(args, 'thinkedit_model') and args.thinkedit_model:
        cmd.extend(["--thinkedit-model", args.thinkedit_model])
    
    if getattr(args, 'enable_llm_evaluation', False):
        cmd.append("--enable-llm-evaluation")
        
    if getattr(args, 'use_same_model_for_eval', False):
        cmd.append("--use-same-model-for-eval")
    
    return run_command(cmd)

def full_pipeline(args):
    """Run the complete experimental pipeline."""
    print("🚀 Starting full experimental pipeline...")
    
    steps = [
        ("Generating responses", lambda: generate_responses(args)),
        ("Extracting directions", lambda: extract_directions(args)),
        ("Testing steering", lambda: test_steering(args))
    ]
    
    if hasattr(args, 'include_iphr') and args.include_iphr:
        steps.append(("Running IPHR evaluation", lambda: run_iphr_evaluation(args)))
    
    for step_name, step_func in steps:
        print(f"\n📊 {step_name}...")
        if not step_func():
            print(f"❌ Pipeline failed at: {step_name}")
            return False
    
    print("\n🎉 Full pipeline completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Run reasoning length steering experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline for basic steering experiment
  python run_experiments.py full --model Qwen/Qwen3-0.6B --num_samples 1000

  # Just generate responses
  python run_experiments.py generate --model Qwen/Qwen3-0.6B --num_samples 500
  
  # Run IPHR evaluation
  python run_experiments.py iphr --normal-model Qwen/Qwen3-0.6B --enable-llm-evaluation
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run complete experimental pipeline')
    full_parser.add_argument('--model', required=True, help='Model name or path')
    full_parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    full_parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    full_parser.add_argument('--components', nargs='+', default=['attn'], help='Components to extract directions for')
    full_parser.add_argument('--direction_weights', nargs='+', type=float, 
                           default=[-0.08, -0.04, 0.0, 0.04, 0.08], help='Steering weights to test')
    full_parser.add_argument('--include-iphr', action='store_true', help='Include IPHR evaluation')
    full_parser.add_argument('--thinkedit-model', help='ThinkEdit model path for IPHR')
    
    # Generate responses command
    gen_parser = subparsers.add_parser('generate', help='Generate model responses')
    gen_parser.add_argument('--model', required=True, help='Model name or path')
    gen_parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    gen_parser.add_argument('--batch_size', type=int, help='Batch size for generation')
    
    # Extract directions command
    extract_parser = subparsers.add_parser('extract', help='Extract reasoning direction vectors')
    extract_parser.add_argument('--model', required=True, help='Model name or path')
    extract_parser.add_argument('--components', nargs='+', default=['attn'], help='Components to extract')
    
    # Test steering command
    steer_parser = subparsers.add_parser('steer', help='Test steering effects')
    steer_parser.add_argument('--model', required=True, help='Model name or path')
    steer_parser.add_argument('--component', default='attn', help='Component to steer')
    steer_parser.add_argument('--direction_weights', nargs='+', type=float,
                            default=[-0.08, -0.04, 0.0, 0.04, 0.08], help='Steering weights')
    
    # IPHR evaluation command
    iphr_parser = subparsers.add_parser('iphr', help='Run IPHR faithfulness evaluation')
    iphr_parser.add_argument('--normal-model', required=True, help='Normal model name or path')
    iphr_parser.add_argument('--thinkedit-model', help='ThinkEdit model path')
    iphr_parser.add_argument('--num-pairs', type=int, default=50, help='Number of question pairs')
    iphr_parser.add_argument('--enable-llm-evaluation', action='store_true', help='Enable LLM-based evaluation')
    iphr_parser.add_argument('--use-same-model-for-eval', action='store_true', help='Use same model for evaluation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Ensure results directories exist
    os.makedirs("results/responses", exist_ok=True)
    os.makedirs("results/directions", exist_ok=True)
    os.makedirs("results/analysis", exist_ok=True)
    
    # Route to appropriate function
    if args.command == 'full':
        success = full_pipeline(args)
    elif args.command == 'generate':
        success = generate_responses(args)
    elif args.command == 'extract':
        success = extract_directions(args)
    elif args.command == 'steer':
        success = test_steering(args)
    elif args.command == 'iphr':
        success = run_iphr_evaluation(args)
    else:
        print(f"Unknown command: {args.command}")
        success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 