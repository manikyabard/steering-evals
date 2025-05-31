#!/bin/bash
# Setup script for Steering vs Faithfulness Experiments
# This script helps you set up and run the experiments from your proposal

set -e

echo "=== Steering vs Faithfulness Experiment Setup ==="

# Check if we're in the correct directory
if [ ! -d "chainscope" ]; then
    echo "Error: Please run this script from the steering-evals directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p directions
mkdir -p thinkedit_analysis
mkdir -p thinkedit_models
mkdir -p steering_results
mkdir -p steering_faithfulness_results

# Check for required files
echo "Checking dependencies..."

REQUIRED_FILES=(
    "steer_reasoning_length.py"
    "get_thinkedit_qwen3_models.py"
    "chainscope/chainscope/cot_paths.py"
    "chainscope/chainscope/cot_paths_eval.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Warning: $file not found"
    else
        echo "✓ Found $file"
    fi
done

# Function to run a pilot experiment
run_pilot_experiment() {
    MODEL=${1:-"Qwen/Qwen3-0.6B"}
    echo "Running pilot experiment with model: $MODEL"
    
    # Step 1: Check if steering directions exist
    MODEL_SHORT=$(basename "$MODEL")
    DIRECTIONS_FILE="directions/${MODEL_SHORT}_thinking_length_direction_gsm8k_attn.pt"
    
    if [ ! -f "$DIRECTIONS_FILE" ]; then
        echo "Warning: Steering directions not found at $DIRECTIONS_FILE"
        echo "You need to run extract_reasoning_length_direction_improved.py first"
        echo "See the ThinkEdit paper repository for this script"
        return 1
    fi
    
    echo "✓ Found steering directions: $DIRECTIONS_FILE"
    
    # Step 2: Run pilot experiment
    echo "Starting pilot experiment..."
    python experiment_steering_faithfulness.py \
        --model "$MODEL" \
        --dataset gsm8k \
        --n-samples 20 \
        --n-paths 3 \
        --alpha-values -0.05 0.0 0.05 \
        --experiment-type pilot \
        --anthropic \
        --output-dir "steering_faithfulness_results/pilot_$(date +%Y%m%d_%H%M%S)"
    
    echo "Pilot experiment completed!"
}

# Function to run full experiment
run_full_experiment() {
    MODEL=${1:-"Qwen/Qwen3-0.6B"}
    echo "Running full experiment with model: $MODEL"
    
    python experiment_steering_faithfulness.py \
        --model "$MODEL" \
        --dataset gsm8k \
        --n-samples 100 \
        --n-paths 5 \
        --alpha-values -0.1 -0.05 0.0 0.05 0.1 0.15 \
        --experiment-type full \
        --anthropic \
        --output-dir "steering_faithfulness_results/full_$(date +%Y%m%d_%H%M%S)"
}

# Function to run ThinkEdit experiment
run_thinkedit_experiment() {
    MODEL=${1:-"Qwen/Qwen3-0.6B"}
    echo "Running ThinkEdit experiment with model: $MODEL"
    
    # First create ThinkEdit model if it doesn't exist
    MODEL_SHORT=$(basename "$MODEL")
    THINKEDIT_MODEL="thinkedit_models/ThinkEdit-${MODEL_SHORT}"
    
    if [ ! -d "$THINKEDIT_MODEL" ]; then
        echo "Creating ThinkEdit model..."
        python get_thinkedit_qwen3_models.py \
            --model "$MODEL" \
            --intervention_weight 1.0 \
            --save_local
    fi
    
    # Run experiment with ThinkEdit model
    python experiment_steering_faithfulness.py \
        --model "$THINKEDIT_MODEL" \
        --dataset gsm8k \
        --n-samples 50 \
        --n-paths 5 \
        --alpha-values 0.0 \
        --experiment-type pilot \
        --anthropic \
        --output-dir "steering_faithfulness_results/thinkedit_$(date +%Y%m%d_%H%M%S)"
}

# Function to setup ChainScope for local evaluation
setup_chainscope() {
    echo "Setting up ChainScope..."
    
    cd chainscope
    
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        echo "Creating Python virtual environment..."
        python3.12 -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install requirements
    pip install --upgrade pip
    pip install -e .
    
    echo "✓ ChainScope setup complete"
    cd ..
}

# Function to demonstrate the workflow
demo_workflow() {
    echo "=== Demonstration Workflow ==="
    echo "This demonstrates the complete steering vs faithfulness pipeline:"
    echo ""
    echo "1. Runtime Steering Approach:"
    echo "   - Load steering direction vectors"
    echo "   - Apply steering during generation (α = -0.05, 0.0, 0.05)"
    echo "   - Generate CoT paths for math problems"
    echo "   - Evaluate faithfulness using ChainScope"
    echo "   - Compare faithfulness across steering strengths"
    echo ""
    echo "2. Expected Insights:"
    echo "   - How does reasoning length correlate with faithfulness?"
    echo "   - Do longer reasoning chains lead to more restoration errors?"
    echo "   - What's the optimal steering strength for faithful reasoning?"
    echo ""
    echo "3. Output Files:"
    echo "   - steering_vs_faithfulness_main.png: Main results plot"
    echo "   - correlation_matrix.png: Correlation analysis"
    echo "   - detailed_analysis.json: Complete numerical results"
    echo "   - README.md: Human-readable report"
}

# Main menu
case "$1" in
    "pilot")
        run_pilot_experiment "$2"
        ;;
    "full")
        run_full_experiment "$2"
        ;;
    "thinkedit")
        run_thinkedit_experiment "$2"
        ;;
    "setup")
        setup_chainscope
        ;;
    "demo")
        demo_workflow
        ;;
    *)
        echo "Usage: $0 {pilot|full|thinkedit|setup|demo} [model]"
        echo ""
        echo "Commands:"
        echo "  pilot MODEL    - Run pilot experiment (20 samples, 3 paths)"
        echo "  full MODEL     - Run full experiment (100 samples, 5 paths)"
        echo "  thinkedit MODEL - Run ThinkEdit weight editing experiment"
        echo "  setup          - Setup ChainScope environment"
        echo "  demo           - Show demonstration workflow"
        echo ""
        echo "Examples:"
        echo "  $0 pilot Qwen/Qwen3-0.6B"
        echo "  $0 full Qwen/Qwen3-1.5B"
        echo "  $0 setup"
        echo ""
        echo "Note: You need to have steering direction vectors ready."
        echo "See ThinkEdit paper repository for extraction scripts."
        exit 1
        ;;
esac 