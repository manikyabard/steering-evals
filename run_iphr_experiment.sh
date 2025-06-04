#!/bin/bash

# IPHR Experiment Runner
# This script runs the complete IPHR (Instruction-Paired Hypothesis Reversal) experiment
# to compare normal and thinkedit models for faithfulness in reasoning.

set -e  # Exit on any error

# Configuration
NORMAL_MODEL="Qwen/Qwen3-0.6B"
THINKEDIT_MODEL="Qwen/Qwen3-ThinkEdit-0.6B"  # Adjust path as needed
NUM_PAIRS=200
RESPONSES_PER_QUESTION=10
SERVER_PORT=30000
OUTPUT_DIR="iphr_experiment_results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_server() {
    local model_path=$1
    local port=$2
    
    print_status "Checking if SGLang server is running for $model_path on port $port..."
    
    if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
        print_success "Server is running on port $port"
        return 0
    else
        print_error "Server is not running on port $port"
        return 1
    fi
}

start_server() {
    local model_path=$1
    local port=$2
    
    print_status "Starting SGLang server for $model_path on port $port..."
    
    # Kill any existing server on this port
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        print_warning "Killing existing process on port $port"
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Start server in background
    python -m sglang.launch_server \
        --model-path "$model_path" \
        --port "$port" \
        --reasoning-parser qwen3 \
        > "server_${port}.log" 2>&1 &
    
    local server_pid=$!
    echo $server_pid > "server_${port}.pid"
    
    print_status "Server started with PID $server_pid, waiting for it to be ready..."
    
    # Wait for server to be ready (up to 5 minutes)
    local timeout=300
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if check_server "$model_path" "$port"; then
            print_success "Server is ready!"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        print_status "Waiting for server... (${elapsed}s/${timeout}s)"
    done
    
    print_error "Server failed to start within $timeout seconds"
    return 1
}

stop_server() {
    local port=$1
    
    if [ -f "server_${port}.pid" ]; then
        local pid=$(cat "server_${port}.pid")
        print_status "Stopping server with PID $pid on port $port..."
        kill $pid 2>/dev/null || true
        rm -f "server_${port}.pid"
        
        # Also kill any process using the port
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
            lsof -ti:$port | xargs kill -9 2>/dev/null || true
        fi
        
        print_success "Server stopped"
    else
        print_warning "No PID file found for port $port"
    fi
}

generate_responses() {
    local model_name=$1
    local model_path=$2
    local port=$3
    local output_file=$4
    
    print_status "Generating IPHR responses for $model_name..."
    
    python generate_iphr_responses.py \
        --model "$model_path" \
        --num-pairs $NUM_PAIRS \
        --responses-per-question $RESPONSES_PER_QUESTION \
        --server-url "http://localhost:$port" \
        --responses-file "$output_file" \
        --batch-size 16
    
    if [ $? -eq 0 ]; then
        print_success "Generated responses for $model_name saved to $output_file"
    else
        print_error "Failed to generate responses for $model_name"
        return 1
    fi
}

evaluate_responses() {
    local model_name=$1
    local responses_file=$2
    local analysis_dir=$3
    
    print_status "Evaluating faithfulness for $model_name..."
    
    python evaluate_iphr_faithfulness.py \
        --responses-file "$responses_file" \
        --output-dir "$analysis_dir" \
        --detailed-analysis
    
    if [ $? -eq 0 ]; then
        print_success "Evaluation complete for $model_name, results saved to $analysis_dir"
    else
        print_error "Failed to evaluate responses for $model_name"
        return 1
    fi
}

compare_models() {
    local normal_file=$1
    local thinkedit_file=$2
    local comparison_dir=$3
    
    print_status "Comparing normal and thinkedit models..."
    
    python evaluate_iphr_faithfulness.py \
        --responses-file "$normal_file" \
        --compare-file "$thinkedit_file" \
        --output-dir "$comparison_dir" \
        --detailed-analysis
    
    if [ $? -eq 0 ]; then
        print_success "Model comparison complete, results saved to $comparison_dir"
    else
        print_error "Failed to compare models"
        return 1
    fi
}

cleanup() {
    print_status "Cleaning up..."
    stop_server $SERVER_PORT
    stop_server $((SERVER_PORT + 1))
    print_success "Cleanup complete"
}

main() {
    print_status "Starting IPHR Experiment"
    print_status "Normal model: $NORMAL_MODEL"
    print_status "ThinkEdit model: $THINKEDIT_MODEL"
    print_status "Number of pairs: $NUM_PAIRS"
    print_status "Responses per question: $RESPONSES_PER_QUESTION"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Define output files
    NORMAL_RESPONSES="$OUTPUT_DIR/normal_model_responses.json"
    THINKEDIT_RESPONSES="$OUTPUT_DIR/thinkedit_model_responses.json"
    NORMAL_ANALYSIS="$OUTPUT_DIR/normal_analysis"
    THINKEDIT_ANALYSIS="$OUTPUT_DIR/thinkedit_analysis"
    COMPARISON_ANALYSIS="$OUTPUT_DIR/comparison_analysis"
    
    # Step 1: Generate responses for normal model
    print_status "=== STEP 1: Normal Model Response Generation ==="
    
    if [ ! -f "$NORMAL_RESPONSES" ]; then
        start_server "$NORMAL_MODEL" $SERVER_PORT
        generate_responses "Normal" "$NORMAL_MODEL" $SERVER_PORT "$NORMAL_RESPONSES"
        stop_server $SERVER_PORT
    else
        print_warning "Normal model responses already exist at $NORMAL_RESPONSES"
        print_status "Skipping generation. Delete the file to regenerate."
    fi
    
    # Step 2: Generate responses for thinkedit model  
    print_status "=== STEP 2: ThinkEdit Model Response Generation ==="
    
    if [ ! -f "$THINKEDIT_RESPONSES" ]; then
        start_server "$THINKEDIT_MODEL" $SERVER_PORT
        generate_responses "ThinkEdit" "$THINKEDIT_MODEL" $SERVER_PORT "$THINKEDIT_RESPONSES"
        stop_server $SERVER_PORT
    else
        print_warning "ThinkEdit model responses already exist at $THINKEDIT_RESPONSES"
        print_status "Skipping generation. Delete the file to regenerate."
    fi
    
    # Step 3: Evaluate normal model
    print_status "=== STEP 3: Normal Model Evaluation ==="
    evaluate_responses "Normal" "$NORMAL_RESPONSES" "$NORMAL_ANALYSIS"
    
    # Step 4: Evaluate thinkedit model
    print_status "=== STEP 4: ThinkEdit Model Evaluation ==="
    evaluate_responses "ThinkEdit" "$THINKEDIT_RESPONSES" "$THINKEDIT_ANALYSIS"
    
    # Step 5: Compare models
    print_status "=== STEP 5: Model Comparison ==="
    compare_models "$NORMAL_RESPONSES" "$THINKEDIT_RESPONSES" "$COMPARISON_ANALYSIS"
    
    # Step 6: Generate summary report
    print_status "=== STEP 6: Generating Summary Report ==="
    
    cat > "$OUTPUT_DIR/experiment_summary.md" << EOF
# IPHR Experiment Results

## Experiment Configuration
- **Normal Model**: $NORMAL_MODEL
- **ThinkEdit Model**: $THINKEDIT_MODEL  
- **Number of Question Pairs**: $NUM_PAIRS
- **Responses per Question**: $RESPONSES_PER_QUESTION
- **Date**: $(date)

## Files Generated
- Normal Model Responses: \`$NORMAL_RESPONSES\`
- ThinkEdit Model Responses: \`$THINKEDIT_RESPONSES\`
- Normal Model Analysis: \`$NORMAL_ANALYSIS/\`
- ThinkEdit Model Analysis: \`$THINKEDIT_ANALYSIS/\`
- Model Comparison: \`$COMPARISON_ANALYSIS/\`

## Key Results
See individual analysis directories for detailed results and visualizations.

### Quick Analysis
\`\`\`bash
# View normal model results
cat $NORMAL_ANALYSIS/faithfulness_analysis.json | jq '.overall_statistics'

# View thinkedit model results  
cat $THINKEDIT_ANALYSIS/faithfulness_analysis.json | jq '.overall_statistics'

# View comparison results
cat $COMPARISON_ANALYSIS/model_comparison.json | jq '.differences'
\`\`\`

### Visualizations
- Normal Model: \`$NORMAL_ANALYSIS/iphr_analysis.png\`
- ThinkEdit Model: \`$THINKEDIT_ANALYSIS/iphr_analysis.png\`
- Comparison: \`$COMPARISON_ANALYSIS/model_comparison.png\`

## Next Steps
1. Examine the visualizations for patterns
2. Look at detailed examples in the analysis files
3. Consider running with more question pairs for statistical significance
4. Analyze specific categories showing the most unfaithfulness
EOF
    
    print_success "=== EXPERIMENT COMPLETE ==="
    print_success "Results saved to: $OUTPUT_DIR"
    print_success "Summary report: $OUTPUT_DIR/experiment_summary.md"
    
    # Show quick results if jq is available
    if command -v jq &> /dev/null; then
        print_status "=== QUICK RESULTS ==="
        
        if [ -f "$COMPARISON_ANALYSIS/model_comparison.json" ]; then
            echo -e "${BLUE}Consistency Improvement:${NC}"
            cat "$COMPARISON_ANALYSIS/model_comparison.json" | jq -r '.differences.consistency_improvement'
            
            echo -e "${BLUE}Unfaithfulness Reduction:${NC}"
            cat "$COMPARISON_ANALYSIS/model_comparison.json" | jq -r '.differences.unfaithfulness_reduction'
            
            echo -e "${BLUE}Thinking Length Change:${NC}"
            cat "$COMPARISON_ANALYSIS/model_comparison.json" | jq -r '.differences.thinking_length_change'
        fi
    else
        print_warning "Install jq for formatted result display: sudo apt-get install jq"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --normal-model)
            NORMAL_MODEL="$2"
            shift 2
            ;;
        --thinkedit-model)
            THINKEDIT_MODEL="$2"
            shift 2
            ;;
        --num-pairs)
            NUM_PAIRS="$2"
            shift 2
            ;;
        --responses-per-question)
            RESPONSES_PER_QUESTION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --normal-model MODEL        Normal model path (default: $NORMAL_MODEL)"
            echo "  --thinkedit-model MODEL     ThinkEdit model path (default: $THINKEDIT_MODEL)"
            echo "  --num-pairs N               Number of question pairs (default: $NUM_PAIRS)"
            echo "  --responses-per-question N  Responses per question (default: $RESPONSES_PER_QUESTION)"
            echo "  --output-dir DIR            Output directory (default: $OUTPUT_DIR)"
            echo "  --help                      Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run the main experiment
main 