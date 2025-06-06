#!/bin/bash

# IPHR Experiment Runner
# This script runs the complete IPHR (Instruction-Paired Hypothesis Reversal) experiment
# to compare normal and thinkedit models for faithfulness in reasoning.
# Now includes enhanced LLM-based pattern analysis similar to chainscope approach.

set -e  # Exit on any error

# Configuration
NORMAL_MODEL="Qwen/Qwen3-0.6B"
THINKEDIT_MODEL="Qwen/Qwen3-ThinkEdit-0.6B"  # Adjust path as needed
EVALUATOR_MODEL="Qwen/Qwen3-0.6B"  # Model for LLM evaluation
NUM_PAIRS=200
RESPONSES_PER_QUESTION=10
OVERSAMPLE_RESPONSES_PER_QUESTION=100  # For sophisticated pattern analysis (chainscope-style)
SERVER_PORT=30000
EVALUATOR_PORT=30001
OUTPUT_DIR="iphr_experiment_results"
ENABLE_LLM_EVALUATION=false
MAX_LLM_ANALYSES=200
MAX_CONCURRENT_REQUESTS=64  # Number of concurrent requests to LLM evaluator
SEQUENTIAL_SERVERS=false  # New option for resource-constrained environments
USE_SAME_MODEL_FOR_EVAL=false  # New option to use the same model for generation and evaluation
OVERSAMPLE_UNFAITHFUL=false  # Chainscope-style oversampling for unfaithful pairs
PATTERN_ANALYSIS_MODE="enhanced"  # Options: basic, enhanced, chainscope

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
    
    # Simple health check - just try the basic endpoint
    if curl -s --connect-timeout 5 --max-time 10 "http://localhost:$port/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

start_server() {
    local model_path=$1
    local port=$2
    local server_name=$3
    
    print_status "Starting SGLang server for $model_path on port $port..."
    
    # Kill any existing server on this port using multiple methods
    local existing_pid=""
    if command -v lsof >/dev/null 2>&1; then
        # Use lsof if available
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_warning "Killing existing process on port $port (using lsof)"
            existing_pid=$(lsof -ti:$port)
            kill -9 $existing_pid 2>/dev/null || true
        fi
    fi
    
    # Wait for port to be free
    if [ -n "$existing_pid" ]; then
        print_status "Waiting for port $port to be free..."
        sleep 3
    fi
    
    # Start server in background
    local log_file="server_${server_name}_${port}.log"
    print_status "Starting server with log file: $log_file"
    
    # Use nohup to ensure server doesn't die when parent exits
    nohup python -m sglang.launch_server \
        --model-path "$model_path" \
        --port "$port" \
        --reasoning-parser qwen3 --attention-backend triton &> "$log_file" &
    
    local server_pid=$!
    echo $server_pid > "server_${server_name}_${port}.pid"
    
    print_status "Server started with PID $server_pid, waiting for health check to pass..."
    
    # Simple health check loop - just keep trying until it works
    local timeout=600  # 10 minutes
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        # Check if process is still alive
        if ! kill -0 $server_pid 2>/dev/null; then
            print_error "Server process $server_pid died unexpectedly"
            print_error "Check server log: $log_file"
            return 1
        fi
        
        # Try health check
        if check_server "$model_path" "$port"; then
            print_success "Server is ready on port $port!"
            return 0
        fi
        
        # Wait and try again
        sleep 10
        elapsed=$((elapsed + 10))
        
        # Progress update every minute
        if [ $((elapsed % 60)) -eq 0 ]; then
            print_status "Still waiting for server... (${elapsed}s/${timeout}s)"
        fi
    done
    
    print_error "Server failed to respond to health checks within $timeout seconds"
    print_error "Check server log: $log_file"
    return 1
}

# New function to verify server is still alive and responding
verify_server_alive() {
    local port=$1
    local server_name=$2
    
    # Check if we have a PID file
    if [ ! -f "server_${server_name}_${port}.pid" ]; then
        print_error "No PID file found for $server_name server on port $port"
        return 1
    fi
    
    local pid=$(cat "server_${server_name}_${port}.pid")
    
    # Check if process is still running
    if ! kill -0 $pid 2>/dev/null; then
        print_error "Server process $pid is not running"
        return 1
    fi
    
    # Check if server responds to health check
    if ! curl -s --connect-timeout 5 --max-time 10 "http://localhost:$port/health" > /dev/null 2>&1; then
        print_error "Server on port $port is not responding to health checks"
        return 1
    fi
    
    return 0
}

# Enhanced function to ensure server is ready before operations
ensure_server_ready() {
    local model_path=$1
    local port=$2
    local server_name=$3
    
    print_status "Ensuring $server_name server is ready on port $port..."
    
    # First check if server is already running
    if verify_server_alive "$port" "$server_name"; then
        print_success "Server is already running and ready"
        return 0
    fi
    
    # If not running, start it
    print_status "Server not ready, starting new server..."
    start_server "$model_path" "$port" "$server_name"
    
    return $?
}

stop_server() {
    local port=$1
    local server_name=$2
    
    print_status "Stopping $server_name server on port $port..."
    
    # Stop by PID if we have it
    if [ -f "server_${server_name}_${port}.pid" ]; then
        local pid=$(cat "server_${server_name}_${port}.pid")
        print_status "Stopping $server_name server with PID $pid..."
        
        # Try graceful shutdown first
        kill -TERM $pid 2>/dev/null || true
        
        # Wait for graceful shutdown
        local wait_count=0
        while [ $wait_count -lt 10 ]; do
            if ! kill -0 $pid 2>/dev/null; then
                break
            fi
            sleep 1
            wait_count=$((wait_count + 1))
        done
        
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            print_warning "Graceful shutdown failed, force killing PID $pid"
            kill -9 $pid 2>/dev/null || true
            
            # Wait a bit more for force kill
            wait_count=0
            while [ $wait_count -lt 5 ]; do
                if ! kill -0 $pid 2>/dev/null; then
                    break
                fi
                sleep 1
                wait_count=$((wait_count + 1))
            done
        fi
        
        rm -f "server_${server_name}_${port}.pid"
    else
        print_warning "No PID file found for $server_name server on port $port"
    fi
    
    # Also kill any process using the port as backup
    local port_pid=""
    if command -v lsof >/dev/null 2>&1; then
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            port_pid=$(lsof -ti:$port)
            print_status "Found additional process $port_pid on port $port, killing..."
            kill -9 $port_pid 2>/dev/null || true
        fi
    else
        # Use ss/netstat as fallback
        if ss -tlnp 2>/dev/null | grep -q ":$port " || netstat -tlnp 2>/dev/null | grep -q ":$port "; then
            port_pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1)
            if [ -z "$port_pid" ]; then
                port_pid=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d/ -f1 | head -1)
            fi
            if [ -n "$port_pid" ] && [ "$port_pid" != "-" ]; then
                print_status "Found additional process $port_pid on port $port, killing..."
                kill -9 $port_pid 2>/dev/null || true
            fi
        fi
    fi
    
    # Wait for port to be free
    print_status "Waiting for port $port to be free..."
    local wait_count=0
    while [ $wait_count -lt 15 ]; do
        if ! (ss -tlnp 2>/dev/null | grep -q ":$port " || netstat -tlnp 2>/dev/null | grep -q ":$port "); then
            break
        fi
        sleep 1
        wait_count=$((wait_count + 1))
    done
    
    print_success "$server_name server stopped"
}

generate_responses() {
    local model_name=$1
    local model_path=$2
    local port=$3
    local output_file=$4
    
    print_status "Generating IPHR responses for $model_name..."
    
    # Ensure server is ready before generation
    if ! verify_server_alive "$port" "main"; then
        print_error "Server on port $port is not ready for response generation"
        return 1
    fi
    
    python generate_iphr_responses.py \
        --model "$model_path" \
        --num-pairs $NUM_PAIRS \
        --responses-per-question $RESPONSES_PER_QUESTION \
        --server-url "http://localhost:$port" \
        --responses-file "$output_file" \
        --questions-file "factual_iphr_questions.json" \
        --batch-size 64
    
    local result=$?
    
    if [ $result -eq 0 ]; then
        print_success "Generated responses for $model_name saved to $output_file"
    else
        print_error "Failed to generate responses for $model_name"
        # Check if server is still alive
        if ! verify_server_alive "$port" "main"; then
            print_error "Server died during response generation"
        fi
        return 1
    fi
    
    return 0
}

start_evaluator_server() {
    local model_path=$1
    local port=$2
    
    # If using sequential servers, we need to stop the main server first
    if [ "$SEQUENTIAL_SERVERS" = true ]; then
        print_status "Sequential mode: stopping main server before starting evaluator"
        stop_server $SERVER_PORT "main" 2>/dev/null || true
        sleep 2  # Give it time to fully shut down
    fi
    
    print_status "Starting LLM evaluator server for $model_path on port $port..."
    start_server "$model_path" "$port" "evaluator"
}

stop_evaluator_server() {
    local port=$1
    
    stop_server "$port" "evaluator"
    
    # If using sequential servers, we might need to restart the main server
    if [ "$SEQUENTIAL_SERVERS" = true ] && [ -n "$CURRENT_MAIN_MODEL" ]; then
        print_status "Sequential mode: restarting main server for $CURRENT_MAIN_MODEL after evaluator use..."
        sleep 2  # Give it time to fully shut down
        start_server "$CURRENT_MAIN_MODEL" $SERVER_PORT "main"
    fi
}

evaluate_responses() {
    local model_name=$1
    local responses_file=$2
    local analysis_dir=$3
    local use_llm_eval=$4
    
    print_status "Evaluating faithfulness for $model_name..."
    
    local eval_cmd="python evaluate_iphr_faithfulness.py \
        --responses-file \"$responses_file\" \
        --output-dir \"$analysis_dir\" \
        --detailed-analysis \
        --max-concurrent-requests $MAX_CONCURRENT_REQUESTS"  # Enable concurrent LLM analysis with progress tracking
    
    if [ "$use_llm_eval" = true ]; then
        print_status "Using enhanced LLM evaluation for $model_name"
        
        # Determine evaluator model and port
        local evaluator_model="$EVALUATOR_MODEL"
        local evaluator_port="$EVALUATOR_PORT"
        
        if [ "$USE_SAME_MODEL_FOR_EVAL" = true ]; then
            # Use the same model as the one being evaluated
            if [ "$model_name" = "Normal" ]; then
                evaluator_model="$NORMAL_MODEL"
            else
                evaluator_model="$THINKEDIT_MODEL"
            fi
            evaluator_port="$SERVER_PORT"
        fi
        
        # Handle server management for LLM evaluation
        if [ "$SEQUENTIAL_SERVERS" = true ] || [ "$USE_SAME_MODEL_FOR_EVAL" = true ]; then
            if [ "$USE_SAME_MODEL_FOR_EVAL" = true ]; then
                # Use the same server that's already running
                print_status "Using current server ($evaluator_model) for LLM evaluation"
                # Ensure the right model is running
                if [ "$CURRENT_MAIN_MODEL" != "$evaluator_model" ]; then
                    stop_server $SERVER_PORT "main" 2>/dev/null || true
                    sleep 2
                    ensure_server_ready "$evaluator_model" $SERVER_PORT "main"
                    CURRENT_MAIN_MODEL="$evaluator_model"
                fi
                evaluator_port="$SERVER_PORT"
            else
                # Stop current server and start evaluator
                stop_server $SERVER_PORT "main" 2>/dev/null || true
                sleep 2
                ensure_server_ready "$evaluator_model" $SERVER_PORT "main"
                evaluator_port="$SERVER_PORT"
            fi
        else
            # Start dedicated evaluator server (original behavior)
            start_evaluator_server "$evaluator_model" "$evaluator_port"
        fi
        
        # Verify evaluator server is ready
        if ! verify_server_alive "$evaluator_port" "evaluator" && ! verify_server_alive "$evaluator_port" "main"; then
            print_error "Evaluator server is not ready"
            return 1
        fi
        
        eval_cmd="$eval_cmd \
            --llm-evaluation \
            --evaluator-server-url \"http://localhost:$evaluator_port\" \
            --evaluator-model \"$evaluator_model\" \
            --max-llm-analyses $MAX_LLM_ANALYSES"
        
        # Run evaluation
        eval $eval_cmd
        local eval_result=$?
        
        # Clean up evaluator server if needed
        if [ "$SEQUENTIAL_SERVERS" = true ] && [ "$USE_SAME_MODEL_FOR_EVAL" = false ]; then
            # Keep the evaluator model running for now, will switch back if needed
            print_status "Sequential mode: evaluation complete, server will be managed as needed"
        elif [ "$SEQUENTIAL_SERVERS" = false ] && [ "$USE_SAME_MODEL_FOR_EVAL" = false ]; then
            stop_evaluator_server "$evaluator_port"
        fi
        
        return $eval_result
    else
        eval $eval_cmd
    fi
    
    local result=$?
    if [ $result -eq 0 ]; then
        print_success "Evaluation complete for $model_name, results saved to $analysis_dir"
    else
        print_error "Failed to evaluate responses for $model_name"
        return 1
    fi
    
    return 0
}

compare_models() {
    local normal_file=$1
    local thinkedit_file=$2
    local comparison_dir=$3
    local use_llm_eval=$4
    
    print_status "Comparing normal and thinkedit models..."
    
    local compare_cmd="python evaluate_iphr_faithfulness.py \
        --responses-file \"$normal_file\" \
        --compare-file \"$thinkedit_file\" \
        --output-dir \"$comparison_dir\" \
        --detailed-analysis \
        --max-concurrent-requests $MAX_CONCURRENT_REQUESTS"  # Enable concurrent LLM analysis with progress tracking
    
    if [ "$use_llm_eval" = true ]; then
        print_status "Using enhanced LLM evaluation for model comparison"
        
        # Determine evaluator model and port
        local evaluator_model="$EVALUATOR_MODEL"
        local evaluator_port="$EVALUATOR_PORT"
        
        if [ "$USE_SAME_MODEL_FOR_EVAL" = true ]; then
            # Use one of the models for evaluation (prefer normal model)
            evaluator_model="$NORMAL_MODEL"
            evaluator_port="$SERVER_PORT"
        fi
        
        # Handle server management for comparison
        if [ "$SEQUENTIAL_SERVERS" = true ] || [ "$USE_SAME_MODEL_FOR_EVAL" = true ]; then
            if [ "$USE_SAME_MODEL_FOR_EVAL" = true ]; then
                # Start normal model server for evaluation
                start_server "$NORMAL_MODEL" $SERVER_PORT "main"
                evaluator_port="$SERVER_PORT"
            else
                # Start evaluator model
                start_server "$evaluator_model" $SERVER_PORT "main"
                evaluator_port="$SERVER_PORT"
            fi
        else
            # Start dedicated evaluator server
            start_evaluator_server "$evaluator_model" "$evaluator_port"
        fi
        
        compare_cmd="$compare_cmd \
            --llm-evaluation \
            --evaluator-server-url \"http://localhost:$evaluator_port\" \
            --evaluator-model \"$evaluator_model\" \
            --max-llm-analyses $MAX_LLM_ANALYSES"
        
        # Run comparison
        eval $compare_cmd
        local compare_result=$?
        
        # Clean up evaluator server if needed
        if [ "$SEQUENTIAL_SERVERS" = true ] || [ "$USE_SAME_MODEL_FOR_EVAL" = true ]; then
            stop_server $evaluator_port "main"
        else
            stop_evaluator_server "$evaluator_port"
        fi
        
        return $compare_result
    else
        eval $compare_cmd
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Model comparison complete, results saved to $comparison_dir"
    else
        print_error "Failed to compare models"
        return 1
    fi
}

cleanup() {
    print_status "Cleaning up..."
    stop_server $SERVER_PORT "main" 2>/dev/null || true
    if [ "$ENABLE_LLM_EVALUATION" = true ] && [ "$SEQUENTIAL_SERVERS" = false ] && [ "$USE_SAME_MODEL_FOR_EVAL" = false ]; then
        stop_server $EVALUATOR_PORT "evaluator" 2>/dev/null || true
    fi
    print_success "Cleanup complete"
}

main() {
    print_status "Starting IPHR Experiment with Enhanced Analysis"
    print_status "Normal model: $NORMAL_MODEL"
    print_status "ThinkEdit model: $THINKEDIT_MODEL"
    if [ "$ENABLE_LLM_EVALUATION" = true ]; then
        if [ "$USE_SAME_MODEL_FOR_EVAL" = true ]; then
            print_status "LLM Evaluator: Using same models as generation"
        else
            print_status "LLM Evaluator model: $EVALUATOR_MODEL"
        fi
        print_status "Max LLM analyses per evaluation: $MAX_LLM_ANALYSES"
        print_status "Max concurrent requests: $MAX_CONCURRENT_REQUESTS"
        print_status "Sequential servers mode: $SEQUENTIAL_SERVERS"
        print_status "Use same model for eval: $USE_SAME_MODEL_FOR_EVAL"
        print_status "Oversample unfaithful pairs: $OVERSAMPLE_UNFAITHFUL"
        print_status "Pattern analysis mode: $PATTERN_ANALYSIS_MODE"
        if [ "$OVERSAMPLE_UNFAITHFUL" = true ]; then
            print_status "Oversample responses per question: $OVERSAMPLE_RESPONSES_PER_QUESTION"
        fi
    fi
    print_status "Number of pairs: $NUM_PAIRS"
    print_status "Responses per question: $RESPONSES_PER_QUESTION"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Don't start evaluator server early if using sequential mode or same model
    if [ "$ENABLE_LLM_EVALUATION" = true ] && [ "$SEQUENTIAL_SERVERS" = false ] && [ "$USE_SAME_MODEL_FOR_EVAL" = false ]; then
        print_status "=== STARTING DEDICATED LLM EVALUATOR SERVER ==="
        start_server "$EVALUATOR_MODEL" $EVALUATOR_PORT "evaluator"
    fi
    
    # Track current main model for sequential mode
    CURRENT_MAIN_MODEL=""
    
    # Define output files
    NORMAL_RESPONSES="$OUTPUT_DIR/normal_model_responses.json"
    THINKEDIT_RESPONSES="$OUTPUT_DIR/thinkedit_model_responses.json"
    NORMAL_ANALYSIS="$OUTPUT_DIR/normal_analysis"
    THINKEDIT_ANALYSIS="$OUTPUT_DIR/thinkedit_analysis"
    COMPARISON_ANALYSIS="$OUTPUT_DIR/comparison_analysis"
    
    # Step 1: Generate responses for normal model
    print_status "=== STEP 1: Normal Model Response Generation ==="
    
    if [ ! -f "$NORMAL_RESPONSES" ]; then
        ensure_server_ready "$NORMAL_MODEL" $SERVER_PORT "main"
        CURRENT_MAIN_MODEL="$NORMAL_MODEL"
        generate_responses "Normal" "$NORMAL_MODEL" $SERVER_PORT "$NORMAL_RESPONSES"
        if [ "$SEQUENTIAL_SERVERS" = true ]; then
            print_status "Sequential mode: keeping main server running for now"
        fi
    else
        print_warning "Normal model responses already exist at $NORMAL_RESPONSES"
        print_status "Skipping generation. Delete the file to regenerate."
    fi
    
    # Step 2: Generate responses for thinkedit model  
    print_status "=== STEP 2: ThinkEdit Model Response Generation ==="
    
    if [ ! -f "$THINKEDIT_RESPONSES" ]; then
        if [ "$SEQUENTIAL_SERVERS" = true ] || [ "$CURRENT_MAIN_MODEL" != "$THINKEDIT_MODEL" ]; then
            if [ "$SEQUENTIAL_SERVERS" = true ]; then
                print_status "Sequential mode: switching to ThinkEdit model"
                stop_server $SERVER_PORT "main" 2>/dev/null || true
                sleep 2
            fi
            ensure_server_ready "$THINKEDIT_MODEL" $SERVER_PORT "main"
            CURRENT_MAIN_MODEL="$THINKEDIT_MODEL"
        fi
        generate_responses "ThinkEdit" "$THINKEDIT_MODEL" $SERVER_PORT "$THINKEDIT_RESPONSES"
        if [ "$SEQUENTIAL_SERVERS" = true ]; then
            print_status "Sequential mode: keeping server running for evaluation"
        fi
    else
        print_warning "ThinkEdit model responses already exist at $THINKEDIT_RESPONSES"
        print_status "Skipping generation. Delete the file to regenerate."
    fi
    
    # Step 3: Evaluate normal model
    print_status "=== STEP 3: Normal Model Evaluation ==="
    evaluate_responses "Normal" "$NORMAL_RESPONSES" "$NORMAL_ANALYSIS" $ENABLE_LLM_EVALUATION
    
    # Step 4: Evaluate thinkedit model
    print_status "=== STEP 4: ThinkEdit Model Evaluation ==="
    evaluate_responses "ThinkEdit" "$THINKEDIT_RESPONSES" "$THINKEDIT_ANALYSIS" $ENABLE_LLM_EVALUATION
    
    # Step 5: Compare models
    print_status "=== STEP 5: Model Comparison ==="
    compare_models "$NORMAL_RESPONSES" "$THINKEDIT_RESPONSES" "$COMPARISON_ANALYSIS" $ENABLE_LLM_EVALUATION
    
    # Step 6: Generate summary report
    print_status "=== STEP 6: Generating Summary Report ==="
    
    local eval_method_note=""
    if [ "$ENABLE_LLM_EVALUATION" = true ]; then
        eval_method_note="
## LLM Evaluation
- **Evaluator Model**: $EVALUATOR_MODEL
- **Max Analyses per Model**: $MAX_LLM_ANALYSES
- **Pattern Categories**: fact-manipulation, argument-switching, answer-flipping, other"
    fi
    
    cat > "$OUTPUT_DIR/experiment_summary.md" << EOF
# Enhanced IPHR Experiment Results (Chainscope-Inspired)

## Experiment Configuration
- **Normal Model**: $NORMAL_MODEL
- **ThinkEdit Model**: $THINKEDIT_MODEL  
- **Number of Question Pairs**: $NUM_PAIRS
- **Responses per Question**: $RESPONSES_PER_QUESTION
- **LLM Evaluation Enabled**: $ENABLE_LLM_EVALUATION
- **Pattern Analysis Mode**: $PATTERN_ANALYSIS_MODE
- **Oversample Unfaithful Pairs**: $OVERSAMPLE_UNFAITHFUL
- **Date**: $(date)
$eval_method_note

## Chainscope-Inspired Enhancements
- **Sophisticated Pattern Analysis**: Enhanced categorization with fact manipulation, argument switching, selective reasoning, confidence manipulation, and context shifting detection
- **XML-Structured LLM Prompts**: Detailed analysis prompts with structured output using XML-like tags
- **Enhanced Confidence Scoring**: 1-10 confidence assessment based on hedging language analysis
- **Systematic Unfaithfulness Detection**: Multi-category pattern detection across response pairs
- **Advanced Answer Flipping Analysis**: Clear distinction between uncertainty and actual reasoning contradictions

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
EOF

    if [ "$ENABLE_LLM_EVALUATION" = true ]; then
        cat >> "$OUTPUT_DIR/experiment_summary.md" << EOF

### Enhanced LLM Analysis
- Normal Model LLM Patterns: \`$NORMAL_ANALYSIS/llm_pattern_analysis.png\`
- ThinkEdit Model LLM Patterns: \`$THINKEDIT_ANALYSIS/llm_pattern_analysis.png\`

#### Pattern Analysis Commands
\`\`\`bash
# View LLM pattern analysis for normal model
cat $NORMAL_ANALYSIS/faithfulness_analysis.json | jq '.enhanced_patterns.llm.patterns'

# View LLM pattern analysis for thinkedit model
cat $THINKEDIT_ANALYSIS/faithfulness_analysis.json | jq '.enhanced_patterns.llm.patterns'
\`\`\`
EOF
    fi

    cat >> "$OUTPUT_DIR/experiment_summary.md" << EOF

## Next Steps
1. Examine the visualizations for patterns
2. Look at detailed examples in the analysis files
3. Consider running with more question pairs for statistical significance
4. Analyze specific categories showing the most unfaithfulness
EOF

    if [ "$ENABLE_LLM_EVALUATION" = true ]; then
        cat >> "$OUTPUT_DIR/experiment_summary.md" << EOF
5. Review LLM-detected unfaithfulness patterns for deeper insights
6. Compare sophisticated reasoning failures between models
EOF
    fi
    
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
        
        if [ "$ENABLE_LLM_EVALUATION" = true ]; then
            echo -e "${BLUE}LLM Analysis Results:${NC}"
            if [ -f "$NORMAL_ANALYSIS/faithfulness_analysis.json" ]; then
                echo "Normal Model - Most Common Pattern:"
                cat "$NORMAL_ANALYSIS/faithfulness_analysis.json" | jq -r '.enhanced_patterns.llm.patterns.most_common_pattern // "none"'
            fi
            if [ -f "$THINKEDIT_ANALYSIS/faithfulness_analysis.json" ]; then
                echo "ThinkEdit Model - Most Common Pattern:"
                cat "$THINKEDIT_ANALYSIS/faithfulness_analysis.json" | jq -r '.enhanced_patterns.llm.patterns.most_common_pattern // "none"'
            fi
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
        --evaluator-model)
            EVALUATOR_MODEL="$2"
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
        --enable-llm-evaluation)
            ENABLE_LLM_EVALUATION=true
            shift
            ;;
        --max-llm-analyses)
            MAX_LLM_ANALYSES="$2"
            shift 2
            ;;
        --evaluator-port)
            EVALUATOR_PORT="$2"
            shift 2
            ;;
        --sequential-servers)
            SEQUENTIAL_SERVERS=true
            shift
            ;;
        --use-same-model-for-eval)
            USE_SAME_MODEL_FOR_EVAL=true
            SEQUENTIAL_SERVERS=true  # Implied when using same model
            shift
            ;;
        --oversample-unfaithful)
            OVERSAMPLE_UNFAITHFUL=true
            shift
            ;;
        --pattern-analysis-mode)
            PATTERN_ANALYSIS_MODE="$2"
            shift 2
            ;;
        --oversample-responses)
            OVERSAMPLE_RESPONSES_PER_QUESTION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --normal-model MODEL           Normal model path (default: $NORMAL_MODEL)"
            echo "  --thinkedit-model MODEL        ThinkEdit model path (default: $THINKEDIT_MODEL)"
            echo "  --evaluator-model MODEL        LLM evaluator model path (default: $EVALUATOR_MODEL)"
            echo "  --num-pairs N                  Number of question pairs (default: $NUM_PAIRS)"
            echo "  --responses-per-question N     Responses per question (default: $RESPONSES_PER_QUESTION)"
            echo "  --output-dir DIR               Output directory (default: $OUTPUT_DIR)"
            echo "  --enable-llm-evaluation        Enable LLM-based pattern analysis"
            echo "  --max-llm-analyses N           Max pairs to analyze with LLM (default: $MAX_LLM_ANALYSES)"
            echo "  --evaluator-port PORT          Port for LLM evaluator server (default: $EVALUATOR_PORT)"
            echo "  --sequential-servers           Run only one server at a time (resource-constrained environments)"
            echo "  --use-same-model-for-eval      Use the same model for generation and evaluation (saves resources)"
            echo "  --oversample-unfaithful        Enable chainscope-style oversampling for unfaithful pairs"
            echo "  --pattern-analysis-mode MODE   Pattern analysis mode (default: $PATTERN_ANALYSIS_MODE)"
            echo "  --oversample-responses N        Over-sample responses per question (default: $OVERSAMPLE_RESPONSES_PER_QUESTION)"
            echo "  --help                         Show this help message"
            echo ""
            echo "Resource-constrained examples:"
            echo "  # Use same model for generation and evaluation (minimal resources)"
            echo "  $0 --enable-llm-evaluation --use-same-model-for-eval"
            echo ""
            echo "  # Use different evaluator model but sequential servers"
            echo "  $0 --enable-llm-evaluation --sequential-servers --evaluator-model 'different/model'"
            echo ""
            echo "  # Standard mode with dedicated evaluator server (requires more resources)"
            echo "  $0 --enable-llm-evaluation --evaluator-model 'different/model'"
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