#!/bin/bash

# Queue IPHR Experiments for Multiple Models
# This script manages running IPHR experiments across multiple model pairs,
# with proper resource management and result organization.

set -e  # Exit on any error

# Default configuration
DEFAULT_NUM_PAIRS=100
DEFAULT_RESPONSES_PER_QUESTION=10
DEFAULT_OVERSAMPLE_RESPONSES_PER_QUESTION=100  # For chainscope-style oversampling
MAX_PARALLEL_JOBS=1  # Number of experiments to run in parallel
QUEUE_OUTPUT_DIR="iphr_queue_results"
WAIT_BETWEEN_JOBS=30  # seconds to wait between jobs
DEFAULT_ENABLE_LLM_EVALUATION=true
DEFAULT_EVALUATOR_MODEL=""
DEFAULT_MAX_LLM_ANALYSES=200
DEFAULT_SEQUENTIAL_SERVERS=true
DEFAULT_USE_SAME_MODEL_FOR_EVAL=false
DEFAULT_OVERSAMPLE_UNFAITHFUL=false  # Chainscope-style oversampling for unfaithful pairs
DEFAULT_PATTERN_ANALYSIS_MODE="enhanced"  # Options: basic, enhanced, chainscope

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[QUEUE]${NC} $1"
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

print_job() {
    echo -e "${PURPLE}[JOB]${NC} $1"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Queue IPHR experiments for multiple model pairs with enhanced LLM evaluation support.

Options:
    --models-file FILE          File containing model pairs (required)
    --num-pairs N               Number of question pairs per experiment (default: $DEFAULT_NUM_PAIRS)
    --responses-per-question N  Responses per question (default: $DEFAULT_RESPONSES_PER_QUESTION)
    --max-parallel N            Maximum parallel jobs (default: $MAX_PARALLEL_JOBS)
    --output-dir DIR            Output directory (default: $QUEUE_OUTPUT_DIR)
    --wait-between N            Seconds to wait between jobs (default: $WAIT_BETWEEN_JOBS)
    --enable-llm-evaluation     Enable LLM-based pattern analysis for all experiments
    --evaluator-model MODEL     Model to use for LLM evaluation (default: use normal model)
    --max-llm-analyses N        Max pairs to analyze with LLM per experiment (default: $DEFAULT_MAX_LLM_ANALYSES)
    --sequential-servers        Run only one server at a time (resource-constrained environments)
    --use-same-model-for-eval   Use the same model for generation and evaluation (saves resources)
    --dry-run                   Show what would be run without executing
    --resume                    Resume interrupted queue
    --oversample-unfaithful     Enable chainscope-style oversampling for unfaithful pairs
    --pattern-analysis-mode MODE Pattern analysis mode: basic, enhanced, chainscope (default: $DEFAULT_PATTERN_ANALYSIS_MODE)
    --oversample-responses N    Over-sample responses per question for unfaithful pairs (default: $DEFAULT_OVERSAMPLE_RESPONSES_PER_QUESTION)
    --help                      Show this help message

Models File Format:
    Each line should contain: normal_model_path,thinkedit_model_path,experiment_name
    Optionally add evaluator model: normal_model_path,thinkedit_model_path,experiment_name,evaluator_model
    Lines starting with # are ignored as comments.
    
    Example:
    # Model comparisons for IPHR experiments
    Qwen/Qwen3-0.6B,./models/Qwen3-ThinkEdit-0.6B,qwen3-0.6b
    Qwen/Qwen3-1.8B,./models/Qwen3-ThinkEdit-1.8B,qwen3-1.8b,Qwen/Qwen3-7B
    meta-llama/Llama-3.1-8B,./models/Llama-3.1-ThinkEdit-8B,llama3.1-8b

Resource Management Examples:
    # Minimal resources - use same model for evaluation
    $0 --models-file models.txt --enable-llm-evaluation --use-same-model-for-eval

    # Sequential servers with different evaluator
    $0 --models-file models.txt --enable-llm-evaluation --sequential-servers --evaluator-model "Qwen/Qwen3-1.5B"

    # Full resources - simultaneous servers (default)
    $0 --models-file models.txt --enable-llm-evaluation --evaluator-model "Qwen/Qwen3-1.5B"

Standard Examples:
    # Run experiments for models defined in models.txt
    $0 --models-file models.txt

    # Run with custom settings
    $0 --models-file models.txt --num-pairs 200 --max-parallel 2

    # Dry run to see what would be executed
    $0 --models-file models.txt --dry-run --enable-llm-evaluation

    # Resume interrupted queue
    $0 --models-file models.txt --resume
EOF
}

parse_models_file() {
    local models_file=$1
    local -n model_pairs_ref=$2
    
    if [ ! -f "$models_file" ]; then
        print_error "Models file not found: $models_file"
        exit 1
    fi
    
    print_status "Parsing models file: $models_file"
    
    local line_num=0
    while IFS= read -r line; do
        line_num=$((line_num + 1))
        
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Parse comma-separated values (support optional 4th column for evaluator model)
        IFS=',' read -r normal_model thinkedit_model experiment_name evaluator_model <<< "$line"
        
        # Trim whitespace
        normal_model=$(echo "$normal_model" | xargs)
        thinkedit_model=$(echo "$thinkedit_model" | xargs)
        experiment_name=$(echo "$experiment_name" | xargs)
        evaluator_model=$(echo "$evaluator_model" | xargs)
        
        # Validate required fields
        if [ -z "$normal_model" ] || [ -z "$thinkedit_model" ] || [ -z "$experiment_name" ]; then
            print_error "Invalid format at line $line_num: $line"
            print_error "Expected format: normal_model,thinkedit_model,experiment_name[,evaluator_model]"
            exit 1
        fi
        
        # Add to array (include evaluator model even if empty)
        model_pairs_ref+=("$normal_model|$thinkedit_model|$experiment_name|$evaluator_model")
        
    done < "$models_file"
    
    print_status "Loaded ${#model_pairs_ref[@]} model pairs"
}

create_job_script() {
    local normal_model=$1
    local thinkedit_model=$2
    local experiment_name=$3
    local job_output_dir=$4
    local num_pairs=$5
    local responses_per_question=$6
    local enable_llm_evaluation=$7
    local evaluator_model=$8
    local max_llm_analyses=$9
    local sequential_servers=${10}
    local use_same_model_for_eval=${11}
    local oversample_unfaithful=${12}
    local pattern_analysis_mode=${13}
    local oversample_responses=${14}
    
    local job_script="$job_output_dir/run_experiment.sh"
    
    # Build the experiment command
    local experiment_cmd="./run_iphr_experiment.sh \\
    --normal-model \"$normal_model\" \\
    --thinkedit-model \"$thinkedit_model\" \\
    --num-pairs $num_pairs \\
    --responses-per-question $responses_per_question \\
    --output-dir \"$job_output_dir\""
    
    # Add LLM evaluation options if enabled
    if [ "$enable_llm_evaluation" = "true" ]; then
        experiment_cmd="$experiment_cmd \\
    --enable-llm-evaluation \\
    --max-llm-analyses $max_llm_analyses \\
    --pattern-analysis-mode $pattern_analysis_mode"
        
        # Add evaluator model if specified
        if [ -n "$evaluator_model" ]; then
            experiment_cmd="$experiment_cmd \\
    --evaluator-model \"$evaluator_model\""
        fi
        
        # Add chainscope-style oversampling if enabled
        if [ "$oversample_unfaithful" = "true" ]; then
            experiment_cmd="$experiment_cmd \\
    --oversample-unfaithful \\
    --oversample-responses $oversample_responses"
        fi
        
        # Add resource management options
        if [ "$use_same_model_for_eval" = "true" ]; then
            experiment_cmd="$experiment_cmd \\
    --use-same-model-for-eval"
        elif [ "$sequential_servers" = "true" ]; then
            experiment_cmd="$experiment_cmd \\
    --sequential-servers"
        fi
    fi
    
    cat > "$job_script" << EOF
#!/bin/bash
set -e

# IPHR Experiment Job: $experiment_name
# Normal: $normal_model
# ThinkEdit: $thinkedit_model
EOF

    if [ "$enable_llm_evaluation" = "true" ]; then
        cat >> "$job_script" << EOF
# LLM Evaluation: Enabled
# Evaluator Model: ${evaluator_model:-"Same as generation models"}
# Max LLM Analyses: $max_llm_analyses
# Sequential Servers: $sequential_servers
# Use Same Model for Eval: $use_same_model_for_eval
EOF
    else
        cat >> "$job_script" << EOF
# LLM Evaluation: Disabled (statistical analysis only)
EOF
    fi

    cat >> "$job_script" << EOF

cd "\$(dirname "\$0")"
cd ../..  # Go back to steering-evals directory

echo "Starting IPHR experiment: $experiment_name"
echo "Normal model: $normal_model"
echo "ThinkEdit model: $thinkedit_model"
EOF

    if [ "$enable_llm_evaluation" = "true" ]; then
        cat >> "$job_script" << EOF
echo "LLM Evaluation: Enabled"
echo "Evaluator model: ${evaluator_model:-"Same as generation models"}"
echo "Max LLM analyses: $max_llm_analyses"
echo "Resource mode: $([ "$use_same_model_for_eval" = "true" ] && echo "Same model" || ([ "$sequential_servers" = "true" ] && echo "Sequential servers" || echo "Simultaneous servers"))"
EOF
    else
        cat >> "$job_script" << EOF
echo "LLM Evaluation: Disabled"
EOF
    fi

    cat >> "$job_script" << EOF
echo "Output directory: $job_output_dir"
echo "Timestamp: \$(date)"

# Run the experiment
$experiment_cmd

# Create completion marker
echo "Experiment completed: \$(date)" > "$job_output_dir/COMPLETED"
echo "Normal model: $normal_model" >> "$job_output_dir/COMPLETED"
echo "ThinkEdit model: $thinkedit_model" >> "$job_output_dir/COMPLETED"
EOF

    if [ "$enable_llm_evaluation" = "true" ]; then
        cat >> "$job_script" << EOF
echo "LLM evaluation: Enabled" >> "$job_output_dir/COMPLETED"
echo "Evaluator model: ${evaluator_model:-"Same as generation models"}" >> "$job_output_dir/COMPLETED"
EOF
    fi

    cat >> "$job_script" << EOF

echo "IPHR experiment $experiment_name completed successfully"
EOF

    chmod +x "$job_script"
    echo "$job_script"
}

run_job() {
    local job_script=$1
    local experiment_name=$2
    local job_output_dir=$3
    
    print_job "Starting experiment: $experiment_name"
    
    # Create status marker
    echo "RUNNING" > "$job_output_dir/STATUS"
    echo "Started: $(date)" >> "$job_output_dir/STATUS"
    
    # Run the job
    if bash "$job_script" > "$job_output_dir/job.log" 2>&1; then
        echo "COMPLETED" > "$job_output_dir/STATUS"
        echo "Completed: $(date)" >> "$job_output_dir/STATUS"
        print_success "Completed experiment: $experiment_name"
        return 0
    else
        echo "FAILED" > "$job_output_dir/STATUS"
        echo "Failed: $(date)" >> "$job_output_dir/STATUS"
        print_error "Failed experiment: $experiment_name"
        print_error "Check log: $job_output_dir/job.log"
        return 1
    fi
}

check_job_status() {
    local job_output_dir=$1
    
    if [ -f "$job_output_dir/COMPLETED" ]; then
        echo "COMPLETED"
    elif [ -f "$job_output_dir/STATUS" ]; then
        head -n 1 "$job_output_dir/STATUS"
    else
        echo "NOT_STARTED"
    fi
}

consolidate_results() {
    local output_dir=$1
    local -n model_pairs_ref=$2
    
    print_status "Consolidating results..."
    
    local summary_file="$output_dir/queue_summary.md"
    local results_json="$output_dir/consolidated_results.json"
    
    cat > "$summary_file" << EOF
# IPHR Queue Results Summary

**Generated:** $(date)
**Total Experiments:** ${#model_pairs_ref[@]}

## Experiment Results

| Experiment | Normal Model | ThinkEdit Model | Evaluator Model | Status | Consistency Improvement | Unfaithfulness Reduction |
|------------|--------------|-----------------|----------------|--------|------------------------|--------------------------|
EOF

    # Initialize JSON structure
    echo "{\"experiments\": [" > "$results_json"
    local first_entry=true
    
    for pair in "${model_pairs_ref[@]}"; do
        IFS='|' read -r normal_model thinkedit_model experiment_name line_evaluator_model <<< "$pair"
        local job_output_dir="$output_dir/$experiment_name"
        local status=$(check_job_status "$job_output_dir")
        
        # Use line-specific evaluator model if provided, otherwise use global one
        local effective_evaluator_model="${line_evaluator_model:-$evaluator_model}"
        
        # Add to markdown summary
        if [ "$status" = "COMPLETED" ]; then
            # Try to extract metrics from comparison results
            local comparison_file="$job_output_dir/comparison_analysis/model_comparison.json"
            local consistency_improvement="N/A"
            local unfaithfulness_reduction="N/A"
            
            if [ -f "$comparison_file" ] && command -v jq >/dev/null 2>&1; then
                consistency_improvement=$(jq -r '.differences.consistency_improvement // "N/A"' "$comparison_file" 2>/dev/null || echo "N/A")
                unfaithfulness_reduction=$(jq -r '.differences.unfaithfulness_reduction // "N/A"' "$comparison_file" 2>/dev/null || echo "N/A")
            fi
            
            echo "| $experiment_name | $normal_model | $thinkedit_model | $effective_evaluator_model | ✅ $status | $consistency_improvement | $unfaithfulness_reduction |" >> "$summary_file"
            
            # Add to JSON
            if [ "$first_entry" = false ]; then
                echo "," >> "$results_json"
            fi
            first_entry=false
            
            cat >> "$results_json" << EOF
    {
      "experiment_name": "$experiment_name",
      "normal_model": "$normal_model",
      "thinkedit_model": "$thinkedit_model",
      "evaluator_model": "$effective_evaluator_model",
      "status": "$status",
      "consistency_improvement": $consistency_improvement,
      "unfaithfulness_reduction": $unfaithfulness_reduction,
      "output_directory": "$job_output_dir"
    }
EOF
        else
            echo "| $experiment_name | $normal_model | $thinkedit_model | $effective_evaluator_model | ❌ $status | N/A | N/A |" >> "$summary_file"
        fi
    done
    
    echo "]}" >> "$results_json"
    
    # Add summary statistics
    cat >> "$summary_file" << EOF

## Summary Statistics

EOF
    
    local completed_count=0
    local failed_count=0
    local total_consistency_improvement=0
    local valid_improvements=0
    
    for pair in "${model_pairs_ref[@]}"; do
        IFS='|' read -r normal_model thinkedit_model experiment_name line_evaluator_model <<< "$pair"
        local job_output_dir="$output_dir/$experiment_name"
        local status=$(check_job_status "$job_output_dir")
        
        # Use line-specific evaluator model if provided, otherwise use global one
        local effective_evaluator_model="${line_evaluator_model:-$evaluator_model}"
        
        if [ "$status" = "COMPLETED" ]; then
            completed_count=$((completed_count + 1))
            
            # Try to sum up improvements
            local comparison_file="$job_output_dir/comparison_analysis/model_comparison.json"
            if [ -f "$comparison_file" ] && command -v jq >/dev/null 2>&1; then
                local improvement=$(jq -r '.differences.consistency_improvement // null' "$comparison_file" 2>/dev/null)
                if [ "$improvement" != "null" ] && [ "$improvement" != "N/A" ]; then
                    total_consistency_improvement=$(echo "$total_consistency_improvement + $improvement" | bc -l 2>/dev/null || echo "$total_consistency_improvement")
                    valid_improvements=$((valid_improvements + 1))
                fi
            fi
        else
            failed_count=$((failed_count + 1))
        fi
    done
    
    cat >> "$summary_file" << EOF
- **Completed Experiments:** $completed_count/${#model_pairs_ref[@]}
- **Failed Experiments:** $failed_count
- **Success Rate:** $(echo "scale=1; $completed_count * 100 / ${#model_pairs_ref[@]}" | bc -l)%
EOF

    if [ $valid_improvements -gt 0 ] && command -v bc >/dev/null 2>&1; then
        local avg_improvement=$(echo "scale=3; $total_consistency_improvement / $valid_improvements" | bc -l)
        echo "- **Average Consistency Improvement:** $avg_improvement" >> "$summary_file"
    fi
    
    cat >> "$summary_file" << EOF

## Individual Experiment Details

EOF
    
    for pair in "${model_pairs_ref[@]}"; do
        IFS='|' read -r normal_model thinkedit_model experiment_name line_evaluator_model <<< "$pair"
        local job_output_dir="$output_dir/$experiment_name"
        
        # Use line-specific evaluator model if provided, otherwise use global one
        local effective_evaluator_model="${line_evaluator_model:-$evaluator_model}"
        
        cat >> "$summary_file" << EOF
### $experiment_name

- **Normal Model:** \`$normal_model\`
- **ThinkEdit Model:** \`$thinkedit_model\`
- **Evaluator Model:** \`${effective_evaluator_model:-"Same as generation models"}\`
- **Status:** $(check_job_status "$job_output_dir")
- **Output Directory:** \`$job_output_dir\`
- **Log File:** \`$job_output_dir/job.log\`

EOF
        
        if [ -f "$job_output_dir/comparison_analysis/model_comparison.json" ]; then
            echo "- **Results:** Available in \`$job_output_dir/comparison_analysis/\`" >> "$summary_file"
        fi
        
        # Check if LLM analysis was performed
        if [ -f "$job_output_dir/normal_analysis/faithfulness_analysis.json" ]; then
            if grep -q "enhanced_patterns" "$job_output_dir/normal_analysis/faithfulness_analysis.json" 2>/dev/null; then
                echo "- **LLM Analysis:** Available in individual analysis directories" >> "$summary_file"
            fi
        fi
        
        echo "" >> "$summary_file"
    done
    
    print_success "Results consolidated:"
    print_success "  Summary: $summary_file"
    print_success "  JSON: $results_json"
}

main() {
    local models_file=""
    local num_pairs=$DEFAULT_NUM_PAIRS
    local responses_per_question=$DEFAULT_RESPONSES_PER_QUESTION
    local max_parallel=$MAX_PARALLEL_JOBS
    local output_dir=$QUEUE_OUTPUT_DIR
    local wait_between=$WAIT_BETWEEN_JOBS
    local enable_llm_evaluation=$DEFAULT_ENABLE_LLM_EVALUATION
    local evaluator_model=$DEFAULT_EVALUATOR_MODEL
    local max_llm_analyses=$DEFAULT_MAX_LLM_ANALYSES
    local sequential_servers=$DEFAULT_SEQUENTIAL_SERVERS
    local use_same_model_for_eval=$DEFAULT_USE_SAME_MODEL_FOR_EVAL
    local dry_run=false
    local resume=false
    local oversample_unfaithful=$DEFAULT_OVERSAMPLE_UNFAITHFUL
    local pattern_analysis_mode=$DEFAULT_PATTERN_ANALYSIS_MODE
    local oversample_responses=$DEFAULT_OVERSAMPLE_RESPONSES_PER_QUESTION
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --models-file)
                models_file="$2"
                shift 2
                ;;
            --num-pairs)
                num_pairs="$2"
                shift 2
                ;;
            --responses-per-question)
                responses_per_question="$2"
                shift 2
                ;;
            --max-parallel)
                max_parallel="$2"
                shift 2
                ;;
            --output-dir)
                output_dir="$2"
                shift 2
                ;;
            --wait-between)
                wait_between="$2"
                shift 2
                ;;
            --enable-llm-evaluation)
                enable_llm_evaluation=true
                shift
                ;;
            --evaluator-model)
                evaluator_model="$2"
                shift 2
                ;;
            --max-llm-analyses)
                max_llm_analyses="$2"
                shift 2
                ;;
            --sequential-servers)
                sequential_servers=true
                shift
                ;;
            --use-same-model-for-eval)
                use_same_model_for_eval=true
                sequential_servers=true  # Implied when using same model
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --resume)
                resume=true
                shift
                ;;
            --oversample-unfaithful)
                oversample_unfaithful=true
                shift
                ;;
            --pattern-analysis-mode)
                pattern_analysis_mode="$2"
                shift 2
                ;;
            --oversample-responses)
                oversample_responses="$2"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Validate required arguments
    if [ -z "$models_file" ]; then
        print_error "Models file is required. Use --models-file option."
        usage
        exit 1
    fi
    
    # Parse models file
    declare -a model_pairs
    parse_models_file "$models_file" model_pairs
    
    if [ ${#model_pairs[@]} -eq 0 ]; then
        print_error "No valid model pairs found in $models_file"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    print_status "IPHR Queue Configuration:"
    print_status "  Models file: $models_file"
    print_status "  Total experiments: ${#model_pairs[@]}"
    print_status "  Num pairs per experiment: $num_pairs"
    print_status "  Responses per question: $responses_per_question"
    print_status "  Max parallel jobs: $max_parallel"
    print_status "  Output directory: $output_dir"
    print_status "  Wait between jobs: ${wait_between}s"
    print_status "  LLM evaluation: $enable_llm_evaluation"
    
    if [ "$enable_llm_evaluation" = "true" ]; then
        print_status "  LLM Evaluation Settings:"
        if [ "$use_same_model_for_eval" = "true" ]; then
            print_status "    Mode: Use same model for evaluation (minimal resources)"
        elif [ "$sequential_servers" = "true" ]; then
            print_status "    Mode: Sequential servers (moderate resources)"
            print_status "    Evaluator model: ${evaluator_model:-"Default from models file or command line"}"
        else
            print_status "    Mode: Simultaneous servers (high resources)"
            print_status "    Evaluator model: ${evaluator_model:-"Default from models file or command line"}"
        fi
        print_status "    Max LLM analyses per experiment: $max_llm_analyses"
    fi
    
    if [ "$dry_run" = true ]; then
        print_warning "DRY RUN MODE - No experiments will be executed"
        echo ""
        print_status "Would execute the following experiments:"
        for pair in "${model_pairs[@]}"; do
            IFS='|' read -r normal_model thinkedit_model experiment_name line_evaluator_model <<< "$pair"
            local effective_evaluator_model="${line_evaluator_model:-$evaluator_model}"
            if [ "$enable_llm_evaluation" = "true" ]; then
                if [ "$use_same_model_for_eval" = "true" ]; then
                    echo "  - $experiment_name: $normal_model vs $thinkedit_model (LLM eval: same models)"
                else
                    echo "  - $experiment_name: $normal_model vs $thinkedit_model (LLM eval: ${effective_evaluator_model:-"default"})"
                fi
            else
                echo "  - $experiment_name: $normal_model vs $thinkedit_model (statistical only)"
            fi
        done
        exit 0
    fi
    
    # Prepare jobs
    declare -a job_scripts
    declare -a job_names
    declare -a job_dirs
    
    for pair in "${model_pairs[@]}"; do
        IFS='|' read -r normal_model thinkedit_model experiment_name line_evaluator_model <<< "$pair"
        local job_output_dir="$output_dir/$experiment_name"
        
        # Use line-specific evaluator model if provided, otherwise use global one
        local effective_evaluator_model="${line_evaluator_model:-$evaluator_model}"
        
        # Check if resuming and job already completed
        if [ "$resume" = true ] && [ "$(check_job_status "$job_output_dir")" = "COMPLETED" ]; then
            print_warning "Skipping completed experiment: $experiment_name"
            continue
        fi
        
        # Create job directory
        mkdir -p "$job_output_dir"
        
        # Create job script
        local job_script=$(create_job_script "$normal_model" "$thinkedit_model" "$experiment_name" "$job_output_dir" "$num_pairs" "$responses_per_question" "$enable_llm_evaluation" "$effective_evaluator_model" "$max_llm_analyses" "$sequential_servers" "$use_same_model_for_eval" "$oversample_unfaithful" "$pattern_analysis_mode" "$oversample_responses")
        
        job_scripts+=("$job_script")
        job_names+=("$experiment_name")
        job_dirs+=("$job_output_dir")
    done
    
    if [ ${#job_scripts[@]} -eq 0 ]; then
        print_success "All experiments already completed!"
        consolidate_results "$output_dir" model_pairs
        exit 0
    fi
    
    print_status "Starting ${#job_scripts[@]} experiments..."
    
    # Run jobs
    local running_jobs=0
    local completed_jobs=0
    local failed_jobs=0
    
    for i in "${!job_scripts[@]}"; do
        local job_script="${job_scripts[$i]}"
        local experiment_name="${job_names[$i]}"
        local job_output_dir="${job_dirs[$i]}"
        
        # Wait if we've reached max parallel jobs
        while [ $running_jobs -ge $max_parallel ]; do
            sleep 5
            
            # Check for completed jobs
            for j in "${!job_scripts[@]}"; do
                local check_dir="${job_dirs[$j]}"
                local status=$(check_job_status "$check_dir")
                
                if [ "$status" = "COMPLETED" ] || [ "$status" = "FAILED" ]; then
                    if [ "$status" = "COMPLETED" ]; then
                        completed_jobs=$((completed_jobs + 1))
                    else
                        failed_jobs=$((failed_jobs + 1))
                    fi
                    running_jobs=$((running_jobs - 1))
                    # Remove from active tracking (simple approach)
                fi
            done
        done
        
        # Start job
        if [ $max_parallel -eq 1 ]; then
            # Sequential execution
            if run_job "$job_script" "$experiment_name" "$job_output_dir"; then
                completed_jobs=$((completed_jobs + 1))
            else
                failed_jobs=$((failed_jobs + 1))
            fi
            
            # Wait between jobs if not the last one
            if [ $i -lt $((${#job_scripts[@]} - 1)) ] && [ $wait_between -gt 0 ]; then
                print_status "Waiting ${wait_between}s before next job..."
                sleep $wait_between
            fi
        else
            # Parallel execution
            run_job "$job_script" "$experiment_name" "$job_output_dir" &
            running_jobs=$((running_jobs + 1))
        fi
    done
    
    # Wait for remaining jobs to complete
    if [ $max_parallel -gt 1 ]; then
        print_status "Waiting for remaining jobs to complete..."
        wait
    fi
    
    # Final status check
    completed_jobs=0
    failed_jobs=0
    for job_dir in "${job_dirs[@]}"; do
        local status=$(check_job_status "$job_dir")
        if [ "$status" = "COMPLETED" ]; then
            completed_jobs=$((completed_jobs + 1))
        else
            failed_jobs=$((failed_jobs + 1))
        fi
    done
    
    print_status "=== QUEUE EXECUTION COMPLETE ==="
    print_success "Completed jobs: $completed_jobs"
    if [ $failed_jobs -gt 0 ]; then
        print_error "Failed jobs: $failed_jobs"
    fi
    
    # Consolidate results
    consolidate_results "$output_dir" model_pairs
    
    print_success "Queue execution finished!"
    print_success "Check summary: $output_dir/queue_summary.md"
}

# Run main function with all arguments
main "$@" 