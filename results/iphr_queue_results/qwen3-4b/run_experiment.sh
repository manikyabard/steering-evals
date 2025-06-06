#!/bin/bash
set -e

# IPHR Experiment Job: qwen3-4b
# Normal: /home/mbardhan/teams/the-most-amazing-group/qwen3_4b
# ThinkEdit: /home/mbardhan/teams/the-most-amazing-group/thinkedit_models/ThinkEdit-Qwen_Qwen3_4B
# LLM Evaluation: Enabled
# Evaluator Model: /home/mbardhan/teams/the-most-amazing-group/qwen3_4b
# Max LLM Analyses: 100
# Sequential Servers: true
# Use Same Model for Eval: false

cd "$(dirname "$0")"
cd ../..  # Go back to steering-evals directory

echo "Starting IPHR experiment: qwen3-4b"
echo "Normal model: /home/mbardhan/teams/the-most-amazing-group/qwen3_4b"
echo "ThinkEdit model: /home/mbardhan/teams/the-most-amazing-group/thinkedit_models/ThinkEdit-Qwen_Qwen3_4B"
echo "LLM Evaluation: Enabled"
echo "Evaluator model: /home/mbardhan/teams/the-most-amazing-group/qwen3_4b"
echo "Max LLM analyses: 100"
echo "Resource mode: Sequential servers"
echo "Output directory: iphr_queue_results/qwen3-4b"
echo "Timestamp: $(date)"

# Run the experiment
./run_iphr_experiment.sh \
    --normal-model "/home/mbardhan/teams/the-most-amazing-group/qwen3_4b" \
    --thinkedit-model "/home/mbardhan/teams/the-most-amazing-group/thinkedit_models/ThinkEdit-Qwen_Qwen3_4B" \
    --num-pairs 100 \
    --responses-per-question 10 \
    --output-dir "iphr_queue_results/qwen3-4b" \
    --enable-llm-evaluation \
    --max-llm-analyses 100 \
    --pattern-analysis-mode chainscope \
    --evaluator-model "/home/mbardhan/teams/the-most-amazing-group/qwen3_4b" \
    --oversample-unfaithful \
    --oversample-responses 100 \
    --sequential-servers

# Create completion marker
echo "Experiment completed: $(date)" > "iphr_queue_results/qwen3-4b/COMPLETED"
echo "Normal model: /home/mbardhan/teams/the-most-amazing-group/qwen3_4b" >> "iphr_queue_results/qwen3-4b/COMPLETED"
echo "ThinkEdit model: /home/mbardhan/teams/the-most-amazing-group/thinkedit_models/ThinkEdit-Qwen_Qwen3_4B" >> "iphr_queue_results/qwen3-4b/COMPLETED"
echo "LLM evaluation: Enabled" >> "iphr_queue_results/qwen3-4b/COMPLETED"
echo "Evaluator model: /home/mbardhan/teams/the-most-amazing-group/qwen3_4b" >> "iphr_queue_results/qwen3-4b/COMPLETED"

echo "IPHR experiment qwen3-4b completed successfully"
