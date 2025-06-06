#!/bin/bash
set -e

# IPHR Experiment Job: qwen3-0.6b
# Normal: Qwen/Qwen3-0.6B
# ThinkEdit: /home/mbardhan/teams/the-most-amazing-group/thinkedit_models/ThinkEdit-Qwen_Qwen3_0.6B
# LLM Evaluation: Enabled
# Evaluator Model: /home/mbardhan/teams/the-most-amazing-group/qwen3_8b
# Max LLM Analyses: 100
# Sequential Servers: true
# Use Same Model for Eval: false

cd "$(dirname "$0")"
cd ../..  # Go back to steering-evals directory

echo "Starting IPHR experiment: qwen3-0.6b"
echo "Normal model: Qwen/Qwen3-0.6B"
echo "ThinkEdit model: /home/mbardhan/teams/the-most-amazing-group/thinkedit_models/ThinkEdit-Qwen_Qwen3_0.6B"
echo "LLM Evaluation: Enabled"
echo "Evaluator model: /home/mbardhan/teams/the-most-amazing-group/qwen3_8b"
echo "Max LLM analyses: 100"
echo "Resource mode: Sequential servers"
echo "Output directory: iphr_queue_results/qwen3-0.6b"
echo "Timestamp: $(date)"

# Run the experiment
./run_iphr_experiment.sh \
    --normal-model "Qwen/Qwen3-0.6B" \
    --thinkedit-model "/home/mbardhan/teams/the-most-amazing-group/thinkedit_models/ThinkEdit-Qwen_Qwen3_0.6B" \
    --num-pairs 100 \
    --responses-per-question 10 \
    --output-dir "iphr_queue_results/qwen3-0.6b" \
    --enable-llm-evaluation \
    --max-llm-analyses 100 \
    --evaluator-model "/home/mbardhan/teams/the-most-amazing-group/qwen3_8b" \
    --sequential-servers

# Create completion marker
echo "Experiment completed: $(date)" > "iphr_queue_results/qwen3-0.6b/COMPLETED"
echo "Normal model: Qwen/Qwen3-0.6B" >> "iphr_queue_results/qwen3-0.6b/COMPLETED"
echo "ThinkEdit model: /home/mbardhan/teams/the-most-amazing-group/thinkedit_models/ThinkEdit-Qwen_Qwen3_0.6B" >> "iphr_queue_results/qwen3-0.6b/COMPLETED"
echo "LLM evaluation: Enabled" >> "iphr_queue_results/qwen3-0.6b/COMPLETED"
echo "Evaluator model: /home/mbardhan/teams/the-most-amazing-group/qwen3_8b" >> "iphr_queue_results/qwen3-0.6b/COMPLETED"

echo "IPHR experiment qwen3-0.6b completed successfully"
