# Getting Started with Reasoning Length Steering

This guide walks you through setting up and running your first reasoning length steering experiment.

## Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended: 16GB+ VRAM for larger models)
- 50GB+ disk space for models and results

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/manikyabard/steering-evals.git
cd steering-evals
```

2. **Create environment**
```bash
conda create -n reasoning-steering python=3.11
conda activate reasoning-steering
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install SGLang**

SGLang provides fast inference with batching support. Follow the official installation guide:
```bash
# Basic installation
pip install sglang[all]

# Or for development
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e ".[all]"
```

## Quick Start: Basic Steering Experiment

### Step 1: Start SGLang Server

```bash
# Start server with Qwen3-0.6B model
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --port 30000 \
    --reasoning-parser qwen3
```

**Note**: The server will download the model on first run (~1.3GB for 0.6B model).

### Step 2: Generate GSM8K Responses

```bash
# Generate 1000 responses for direction extraction
python src/generation/generate_responses_gsm8k.py \
    --model Qwen/Qwen3-0.6B \
    --num_samples 1000 \
    --batch_size 32
```

This creates `results/responses/Qwen3-0.6B_gsm8k_responses.json` with model responses.

### Step 3: Extract Direction Vectors

```bash
# Extract reasoning length directions
python src/extraction/extract_reasoning_length_direction_improved.py \
    --model Qwen/Qwen3-0.6B \
    --responses-file results/responses/Qwen3-0.6B_gsm8k_responses.json \
    --components attn
```

This creates direction vectors in `results/directions/` and generates visualization plots.

### Step 4: Test Steering Effects

```bash
# Apply steering with different strengths
python src/generation/steer_reasoning_length.py \
    --model Qwen/Qwen3-0.6B \
    --component attn \
    --direction_weights -0.08 -0.04 0 0.04 0.08 \
    --num_samples 100
```

Results will show how different steering strengths affect reasoning length and accuracy.

## Running Faithfulness Evaluation

### IPHR (Inverted Pair Harmful Reasoning) Evaluation

The IPHR evaluation tests whether models maintain logical consistency with reversed questions.

```bash
# Run complete IPHR experiment
./experiments/run_iphr_experiment.sh \
    --normal-model "Qwen/Qwen3-0.6B" \
    --thinkedit-model "path/to/thinkedit/model" \
    --num-pairs 50 \
    --enable-llm-evaluation \
    --max-llm-analyses 15
```

For resource-constrained environments:
```bash
./experiments/run_iphr_experiment.sh \
    --normal-model "Qwen/Qwen3-0.6B" \
    --thinkedit-model "path/to/thinkedit/model" \
    --num-pairs 25 \
    --enable-llm-evaluation \
    --use-same-model-for-eval \
    --max-llm-analyses 10
```

## Understanding the Results

### Steering Results

After running steering experiments, you'll find:

- **Performance plots**: Show accuracy vs. steering strength
- **Length analysis**: Reasoning length changes across steering values
- **Direction visualizations**: Heatmaps of attention head contributions

### IPHR Results

The faithfulness evaluation produces:

- **Consistency rates**: Percentage of logically consistent responses
- **Unfaithfulness patterns**: Categories of reasoning failures
- **Model comparisons**: Normal vs. ThinkEdit model analysis

## Common Workflows

### Research Workflow

1. **Generate responses** on your target dataset
2. **Extract directions** from paired short/long examples
3. **Test steering effects** on model performance
4. **Evaluate faithfulness** using IPHR methodology
5. **Analyze results** and generate visualizations

### Comparison Workflow

1. **Set up multiple models** (normal and modified versions)
2. **Run parallel experiments** using queue scripts
3. **Compare performance** across model sizes and modifications
4. **Analyze scaling effects** on reasoning and faithfulness

## Configuration

### Model Configuration

Edit `configs/model_config.yaml` to:
- Add new models
- Adjust generation parameters
- Configure evaluation settings

### Experiment Templates

Use templates in `experiments/`:
- `basic_steering.yaml`: Simple steering experiments
- `faithfulness_eval.yaml`: IPHR evaluation setup
- `model_comparison.yaml`: Multi-model comparisons

## Troubleshooting

### Server Connection Issues

```bash
# Check if server is running
curl http://localhost:30000/health

# If not responding, restart with logs
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --port 30000 \
    --reasoning-parser qwen3 \
    --log-level info
```

### Memory Issues

- **Reduce batch size** in generation scripts
- **Use gradient checkpointing** for direction extraction
- **Enable memory cleanup** with `--memory-cleanup-frequency`

### Direction Extraction Failures

- **Check response file format**: Ensure valid JSON structure
- **Verify sufficient examples**: Need both short (<100 tokens) and long (>1000 tokens) examples
- **Adjust thresholds**: Modify short/long token thresholds if needed

## Next Steps

Once you've run basic experiments:

1. **Try different models**: Scale up to larger Qwen3 models
2. **Experiment with components**: Test MLP directions alongside attention
3. **Custom datasets**: Generate responses on domain-specific problems
4. **Advanced analysis**: Dive into attention head analysis and faithfulness patterns

## Getting Help

- **Documentation**: Check `docs/` for detailed guides
- **Issues**: Review common troubleshooting in `docs/TROUBLESHOOTING.md`
- **Examples**: See `examples/` for complete workflow demonstrations

Happy experimenting! 