# Reasoning Length Steering and Faithfulness in Large Language Models

This repository contains code for experimenting with steering reasoning length in language models based on the ThinkEdit methodology. The code is designed to identify reasoning length direction vectors and use them to control the verbosity of model reasoning.

This repository contains our implementation and analysis of reasoning length steering techniques in large language models, with a focus on understanding how these interventions affect both performance and reasoning faithfulness.

## Research Overview

We investigate the relationship between reasoning length manipulation and faithfulness in language models by:

1. **Implementing ThinkEdit methodology** to extract and apply reasoning length steering vectors
2. **Developing IPHR (Inverted Pair Harmful Reasoning) evaluation** to measure reasoning faithfulness
3. **Analyzing the trade-offs** between reasoning length, accuracy, and logical consistency

### Key Findings

- **Model scale matters more than steering**: 0.6B→4B parameter scaling provides +12% accuracy improvement vs. ThinkEdit's ±1% changes
- **Concise reasoning can be effective**: Models maintain 93-100% accuracy even in their shortest 5% of responses
- **Faithfulness varies by model size**: Smaller models (0.6B) show concerning logical inconsistency under IPHR testing
- **Limited generalization**: ThinkEdit benefits appear task-specific and don't broadly improve logical consistency

## Repository Structure

```
steering-evals/
├── src/                              # Core implementation scripts
│   ├── generation/                   # Response generation
│   ├── extraction/                   # Direction vector extraction  
│   ├── evaluation/                   # Faithfulness evaluation
│   └── analysis/                     # Results analysis
├── experiments/                      # Experimental configurations
├── results/                          # Generated results and analysis
├── docs/                            # Documentation and guides
└── configs/                         # Configuration files
```

## Quick Start

### 1. Environment Setup

```bash
# Create and activate environment
conda create -n reasoning-steering python=3.11
conda activate reasoning-steering

# Install dependencies
pip install -r requirements.txt

# Install SGLang (for fast inference)
# Follow instructions at: https://docs.sglang.ai/backend/installation.html
```

### 2. Basic Steering Experiment

```bash
# Start SGLang server
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --reasoning-parser qwen3

# Generate responses on GSM8K
python src/generation/generate_responses_gsm8k.py \
    --model Qwen/Qwen3-0.6B \
    --num_samples 1000

# Extract reasoning direction vectors
python src/extraction/extract_reasoning_length_direction_improved.py \
    --model Qwen/Qwen3-0.6B \
    --responses-file responses/Qwen3-0.6B_gsm8k_responses.json

# Test steering effects
python src/generation/steer_reasoning_length.py \
    --model Qwen/Qwen3-0.6B \
    --direction_weights -0.08 -0.04 0 0.04 0.08
```

### 3. Faithfulness Evaluation

```bash
# Run IPHR faithfulness evaluation
./experiments/run_iphr_experiment.sh \
    --normal-model "Qwen/Qwen3-0.6B" \
    --thinkedit-model "path/to/thinkedit/model" \
    --enable-llm-evaluation \
    --num-pairs 100
```

## Core Components

### Reasoning Length Steering (ThinkEdit Implementation)

Our implementation follows the ThinkEdit methodology with improvements:

- **Direction Extraction**: Identify neural directions that correlate with reasoning length
- **Attention Head Analysis**: Find specific heads that bias toward short reasoning
- **Targeted Interventions**: Apply steering during inference or edit model weights

Key scripts:
- `src/extraction/extract_reasoning_length_direction_improved.py`
- `src/analysis/find_short_thinking_attn_heads_qwen3.py`
- `src/generation/steer_reasoning_length.py`

### Faithfulness Evaluation (IPHR)

We developed an enhanced IPHR system inspired by ChainScope for measuring reasoning faithfulness:

- **Inverted Question Pairs**: Test logical consistency with reversed comparative questions
- **Pattern Detection**: Identify fact manipulation, argument switching, and reasoning shortcuts
- **LLM-based Analysis**: Use evaluator models to detect sophisticated unfaithfulness patterns

Key scripts:
- `src/evaluation/create_factual_question_pairs.py`
- `src/evaluation/generate_iphr_responses.py`
- `src/evaluation/evaluate_iphr_faithfulness.py`

## Experimental Results

### Model Performance Summary

| Model | Normal Accuracy | ThinkEdit Accuracy | Normal Length | ThinkEdit Length |
|-------|----------------|-------------------|---------------|------------------|
| Qwen3-0.6B | 83.25% | 82.15% (-1.1%) | 1559 tokens | 1608 (+3.1%) |
| Qwen3-4B | 94.95% | 95.35% (+0.4%) | 1760 tokens | 1765 (+0.3%) |
| Qwen3-8B | 96.05% | 95.55% (-0.5%) | 1852 tokens | 1996 (+7.8%) |

### Faithfulness Results (IPHR)

- **0.6B models**: Show concerning unfaithfulness patterns (-15.4% consistency)
- **4B+ models**: Maintain better logical consistency (minimal IPHR impact)
- **ThinkEdit impact**: Limited improvement in faithfulness across model sizes

See `results/analysis_results_*` directories for detailed breakdowns.

## Configuration

### Model Configuration

Edit `configs/model_config.yaml`:

```yaml
models:
  qwen3_0_6b:
    name: "Qwen/Qwen3-0.6B"
    reasoning_parser: "qwen3"
    max_tokens: 32768
  
generation:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  batch_size: 64
```

### Experiment Configuration

Use experiment templates in `experiments/`:
- `basic_steering.yaml`: Simple steering experiments
- `faithfulness_eval.yaml`: IPHR faithfulness evaluation
- `model_comparison.yaml`: Multi-model comparison

## Advanced Usage

### Custom Direction Extraction

```python
from src.extraction.direction_extractor import ReasoningDirectionExtractor

extractor = ReasoningDirectionExtractor(
    model_name="Qwen/Qwen3-0.6B",
    component="attn",  # or "mlp"
    short_threshold=100,
    long_threshold=1000
)

directions = extractor.extract_from_responses("responses.json")
```

### Faithfulness Analysis

```python
from src.evaluation.iphr_evaluator import IPHRFaithfulnessEvaluator

evaluator = IPHRFaithfulnessEvaluator(
    evaluator_model="Qwen/Qwen3-4B",
    enable_llm_analysis=True
)

results = evaluator.evaluate_model_pair(
    normal_responses="normal_responses.json",
    thinkedit_responses="thinkedit_responses.json"
)
```

## Visualization and Analysis

Generated visualizations include:
- **Attention head heatmaps**: Show which heads bias toward short reasoning
- **Direction magnitude plots**: Visualize steering vectors across model layers
- **IPHR results dashboards**: Compare faithfulness across models
- **Scaling analysis plots**: Demonstrate model size vs. performance relationships

Access visualization scripts in `src/analysis/` and view results in `results/visualizations/`.

## Troubleshooting

### Common Issues

1. **SGLang Connection Errors**
   ```bash
   # Check server status
   curl http://localhost:30000/health
   
   # Restart with specific port
   python -m sglang.launch_server --model-path MODEL --port 30001
   ```

2. **Memory Issues**
   - Reduce batch size in generation scripts
   - Use `--use-gradient-checkpointing` for direction extraction
   - Enable memory cleanup with `--memory-cleanup-frequency`

3. **Direction Extraction Failures**
   - Ensure sufficient examples in both short/long categories
   - Check response file format and content
   - Verify model compatibility with extraction script

### Performance Optimization

- **Use SGLang**: Significantly faster than standard HuggingFace generation
- **Batch processing**: Adjust batch sizes based on available GPU memory
- **Resume functionality**: All scripts support resuming from checkpoints

## Citation

If you use this work in your research, please cite:

Parameters specific to `generate_responses_gsm8k.py`:
- `--batch_size`: Number of requests to process in parallel (default: 4)
- `--server_url`: URL of the SGLang server (default: "http://localhost:30000")
- `--temperature`: Temperature for sampling (default: 0.6)
- `--top_p`: Top-p (nucleus) sampling parameter (default: 0.95)
- `--top_k`: Top-k sampling parameter (default: 20)

## Acknowledgments

This work builds upon:
- **ThinkEdit** by Sun, Yan, and Weng (2025)
- **ChainScope** by Arcuschin et al. (2025)

We thank the authors for making their methodologies available and inspiring this research.

## License

MIT License - see [LICENSE](LICENSE) for details.