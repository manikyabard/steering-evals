# Steering vs Faithfulness Integration Guide

This guide explains how to integrate reasoning length steering techniques with ChainScope's faithfulness evaluation to investigate your research question: **Does steering reasoning length affect the faithfulness of the reasoning process?**

## Overview

Your research combines two powerful approaches:

1. **Steering Techniques** (from ThinkEdit): Control reasoning length via direction vectors or weight editing
2. **Faithfulness Evaluation** (from ChainScope): Detect restoration errors and unfaithful reasoning

## Integration Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Steering      │    │   ChainScope     │    │   Analysis      │
│   Techniques    │───▶│   Evaluation     │───▶│   & Results     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
│                      │                       │
├─ Runtime Steering    ├─ Answer Correctness   ├─ Correlation Plots
├─ Weight Editing      ├─ Step Evaluation      ├─ Faithfulness Trends  
└─ Direction Vectors   └─ Restoration Errors   └─ Research Insights
```

## Two Steering Approaches

### 1. Runtime Steering (`steer_reasoning_length.py`)

**How it works:**
- Loads pre-extracted direction vectors that represent reasoning length
- Applies steering hooks during model inference
- Controlled by `alpha` parameter (negative = shorter, positive = longer)

**Advantages:**
- Fine-grained control over steering strength
- Can test multiple α values with same model
- Non-destructive (original model unchanged)

**Usage:**
```python
# Apply runtime steering with α = 0.08 for longer reasoning
hooks = apply_steering_layers(model, directions, alpha=0.08, component="attn")
# Generate with steering applied
response = model.generate(...)
remove_steering_layers(hooks)
```

### 2. Weight Editing (`get_thinkedit_qwen3_models.py`)

**How it works:**
- Permanently modifies attention head weights
- Removes projection along reasoning length direction
- Creates a new model with altered reasoning behavior

**Advantages:**
- Persistent changes (no need to apply during inference)
- Can be distributed as a modified model
- More computationally efficient for repeated use

**Usage:**
```bash
# Create ThinkEdit model
python get_thinkedit_qwen3_models.py --model Qwen/Qwen3-0.6B --intervention_weight 1.0

# Use the edited model normally
model = AutoModelForCausalLM.from_pretrained("thinkedit_models/ThinkEdit-Qwen3-0.6B")
```

## ChainScope's Restoration Errors Pipeline

ChainScope evaluates faithfulness through a sophisticated multi-pass system:

### Pass 0: Answer Correctness
- Checks if the final answer matches the correct solution
- Classifies problem descriptions as CLEAR/INCOMPLETE/AMBIGUOUS

### Pass 1: Step-by-Step Evaluation
- Evaluates each reasoning step individually
- Marks steps as correct/incorrect/uncertain

### Pass 2: Restoration Error Detection
- Identifies unfaithful steps that contain errors but are implicitly corrected later
- Categorizes by type: `unused`, `unfaithful`, `incorrect`
- Rates severity: `trivial`, `minor`, `major`, `critical`

### Pass 3: Re-examination
- Double-checks flagged steps for final classification
- Reduces false positives through careful re-analysis

## Key Faithfulness Metrics

ChainScope identifies several types of unfaithfulness:

1. **Implicit Post-Hoc Rationalization**: Model decides answer first, then constructs reasoning
2. **Restoration Errors**: Model makes mistakes but silently corrects them later
3. **Unfaithful Shortcuts**: Logical leaps or invalid reasoning steps

## Your Research Questions

### Primary Question
**Does steering reasoning length affect faithfulness?**

**Hypothesis Testing:**
- **H1**: Longer reasoning (positive α) increases restoration errors due to more opportunities for mistakes
- **H2**: Shorter reasoning (negative α) increases unfaithful shortcuts due to missing steps
- **H3**: There exists an optimal α that maximizes both accuracy and faithfulness

### Experimental Design

#### Independent Variables:
- **Steering Strength (α)**: Range from -0.15 to +0.15
- **Steering Method**: Runtime vs Weight Editing
- **Model Architecture**: Qwen3-0.6B, 1.5B, 14B

#### Dependent Variables:
- **Accuracy**: % of correct final answers
- **Unfaithfulness Rate**: % of steps marked as unfaithful
- **Reasoning Length**: Average words per reasoning chain
- **Error Types**: Distribution of unfaithfulness categories

## Practical Implementation

### Step 1: Setup
```bash
# Setup environment
./setup_experiment.sh setup

# Check dependencies
./setup_experiment.sh demo
```

### Step 2: Generate Steered CoT Paths
```python
# Generate paths with different steering strengths
python gen_steered_cot_paths.py \
    -n 10 -d gsm8k -m Qwen/Qwen3-0.6B \
    --steering-mode runtime --alpha 0.08 --component attn
```

### Step 3: Evaluate Faithfulness
```python
# Run ChainScope evaluation
python chainscope/scripts/restoration_errors/eval_cot_paths.py \
    chainscope/chainscope/data/cot_paths/gsm8k/Qwen__Qwen3-0.6B_steered_alpha_0.08_attn.yaml \
    --evaluator_model_id anthropic/claude-3.5-sonnet --anthropic
```

### Step 4: Comprehensive Analysis
```bash
# Run full experiment pipeline
python experiment_steering_faithfulness.py \
    --model Qwen/Qwen3-0.6B \
    --alpha-values -0.1 -0.05 0.0 0.05 0.1 0.15 \
    --experiment-type full --anthropic
```

## Expected Results & Insights

### Potential Findings:

1. **Accuracy vs Steering**: 
   - Mild positive steering may improve accuracy
   - Extreme steering (|α| > 0.1) may hurt performance

2. **Faithfulness vs Steering**:
   - Positive α might increase restoration errors
   - Negative α might increase unfaithful shortcuts
   - Sweet spot around α ≈ 0.05 for optimal faithfulness

3. **Model Scale Effects**:
   - Larger models may be more robust to steering
   - Different optimal α values for different model sizes

### Visualization Examples:

```python
# Expected plot: Correlation between steering and faithfulness
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(alpha_values, accuracy_scores, 'bo-')
plt.xlabel('Steering Strength (α)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Steering')

plt.subplot(1, 3, 2)
plt.plot(alpha_values, unfaithfulness_rates, 'ro-')
plt.xlabel('Steering Strength (α)')
plt.ylabel('Unfaithfulness Rate')
plt.title('Faithfulness vs Steering')

plt.subplot(1, 3, 3)
plt.plot(alpha_values, reasoning_lengths, 'go-')
plt.xlabel('Steering Strength (α)')
plt.ylabel('Reasoning Length')
plt.title('Length vs Steering')
```

## Advanced Analyses

### 1. Unfaithfulness Type Analysis
```python
# Analyze specific types of unfaithfulness by steering strength
unfaithfulness_by_type = {
    'unused': [],
    'unfaithful': [],
    'incorrect': []
}

for alpha in alpha_values:
    # Count each type for this alpha
    type_counts = analyze_unfaithfulness_types(results[alpha])
    for utype, count in type_counts.items():
        unfaithfulness_by_type[utype].append(count)
```

### 2. Model Architecture Comparison
```python
# Compare faithfulness across model sizes
models = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.5B", "Qwen/Qwen3-14B"]
for model in models:
    results = run_steering_experiment(model, alpha_values)
    analyze_faithfulness_trends(results, model)
```

### 3. Problem Difficulty Analysis
```python
# Test if steering effects vary by problem difficulty
easy_problems = filter_problems_by_difficulty(dataset, "easy")
hard_problems = filter_problems_by_difficulty(dataset, "hard")

for difficulty, problems in [("easy", easy_problems), ("hard", hard_problems)]:
    results = run_steering_on_subset(problems, alpha_values)
    print(f"Steering effects on {difficulty} problems:")
    analyze_results(results)
```

## Safety Considerations

### Mitigation Strategies:
1. **Gradual Steering**: Start with small α values (±0.05)
2. **Multiple Metrics**: Don't optimize for faithfulness alone
3. **Human Evaluation**: Validate automated faithfulness detection
4. **Robustness Testing**: Test across multiple datasets and models

### Potential Risks:
- Over-steering may make models produce nonsensical reasoning
- Faithfulness evaluation may miss subtle forms of unfaithfulness
- Results may not generalize across different reasoning domains

## File Organization

```
steering-evals/
├── chainscope/                          # ChainScope evaluation code
├── directions/                          # Steering direction vectors
├── thinkedit_models/                    # Weight-edited models
├── steering_faithfulness_results/       # Experiment results
├── gen_steered_cot_paths.py            # Custom CoT generation with steering
├── experiment_steering_faithfulness.py  # Main experiment script
├── setup_experiment.sh                 # Setup and run script
└── INTEGRATION_GUIDE.md               # This guide
```

## Quick Start Commands

```bash
# 1. Setup environment
./setup_experiment.sh setup

# 2. Run pilot experiment
./setup_experiment.sh pilot Qwen/Qwen3-0.6B

# 3. Run full experiment
./setup_experiment.sh full Qwen/Qwen3-0.6B

# 4. Test ThinkEdit approach
./setup_experiment.sh thinkedit Qwen/Qwen3-0.6B
```

## Contributing to Research

This integration enables several research directions:

1. **Core Question**: Steering ↔ Faithfulness relationship
2. **Model Scaling**: How does the relationship change with model size?
3. **Domain Transfer**: Do findings generalize beyond math reasoning?
4. **Intervention Design**: Can we design better steering techniques?
5. **Safety Applications**: Use faithfulness as a safety metric for AI systems

## Troubleshooting

### Common Issues:

1. **Missing Direction Vectors**: 
   - Need to run `extract_reasoning_length_direction_improved.py` first
   - See ThinkEdit repository for this script

2. **API Rate Limits**:
   - Use `--max-retries` and implement exponential backoff
   - Consider using local models for evaluation

3. **Memory Issues**:
   - Reduce batch size in steering generation
   - Use gradient checkpointing for large models

4. **Evaluation Consistency**:
   - Use deterministic evaluation (temperature=0.0)
   - Run multiple seeds and average results

---

This integration provides a comprehensive framework for investigating how reasoning length steering affects faithfulness. The combination of ThinkEdit's steering techniques with ChainScope's sophisticated faithfulness evaluation creates a powerful research platform for understanding AI reasoning reliability. 