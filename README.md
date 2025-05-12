# Reasoning Length Steering Experiments

This repository contains code for experimenting with steering reasoning length in language models based on the ThinkEdit methodology. The code is designed to identify reasoning length direction vectors and use them to control the verbosity of model reasoning.

## Setup

1. Create a conda environment and activate it:
```bash
conda create -n steer python=3.10
conda activate steer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Available Scripts

The codebase consists of three main scripts:

1. `generate_responses_gsm8k.py`: Generates and saves model responses from GSM8K, both with and without the "thinking" mode.
2. `extract_reasoning_length_direction.py`: Uses the paired responses to extract the reasoning length direction vectors.
3. `steer_reasoning_length.py`: Applies the extracted directions during generation to control reasoning length.

## Workflow

### Step 1: Generate Responses

Generate responses from the GSM8K dataset with both thinking and non-thinking modes:

```bash
python generate_responses_gsm8k.py --model Qwen/Qwen3-0.6B --num_samples 100
```

This will create a JSON file with paired responses in the `responses` directory.

### Step 2: Extract Reasoning Length Direction

Extract the reasoning length direction from the paired responses:

```bash
python extract_reasoning_length_direction.py --model Qwen/Qwen3-0.6B --num_samples 50
```

This will create direction vectors saved as a PyTorch file in the `directions` directory and also generate visualization plots.

### Step 3: Experiment with Steering

Apply the extracted direction during generation to control reasoning length:

```bash
python steer_reasoning_length.py --model Qwen/Qwen3-0.6B --component attn --direction_weights -0.08 -0.04 0 0.04 0.08
```

This will generate responses with varying steering strengths, evaluate them, and produce visualizations showing the relationship between steering strength, reasoning length, and accuracy.

## Parameters

All scripts support various command-line arguments. Here are some important ones:

- `--model`: Model name or path (default: "Qwen/Qwen3-0.6B")
- `--num_samples`: Number of samples to process (default varies by script)
- `--component`: Which components to steer (attn, mlp, or both) (default: attn)
- `--direction_weights`: List of steering strengths to apply (default: [-0.08, -0.04, 0, 0.04, 0.08])

Run any script with `--help` to see all available options.

## Converting to Jupyter Notebooks

All scripts are written with `#%%` cell markers and markdown sections, so they can be directly opened and run as notebooks in environments that support this format, such as VS Code or PyCharm.

## Acknowledgements

This implementation is based on the methodology described in the paper "ThinkEdit: Interpretable Weight Editing to Mitigate Overly Short Thinking in Reasoning Models" by Chung-En Sun, Ge Yan, and Tsui-Wei Weng.