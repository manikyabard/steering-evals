# Reasoning Length Steering Experiments

This repository contains code for experimenting with steering reasoning length in language models based on the ThinkEdit methodology. The code is designed to identify reasoning length direction vectors and use them to control the verbosity of model reasoning.

## Setup

1. Create a conda environment and activate it:
```bash
conda create -n steer python=3.11
conda activate steer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Install SGLang by following the instructions [here](https://docs.sglang.ai/backend/installation.html).

## Available Scripts

The codebase consists of three main scripts:

1. `generate_responses_gsm8k.py`: Generates and saves model responses from GSM8K with thinking enabled using SGLang for faster inference and parallelization.
2. `extract_reasoning_length_direction.py`: Uses the paired responses to extract the reasoning length direction vectors.
3. `steer_reasoning_length.py`: Applies the extracted directions during generation to control reasoning length.

## Workflow

### Step 1: Generate Responses

First, start the SGLang server in a separate terminal:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --reasoning-parser qwen3
```

Then generate responses from the GSM8K dataset with thinking enabled:

```bash
python generate_responses_gsm8k.py --model Qwen/Qwen3-0.6B --num_samples 2000 --batch_size 64
```

This will create a JSON file with responses in the `responses` directory. The SGLang implementation offers:
- Faster inference with batch processing
- Automatic separation of thinking from responses using a reasoning parser
- Configurable sampling parameters (Temperature, TopP, TopK)

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

Parameters specific to `generate_responses_gsm8k.py`:
- `--batch_size`: Number of requests to process in parallel (default: 4)
- `--server_url`: URL of the SGLang server (default: "http://localhost:30000")
- `--temperature`: Temperature for sampling (default: 0.6)
- `--top_p`: Top-p (nucleus) sampling parameter (default: 0.95)
- `--top_k`: Top-k sampling parameter (default: 20)

Run any script with `--help` to see all available options.

## Converting to Jupyter Notebooks

All scripts are written with `#%%` cell markers and markdown sections, so they can be directly opened and run as notebooks in environments that support this format, such as VS Code or PyCharm.

## Acknowledgements

This implementation is based on the methodology described in the paper "ThinkEdit: Interpretable Weight Editing to Mitigate Overly Short Thinking in Reasoning Models" by Chung-En Sun, Ge Yan, and Tsui-Wei Weng.

IPHR evaluation is based on the methodology described in the paper "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful" by Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, et al.
