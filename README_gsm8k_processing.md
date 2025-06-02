# GSM8K Response Processing Pipeline

This document explains how to use the updated scripts for generating, processing, and validating GSM8K responses with improved answer parsing capabilities.

## Overview

The pipeline consists of three main scripts:

1. **`generate_responses_gsm8k.py`** - Generate model responses with thinking
2. **`postprocess_responses.py`** - Add improved answer parsing to existing response files
3. **`validate_single_model_sglang.py`** - Validate model performance on selected questions

## Features

### Improved Answer Parsing

All scripts now include robust answer extraction using multiple strategies:

- **GSM8K format**: Extracts answers in the standard `#### answer` format
- **Boxed format**: Extracts answers from `\\boxed{answer}` format
- **Fallback extraction**: Extracts the last number found in the response
- **Floating point comparison**: Handles numerical comparisons with proper tolerance

### Enhanced Metrics

The scripts now provide:
- Extracted answers for each response
- Correctness evaluation (boolean)
- Thinking length metrics (word count, character count, reasoning steps)
- Accuracy statistics
- Response length analysis

## Usage

### 1. Generate Responses

First, start your SGLang server:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --port 30000 \
    --reasoning-parser qwen3
```

Then generate responses:

```bash
python generate_responses_gsm8k.py \
    --model Qwen/Qwen3-0.6B \
    --num_samples 100 \
    --output_dir responses \
    --batch_size 32
```

#### Resume Functionality

The script now supports resuming interrupted runs:

```bash
# Auto-resume from the last processed ID
python generate_responses_gsm8k.py \
    --model Qwen/Qwen3-0.6B \
    --num_samples 1000 \
    --output_dir responses

# Resume from a specific ID
python generate_responses_gsm8k.py \
    --model Qwen/Qwen3-0.6B \
    --num_samples 1000 \
    --output_dir responses \
    --resume_from_id 500

# Force resume from specific ID even if no existing file
python generate_responses_gsm8k.py \
    --model Qwen/Qwen3-0.6B \
    --num_samples 1000 \
    --output_dir responses \
    --resume_from_id 500 \
    --force_resume
```

**Resume Features:**
- **Auto-detection**: Automatically detects existing output files and resumes from the next unprocessed ID
- **Manual override**: Specify exact resume point with `--resume_from_id`
- **Skip processed**: Intelligently skips already completed examples
- **Maintains order**: Output file is always sorted by ID for consistency
- **Safe operation**: Won't overwrite existing valid responses

**New Output Format:**
```json
{
  "id": 0,
  "question": "If there are 5 apples...",
  "answer": "#### 2", 
  "with_thinking": {
    "thinking": "Let me think about this...",
    "response": "The answer is \\boxed{2}",
    "extracted_answer": 2,
    "correct": true,
    "full_text": "complete generated text..."
  }
}
```

### 2. Post-process Existing Files (Optional)

If you have response files generated before the improvements, you can add the new fields:

```bash
# Process existing file
python postprocess_responses.py \
    --input_file responses/Qwen3-0.6B_gsm8k_responses.json \
    --output_file responses/Qwen3-0.6B_gsm8k_responses_processed.json

# Or overwrite the original (with backup)
python postprocess_responses.py \
    --input_file responses/Qwen3-0.6B_gsm8k_responses.json \
    --overwrite_input \
    --backup_original
```

### 3. Validate Model Performance

Test the model on the questions with shortest responses:

```bash
python validate_single_model_sglang.py \
    --port 30000 \
    --model_name original \
    --num_questions 6 \
    --response_file responses/Qwen3-0.6B_gsm8k_responses.json \
    --output_dir validation_results
```

## Output Files

### Response Generation
- `{model}_gsm8k_responses.json` - Main response file with all data
- Logs with real-time accuracy and thinking length statistics

### Post-processing  
- `{input_file}_processed.json` - Updated response file with new fields
- `{input_file}_processed_analysis.json` - Statistical analysis
- `{input_file}.backup` - Original file backup (if requested)

### Validation
- `{model_name}_results.json` - Detailed validation results
- `{model_name}_analysis.json` - Statistical analysis
- `{model_name}_summary.txt` - Human-readable summary

## Example Analysis Output

```
=== FINAL RESULTS ===
Total examples: 100
Final accuracy: 85.0% (85/100)
Average thinking length: 127.3 words
Thinking length range: 15-456 words
Response length range: 8-89 words
```

## Key Improvements

1. **Robust Answer Extraction**: Handles multiple answer formats commonly used by LLMs
2. **Real-time Accuracy**: See accuracy statistics during generation
3. **Comprehensive Metrics**: Detailed analysis of thinking patterns and performance
4. **Backward Compatibility**: Can process existing response files
5. **Error Handling**: Graceful handling of parsing errors and edge cases

## Dependencies

Make sure you have the required dependencies:

```bash
pip install numpy tqdm datasets transformers requests sglang
```

## Troubleshooting

### Common Issues

1. **Server Connection**: Make sure SGLang server is running before using the scripts
2. **Memory Issues**: Reduce `batch_size` if you encounter OOM errors
3. **Parsing Errors**: The scripts include fallback methods for robust answer extraction
4. **Resume Issues**: 
   - If resume doesn't work as expected, check the output file format and IDs
   - Use `--force_resume` to start from a specific ID without an existing file
   - Check logs for "Already processed X examples" to verify resume detection

### Logging

All scripts include detailed logging. Check the logs for:
- Processing progress
- Accuracy statistics
- Error messages
- Performance metrics
- Resume status and detection

## Advanced Usage

### Custom Evaluation

You can modify the answer parsing functions in any script to handle domain-specific formats:

```python
def extract_answer_custom(completion):
    # Your custom parsing logic here
    pass
```

### Batch Processing

For large datasets, use appropriate batch sizes:
- Generation: `--batch_size 64` (default)
- Validation: `--batch_size 4` (default, more conservative)

### Model Comparison

Use different `model_name` values in validation to compare multiple models:

```bash
python validate_single_model_sglang.py --model_name baseline_model ...
python validate_single_model_sglang.py --model_name improved_model ...
```

## Resume Functionality Examples

### Scenario 1: Interrupted Long Run

```bash
# Start processing 2000 examples
python generate_responses_gsm8k.py --model Qwen/Qwen3-0.6B --num_samples 2000

# Script gets interrupted at example 750...

# Simply restart - auto-resumes from ID 751
python generate_responses_gsm8k.py --model Qwen/Qwen3-0.6B --num_samples 2000
```

### Scenario 2: Incremental Processing

```bash
# Process first 500 examples
python generate_responses_gsm8k.py --model Qwen/Qwen3-0.6B --num_samples 500

# Later, extend to 1000 examples (processes 501-1000)  
python generate_responses_gsm8k.py --model Qwen/Qwen3-0.6B --num_samples 1000

# Even later, extend to 2000 examples (processes 1001-2000)
python generate_responses_gsm8k.py --model Qwen/Qwen3-0.6B --num_samples 2000
```

### Scenario 3: Selective Reprocessing

```bash
# Reprocess from a specific point (maybe due to server issues)
python generate_responses_gsm8k.py \
    --model Qwen/Qwen3-0.6B \
    --num_samples 1000 \
    --resume_from_id 800
```

**Expected Log Output:**
```
INFO - Checking for existing responses...
INFO - Found 750 existing responses  
INFO - Resuming from ID 751
INFO - Already processed 750 examples
INFO - Found 250 examples to process (after resume filtering)
``` 