#!/usr/bin/env python3
"""
Generate CoT paths with steering applied for faithfulness evaluation.

This script combines ChainScope's restoration errors evaluation pipeline
with steering techniques from steer_reasoning_length.py and ThinkEdit models.

Usage:
    # Using runtime steering vectors
    python gen_steered_cot_paths.py -n 10 -d gsm8k -m Qwen/Qwen3-0.6B --steering-mode runtime --alpha 0.08

    # Using ThinkEdit model
    python gen_steered_cot_paths.py -n 10 -d gsm8k -m ./thinkedit_models/ThinkEdit-Qwen3-0.6B --steering-mode edited
"""

import logging
import os
import torch
import click
from pathlib import Path

from chainscope.api_utils.api_selector import APIPreferences
from chainscope.cot_paths import gen_cot_paths_async
from chainscope.typing import *
from chainscope.utils import MODELS_MAP
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import steering utilities
import sys

sys.path.append(".")  # Add current directory to path
from steer_reasoning_length import apply_steering_layers, remove_steering_layers


@click.command()
@click.option(
    "-n", "--n-paths", type=int, required=True, help="Number of CoT paths to generate"
)
@click.option(
    "-d",
    "--problem-dataset-name",
    type=str,
    required=True,
    help="Dataset name (e.g., gsm8k, math)",
)
@click.option("-m", "--model-id", type=str, required=True, help="Model name or path")
@click.option("-t", "--temperature", type=float, default=0.7)
@click.option("-p", "--top-p", type=float, default=0.9)
@click.option("--max-new-tokens", type=int, default=5_000)
@click.option(
    "--steering-mode",
    type=click.Choice(["runtime", "edited", "none"]),
    default="none",
    help="Steering approach: runtime (direction vectors), edited (ThinkEdit model), or none",
)
@click.option(
    "--alpha", type=float, default=0.0, help="Steering strength (for runtime mode)"
)
@click.option(
    "--directions-file",
    type=str,
    default=None,
    help="Path to steering directions file (for runtime mode)",
)
@click.option(
    "--component",
    type=str,
    choices=["attn", "mlp", "both"],
    default="attn",
    help="Which component to steer (for runtime mode)",
)
@click.option(
    "--open-router",
    "--or",
    is_flag=True,
    help="Use OpenRouter API instead of local models",
)
@click.option(
    "--open-ai", "--oa", is_flag=True, help="Use OpenAI API instead of local models"
)
@click.option(
    "--anthropic",
    "--an",
    is_flag=True,
    help="Use Anthropic API instead of local models",
)
@click.option(
    "--deepseek", "--ds", is_flag=True, help="Use DeepSeek API instead of local models"
)
@click.option(
    "--append", is_flag=True, help="Append to existing data instead of starting fresh"
)
@click.option("-v", "--verbose", is_flag=True)
def main(
    n_paths: int,
    problem_dataset_name: str,
    model_id: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    steering_mode: str,
    alpha: float,
    directions_file: str,
    component: str,
    open_router: bool,
    open_ai: bool,
    anthropic: bool,
    deepseek: bool,
    append: bool,
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Setup model and steering
    original_model_id = model_id
    model_id = MODELS_MAP.get(model_id, model_id)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    api_preferences = APIPreferences.from_args(
        open_router=open_router,
        open_ai=open_ai,
        anthropic=anthropic,
        deepseek=deepseek,
    )

    # Handle different steering modes
    steered_model_id = model_id
    steering_info = {}

    if steering_mode == "runtime":
        # Load steering directions
        if directions_file is None:
            model_short_name = original_model_id.split("/")[-1]
            directions_file = f"directions/{model_short_name}_thinking_length_direction_gsm8k_{component}.pt"

        if not os.path.exists(directions_file):
            raise FileNotFoundError(f"Directions file not found: {directions_file}")

        directions = torch.load(directions_file)
        logging.info(f"Loaded steering directions from {directions_file}")

        steering_info = {
            "mode": "runtime",
            "alpha": alpha,
            "directions_file": directions_file,
            "component": component,
        }

        # For runtime steering, we'll modify the model_id to include steering info
        steered_model_id = f"{model_id}_steered_alpha_{alpha}_{component}"

    elif steering_mode == "edited":
        # Using ThinkEdit model - model_id should point to edited model directory
        if not os.path.exists(model_id):
            raise FileNotFoundError(f"ThinkEdit model not found: {model_id}")

        # Load metadata if available
        metadata_file = os.path.join(model_id, "thinkedit_metadata.json")
        if os.path.exists(metadata_file):
            import json

            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            steering_info = {
                "mode": "edited",
                "base_model": metadata.get("base_model"),
                "intervention_weight": metadata.get("intervention_weight"),
                "edited_heads": metadata.get("edited_heads"),
            }
        else:
            steering_info = {"mode": "edited", "model_path": model_id}

        steered_model_id = f"{model_id.replace('/', '__')}_thinkedit"

    else:  # steering_mode == 'none'
        steering_info = {"mode": "none"}

    # Try to load existing paths if append is True
    existing_paths = None
    if append:
        try:
            cot_paths_dir = DATA_DIR / "cot_paths" / problem_dataset_name
            response_path = (
                cot_paths_dir / f"{steered_model_id.replace('/', '__')}.yaml"
            )
            if response_path.exists():
                existing_paths = CoTPath.load_from_path(response_path)
                logging.info(f"Loaded existing paths from {response_path}")
            else:
                logging.warning(
                    f"No existing paths found at {response_path}, starting fresh"
                )
        except Exception as e:
            logging.warning(f"Error loading existing paths: {e}, starting fresh")

    # Create a custom generation function for steering
    async def generate_steered_cot_paths():
        if steering_mode == "runtime":
            # For runtime steering, we need to modify the generation process
            return await gen_cot_paths_with_runtime_steering(
                model_id=original_model_id,
                problem_dataset_name=problem_dataset_name,
                sampling_params=sampling_params,
                n_paths=n_paths,
                existing_paths=existing_paths,
                api_preferences=api_preferences,
                directions=directions,
                alpha=alpha,
                component=component,
            )
        else:
            # For edited models or no steering, use standard generation
            return await gen_cot_paths_async(
                model_id=model_id,
                problem_dataset_name=problem_dataset_name,
                sampling_params=sampling_params,
                n_paths=n_paths,
                existing_paths=existing_paths,
                api_preferences=api_preferences,
            )

    # Generate CoT paths
    import asyncio

    cot_paths = asyncio.run(generate_steered_cot_paths())

    # Update the model_id to reflect steering
    cot_paths.model_id = steered_model_id

    # Add steering metadata
    if not hasattr(cot_paths, "metadata"):
        cot_paths.metadata = {}
    cot_paths.metadata["steering_info"] = steering_info

    # Save the results
    cot_paths.save()
    logging.info(f"Saved steered CoT paths for {steered_model_id}")


async def gen_cot_paths_with_runtime_steering(
    model_id: str,
    problem_dataset_name: str,
    sampling_params: SamplingParams,
    n_paths: int,
    existing_paths: CoTPath | None,
    api_preferences: APIPreferences,
    directions: torch.Tensor,
    alpha: float,
    component: str,
):
    """Generate CoT paths with runtime steering applied."""

    # Load model locally for steering
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map=device
    ).eval()

    directions = directions.to(device)

    # Load the problem dataset
    problem_dataset = ProblemDataset.load(problem_dataset_name)

    # Prepare for generation with steering
    cot_instruction = """Write down your answer step by step, and number each step ("1.", "2.", etc.)."""

    cot_path_by_qid = {}

    # Initialize with existing paths if any
    if existing_paths:
        cot_path_by_qid = existing_paths.cot_path_by_qid.copy()

    for qid, problem in problem_dataset.problems_by_qid.items():
        # Skip if we already have enough paths for this problem
        if existing_paths and qid in existing_paths.cot_path_by_qid:
            existing_count = len(existing_paths.cot_path_by_qid[qid])
            if existing_count >= n_paths:
                continue
            paths_needed = n_paths - existing_count
        else:
            paths_needed = n_paths

        if qid not in cot_path_by_qid:
            cot_path_by_qid[qid] = {}

        for path_idx in range(paths_needed):
            prompt = f"{problem.q_str}\n\n{cot_instruction}"

            # Apply steering and generate
            try:
                response = await generate_single_with_steering(
                    model,
                    tokenizer,
                    problem.q_str,
                    alpha,
                    directions,
                    component,
                    sampling_params.max_new_tokens,
                    sampling_params.temperature,
                )

                # Process response into steps (reuse from chainscope/chainscope/cot_paths.py)
                from chainscope.cot_paths import process_response

                steps_list = process_response(
                    response.get("thinking", "") + " " + response.get("response", "")
                )

                response_uuid = str(uuid.uuid4())
                cot_path_by_qid[qid][response_uuid] = {}
                for step_number, step in enumerate(steps_list):
                    cot_path_by_qid[qid][response_uuid][step_number] = step

            except Exception as e:
                logging.error(f"Error generating path for qid={qid}: {e}")
                continue

    return CoTPath(
        cot_path_by_qid=cot_path_by_qid,
        model_id=f"{model_id}_steered_alpha_{alpha}_{component}",
        problem_dataset_name=problem_dataset_name,
        sampling_params=sampling_params,
    )


async def generate_single_with_steering(
    model,
    tokenizer,
    question,
    alpha,
    directions,
    component,
    max_new_tokens,
    temperature,
):
    """Generate a single response with steering applied."""
    prompt = f"Solve this math problem step by step:\n{question}"

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    hooks = apply_steering_layers(model, directions, alpha, component)

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()

        # Parse thinking content
        try:
            think_end_token = tokenizer.encode("</think>", add_special_tokens=False)[-1]
            think_end_index = (
                output_ids.index(think_end_token)
                if think_end_token in output_ids
                else -1
            )

            if think_end_index != -1:
                thinking_content = tokenizer.decode(
                    output_ids[:think_end_index], skip_special_tokens=True
                ).strip()
                if thinking_content.startswith("<think>"):
                    thinking_content = thinking_content[len("<think>") :].strip()

                content = tokenizer.decode(
                    output_ids[think_end_index + 1 :], skip_special_tokens=True
                ).strip()
                return {"thinking": thinking_content, "response": content}
        except ValueError:
            pass

        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return {"thinking": "", "response": content}

    finally:
        remove_steering_layers(hooks)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
