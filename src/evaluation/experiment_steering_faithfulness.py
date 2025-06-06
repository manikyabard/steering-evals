#!/usr/bin/env python3
"""
Comprehensive Experiment: Steering Reasoning Length vs. Faithfulness

This script implements the research proposal to investigate how steering reasoning
length affects the faithfulness of reasoning in language models.

It combines:
1. ThinkEdit-style reasoning length steering
2. ChainScope's restoration errors evaluation
3. Comprehensive analysis of faithfulness metrics

Usage:
    python experiment_steering_faithfulness.py --model Qwen/Qwen3-0.6B --experiment-type full
"""

import argparse
import logging
import json
import os
import asyncio
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from chainscope.typing import *
from chainscope.cot_paths_eval import evaluate_cot_paths
from chainscope.api_utils.api_selector import APIPreferences

# Import our custom generation script
import sys

sys.path.append(".")
from gen_steered_cot_paths import main as generate_steered_paths


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("steering_faithfulness_experiment.log"),
        ],
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment: Steering vs Faithfulness")

    # Model and dataset configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model to experiment with",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Problem dataset (gsm8k, math, mmlu)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=100, help="Number of problems to test"
    )
    parser.add_argument(
        "--n-paths", type=int, default=5, help="Number of CoT paths per problem"
    )

    # Steering configuration
    parser.add_argument(
        "--alpha-values",
        nargs="+",
        type=float,
        default=[-0.1, -0.05, 0.0, 0.05, 0.1, 0.15],
        help="Steering strengths to test",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["attn", "mlp"],
        default="attn",
        help="Component to steer",
    )
    parser.add_argument(
        "--test-thinkedit",
        action="store_true",
        help="Also test ThinkEdit weight editing approach",
    )

    # Experiment configuration
    parser.add_argument(
        "--experiment-type",
        type=str,
        choices=["pilot", "full", "analysis-only"],
        default="pilot",
        help="Experiment scope",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="steering_faithfulness_results",
        help="Output directory",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="anthropic/claude-3.5-sonnet",
        help="Model for faithfulness evaluation",
    )

    # API configuration
    parser.add_argument("--anthropic", action="store_true", help="Use Anthropic API")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI API")

    return parser.parse_args()


class SteeringFaithfulnessExperiment:
    """Main experiment class for steering vs faithfulness analysis."""

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Experiment configuration
        self.model_short_name = args.model.split("/")[-1]
        self.alpha_values = args.alpha_values
        self.dataset = args.dataset

        # Results storage
        self.results = {}
        self.faithfulness_stats = {}

    async def run_experiment(self):
        """Run the complete steering vs faithfulness experiment."""
        self.logger.info(f"Starting steering vs faithfulness experiment")
        self.logger.info(f"Model: {self.args.model}")
        self.logger.info(f"Dataset: {self.dataset}")
        self.logger.info(f"Alpha values: {self.alpha_values}")

        if self.args.experiment_type != "analysis-only":
            # Phase 1: Generate steered CoT paths
            await self.generate_steered_cot_paths()

            # Phase 2: Evaluate faithfulness
            await self.evaluate_faithfulness()

        # Phase 3: Analyze results
        self.analyze_results()

        # Phase 4: Create visualizations
        self.create_visualizations()

        # Phase 5: Generate report
        self.generate_report()

    async def generate_steered_cot_paths(self):
        """Generate CoT paths for each steering strength."""
        self.logger.info("Phase 1: Generating steered CoT paths")

        for alpha in self.alpha_values:
            self.logger.info(f"Generating paths for α = {alpha}")

            # Check if paths already exist
            steered_model_id = (
                f"{self.args.model}_steered_alpha_{alpha}_{self.args.component}"
            )
            cot_paths_dir = Path("chainscope/chainscope/data/cot_paths") / self.dataset
            response_path = (
                cot_paths_dir / f"{steered_model_id.replace('/', '__')}.yaml"
            )

            if response_path.exists():
                self.logger.info(f"Paths for α = {alpha} already exist, skipping")
                continue

            # Generate using our custom script
            generation_args = [
                "--n-paths",
                str(self.args.n_paths),
                "--problem-dataset-name",
                self.dataset,
                "--model-id",
                self.args.model,
                "--steering-mode",
                "runtime",
                "--alpha",
                str(alpha),
                "--component",
                self.args.component,
                "--verbose",
            ]

            # This would need to be refactored to call the function directly
            # For now, we'll create a simplified version
            await self.generate_paths_for_alpha(alpha)

    async def generate_paths_for_alpha(self, alpha: float):
        """Generate CoT paths for a specific alpha value."""
        # This is a simplified version - in practice, you'd integrate with the
        # actual generation code from gen_steered_cot_paths.py

        try:
            from gen_steered_cot_paths import gen_cot_paths_with_runtime_steering

            # Load directions
            directions_file = f"directions/{self.model_short_name}_thinking_length_direction_gsm8k_{self.args.component}.pt"
            if not os.path.exists(directions_file):
                self.logger.error(f"Directions file not found: {directions_file}")
                return

            directions = torch.load(directions_file)

            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=5000,
            )

            api_preferences = APIPreferences.from_args(
                anthropic=self.args.anthropic,
                open_ai=self.args.openai,
            )

            # Generate paths
            cot_paths = await gen_cot_paths_with_runtime_steering(
                model_id=self.args.model,
                problem_dataset_name=self.dataset,
                sampling_params=sampling_params,
                n_paths=self.args.n_paths,
                existing_paths=None,
                api_preferences=api_preferences,
                directions=directions,
                alpha=alpha,
                component=self.args.component,
            )

            # Save paths
            cot_paths.save()
            self.logger.info(f"Saved CoT paths for α = {alpha}")

        except Exception as e:
            self.logger.error(f"Error generating paths for α = {alpha}: {e}")

    async def evaluate_faithfulness(self):
        """Evaluate faithfulness for all generated CoT paths."""
        self.logger.info("Phase 2: Evaluating faithfulness")

        api_preferences = APIPreferences.from_args(
            anthropic=self.args.anthropic,
            open_ai=self.args.openai,
        )

        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic for evaluation
            top_p=0.9,
            max_new_tokens=15000,
        )

        for alpha in self.alpha_values:
            self.logger.info(f"Evaluating faithfulness for α = {alpha}")

            # Load CoT paths
            steered_model_id = (
                f"{self.args.model}_steered_alpha_{alpha}_{self.args.component}"
            )
            cot_paths_dir = Path("chainscope/chainscope/data/cot_paths") / self.dataset
            response_path = (
                cot_paths_dir / f"{steered_model_id.replace('/', '__')}.yaml"
            )

            if not response_path.exists():
                self.logger.warning(f"No CoT paths found for α = {alpha}")
                continue

            try:
                cot_paths = CoTPath.load_from_path(response_path)

                # Evaluate faithfulness
                cot_paths_eval = evaluate_cot_paths(
                    cot_paths=cot_paths,
                    evaluator_model_id=self.args.evaluator_model,
                    sampling_params=sampling_params,
                    api_preferences=api_preferences,
                    max_retries=3,
                )

                # Store results
                self.results[alpha] = {
                    "cot_paths": cot_paths,
                    "cot_paths_eval": cot_paths_eval,
                }

                self.logger.info(f"Completed evaluation for α = {alpha}")

            except Exception as e:
                self.logger.error(f"Error evaluating α = {alpha}: {e}")

    def analyze_results(self):
        """Analyze the relationship between steering and faithfulness."""
        self.logger.info("Phase 3: Analyzing results")

        analysis_results = []

        for alpha, result_data in self.results.items():
            if "cot_paths_eval" not in result_data:
                continue

            cot_paths = result_data["cot_paths"]
            cot_paths_eval = result_data["cot_paths_eval"]

            # Calculate metrics for this alpha
            metrics = self.calculate_faithfulness_metrics(
                alpha, cot_paths, cot_paths_eval
            )
            analysis_results.append(metrics)

            self.logger.info(f"α = {alpha}: {metrics['summary']}")

        # Convert to DataFrame for analysis
        self.analysis_df = pd.DataFrame(analysis_results)

        # Save detailed analysis
        analysis_file = self.output_dir / "detailed_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis_results, f, indent=2)

        # Save DataFrame
        csv_file = self.output_dir / "analysis_summary.csv"
        self.analysis_df.to_csv(csv_file, index=False)

        self.logger.info(f"Analysis saved to {analysis_file} and {csv_file}")

    def calculate_faithfulness_metrics(
        self, alpha: float, cot_paths: CoTPath, cot_paths_eval: CoTPathEval
    ) -> Dict:
        """Calculate comprehensive faithfulness metrics for a given alpha."""

        total_paths = 0
        correct_answers = 0
        total_steps = 0
        unfaithful_steps = 0
        reasoning_lengths = []

        unfaithful_by_severity = {"TRIVIAL": 0, "MINOR": 0, "MAJOR": 0, "CRITICAL": 0}
        unfaithful_by_type = {"unused": 0, "unfaithful": 0, "incorrect": 0}

        for qid in cot_paths.cot_path_by_qid:
            for response_uuid in cot_paths.cot_path_by_qid[qid]:
                total_paths += 1

                # Check answer correctness
                if (
                    qid in cot_paths_eval.answer_correct_by_qid
                    and response_uuid in cot_paths_eval.answer_correct_by_qid[qid]
                ):
                    answer_result = cot_paths_eval.answer_correct_by_qid[qid][
                        response_uuid
                    ]
                    if answer_result.answer_status == "CORRECT":
                        correct_answers += 1

                # Analyze reasoning steps
                steps = cot_paths.cot_path_by_qid[qid][response_uuid]
                reasoning_length = sum(len(step.split()) for step in steps.values())
                reasoning_lengths.append(reasoning_length)
                total_steps += len(steps)

                # Check for unfaithful steps
                if (
                    qid in cot_paths_eval.second_pass_eval_by_qid
                    and response_uuid in cot_paths_eval.second_pass_eval_by_qid[qid]
                ):
                    second_pass = cot_paths_eval.second_pass_eval_by_qid[qid][
                        response_uuid
                    ]

                    for step_status in second_pass.steps_status.values():
                        if step_status.node_status in ["UNFAITHFUL", "INCORRECT"]:
                            unfaithful_steps += 1
                            unfaithful_by_severity[step_status.node_severity] += 1
                            unfaithful_by_type[step_status.node_status.lower()] += 1

        # Calculate metrics
        accuracy = correct_answers / total_paths if total_paths > 0 else 0
        unfaithfulness_rate = unfaithful_steps / total_steps if total_steps > 0 else 0
        avg_reasoning_length = np.mean(reasoning_lengths) if reasoning_lengths else 0

        return {
            "alpha": alpha,
            "total_paths": total_paths,
            "total_steps": total_steps,
            "accuracy": accuracy,
            "unfaithfulness_rate": unfaithfulness_rate,
            "avg_reasoning_length": avg_reasoning_length,
            "unfaithful_by_severity": unfaithful_by_severity,
            "unfaithful_by_type": unfaithful_by_type,
            "summary": f"Accuracy: {accuracy:.3f}, Unfaithfulness: {unfaithfulness_rate:.3f}, Avg Length: {avg_reasoning_length:.1f}",
        }

    def create_visualizations(self):
        """Create comprehensive visualizations of the results."""
        self.logger.info("Phase 4: Creating visualizations")

        if hasattr(self, "analysis_df") and len(self.analysis_df) > 0:
            self.create_main_plot()
            self.create_detailed_plots()
        else:
            self.logger.warning("No analysis data available for visualization")

    def create_main_plot(self):
        """Create the main steering vs faithfulness plot."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        alphas = self.analysis_df["alpha"]

        # Plot 1: Accuracy vs Alpha
        ax1.plot(alphas, self.analysis_df["accuracy"], "bo-", linewidth=2, markersize=8)
        ax1.set_xlabel("Steering Strength (α)", fontsize=12)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.set_title("Accuracy vs Steering Strength", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Unfaithfulness Rate vs Alpha
        ax2.plot(
            alphas,
            self.analysis_df["unfaithfulness_rate"],
            "ro-",
            linewidth=2,
            markersize=8,
        )
        ax2.set_xlabel("Steering Strength (α)", fontsize=12)
        ax2.set_ylabel("Unfaithfulness Rate", fontsize=12)
        ax2.set_title("Unfaithfulness vs Steering Strength", fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Reasoning Length vs Alpha
        ax3.plot(
            alphas,
            self.analysis_df["avg_reasoning_length"],
            "go-",
            linewidth=2,
            markersize=8,
        )
        ax3.set_xlabel("Steering Strength (α)", fontsize=12)
        ax3.set_ylabel("Average Reasoning Length (words)", fontsize=12)
        ax3.set_title("Reasoning Length vs Steering Strength", fontsize=14)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_file = self.output_dir / "steering_vs_faithfulness_main.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Main plot saved to {plot_file}")

    def create_detailed_plots(self):
        """Create detailed analysis plots."""
        # Correlation plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Create correlation matrix
        numeric_cols = [
            "alpha",
            "accuracy",
            "unfaithfulness_rate",
            "avg_reasoning_length",
        ]
        corr_matrix = self.analysis_df[numeric_cols].corr()

        sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Correlation Matrix: Steering vs Metrics", fontsize=14)

        corr_file = self.output_dir / "correlation_matrix.png"
        plt.savefig(corr_file, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Correlation plot saved to {corr_file}")

    def generate_report(self):
        """Generate a comprehensive experimental report."""
        self.logger.info("Phase 5: Generating report")

        report = {
            "experiment_metadata": {
                "model": self.args.model,
                "dataset": self.dataset,
                "alpha_values": self.alpha_values,
                "n_paths": self.args.n_paths,
                "component": self.args.component,
                "evaluator_model": self.args.evaluator_model,
            },
            "key_findings": self.extract_key_findings(),
            "detailed_results": self.results if hasattr(self, "results") else {},
            "analysis_summary": (
                self.analysis_df.to_dict("records")
                if hasattr(self, "analysis_df")
                else []
            ),
        }

        report_file = self.output_dir / "experiment_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate markdown report
        self.generate_markdown_report()

        self.logger.info(f"Report saved to {report_file}")

    def extract_key_findings(self):
        """Extract key findings from the analysis."""
        if not hasattr(self, "analysis_df") or len(self.analysis_df) == 0:
            return {"status": "No analysis data available"}

        df = self.analysis_df

        # Find correlations
        accuracy_alpha_corr = df["accuracy"].corr(df["alpha"])
        unfaithfulness_alpha_corr = df["unfaithfulness_rate"].corr(df["alpha"])
        length_alpha_corr = df["avg_reasoning_length"].corr(df["alpha"])

        # Find optimal alpha
        best_accuracy_idx = df["accuracy"].idxmax()
        best_accuracy_alpha = df.loc[best_accuracy_idx, "alpha"]

        lowest_unfaithfulness_idx = df["unfaithfulness_rate"].idxmin()
        lowest_unfaithfulness_alpha = df.loc[lowest_unfaithfulness_idx, "alpha"]

        return {
            "correlations": {
                "accuracy_vs_alpha": accuracy_alpha_corr,
                "unfaithfulness_vs_alpha": unfaithfulness_alpha_corr,
                "reasoning_length_vs_alpha": length_alpha_corr,
            },
            "optimal_values": {
                "best_accuracy_alpha": best_accuracy_alpha,
                "lowest_unfaithfulness_alpha": lowest_unfaithfulness_alpha,
            },
            "summary": f"Accuracy-Alpha correlation: {accuracy_alpha_corr:.3f}, "
            f"Unfaithfulness-Alpha correlation: {unfaithfulness_alpha_corr:.3f}",
        }

    def generate_markdown_report(self):
        """Generate a human-readable markdown report."""

        md_content = f"""# Steering vs Faithfulness Experiment Report

## Experiment Configuration

- **Model**: {self.args.model}
- **Dataset**: {self.dataset}
- **Alpha Values**: {self.alpha_values}
- **Number of Paths**: {self.args.n_paths}
- **Component**: {self.args.component}
- **Evaluator Model**: {self.args.evaluator_model}

## Key Findings

"""

        if hasattr(self, "analysis_df") and len(self.analysis_df) > 0:
            findings = self.extract_key_findings()

            md_content += f"""
### Correlations
- **Accuracy vs Steering Strength**: {findings['correlations']['accuracy_vs_alpha']:.3f}
- **Unfaithfulness vs Steering Strength**: {findings['correlations']['unfaithfulness_vs_alpha']:.3f}
- **Reasoning Length vs Steering Strength**: {findings['correlations']['reasoning_length_vs_alpha']:.3f}

### Optimal Settings
- **Best Accuracy**: α = {findings['optimal_values']['best_accuracy_alpha']}
- **Lowest Unfaithfulness**: α = {findings['optimal_values']['lowest_unfaithfulness_alpha']}

## Results Summary

| Alpha | Accuracy | Unfaithfulness Rate | Avg Reasoning Length |
|-------|----------|-------------------|---------------------|
"""

            for _, row in self.analysis_df.iterrows():
                md_content += f"| {row['alpha']:.2f} | {row['accuracy']:.3f} | {row['unfaithfulness_rate']:.3f} | {row['avg_reasoning_length']:.1f} |\n"

        else:
            md_content += "No analysis results available.\n"

        md_content += f"""

## Methodology

This experiment investigates the relationship between reasoning length steering and faithfulness using:

1. **Steering Techniques**: Runtime application of direction vectors to modify reasoning length
2. **Faithfulness Evaluation**: ChainScope's restoration errors detection pipeline
3. **Multi-pass Evaluation**: Systematic detection of computational mistakes and silent corrections

## Files Generated

- `detailed_analysis.json`: Complete numerical results
- `analysis_summary.csv`: Summary metrics in CSV format
- `steering_vs_faithfulness_main.png`: Main visualization
- `correlation_matrix.png`: Correlation analysis
- `experiment_report.json`: Machine-readable report

"""

        md_file = self.output_dir / "README.md"
        with open(md_file, "w") as f:
            f.write(md_content)

        self.logger.info(f"Markdown report saved to {md_file}")


async def main():
    """Main function to run the steering vs faithfulness experiment."""
    args = parse_args()
    logger = setup_logging()

    # Create and run experiment
    experiment = SteeringFaithfulnessExperiment(args, logger)
    await experiment.run_experiment()

    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
