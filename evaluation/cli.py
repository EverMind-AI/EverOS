"""
CLI entry point for the evaluation framework.

Usage:
    python -m evaluation.cli --dataset locomo --system evermemos
    python -m evaluation.cli --dataset locomo --system evermemos --smoke 10
    python -m evaluation.cli --dataset locomo --system evermemos --stages search answer evaluate
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Environment initialization - must be done before importing EverMemOS components
# Reference: src/bootstrap.py initialization logic

# Add project paths
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Load environment variables
from common_utils.load_env import setup_environment

setup_environment(load_env_file_name=".env", check_env_var="MONGODB_HOST")

from evaluation.src.core.loaders import load_dataset
from evaluation.src.core.pipeline import Pipeline
from evaluation.src.adapters.registry import create_adapter
from evaluation.src.evaluators.registry import create_evaluator
from evaluation.src.utils.config import load_yaml
from evaluation.src.utils.logger import get_console

from memory_layer.llm.llm_provider import LLMProvider


def deep_merge_config(base: dict, override: dict) -> dict:
    """
    Deep merge configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_config(result[key], value)
        else:
            # Direct override
            result[key] = value
    return result


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Memory System Evaluation Framework")

    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (e.g., locomo)"
    )
    parser.add_argument(
        "--system", type=str, required=True, help="System name (e.g., evermemos)"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Stages to run (add, search, answer, evaluate). Default: all",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable smoke test mode (process small dataset for quick validation)",
    )
    parser.add_argument(
        "--smoke-messages",
        type=int,
        default=10,
        help="Smoke test: number of messages to process (use 0 for all). Default: 10",
    )
    parser.add_argument(
        "--smoke-questions",
        type=int,
        default=3,
        help="Smoke test: number of questions to test (use 0 for all). Default: 3",
    )
    parser.add_argument(
        "--from-conv",
        type=int,
        default=0,
        help="Starting conversation index to process (inclusive, 0-based). Default: 0",
    )
    parser.add_argument(
        "--to-conv",
        type=int,
        default=None,
        help="Ending conversation index to process (exclusive). Default: None (process all remaining)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name/version for distinguishing multiple runs (e.g., 'v1', 'baseline', '20241104')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: results/{dataset}-{system}[-{run_name}]",
    )
    parser.add_argument(
        "--clean-groups",
        action="store_true",
        help="Before Add stage, clear database data for the groups (group_id=conversation_id) involved in this run. "
             "Useful for debugging to avoid polluted data.",
    )

    args = parser.parse_args()

    console = get_console()

    # Load configurations
    console.print("\n[bold cyan]Loading configurations...[/bold cyan]")

    evaluation_root = Path(__file__).parent

    # Load dataset configuration
    dataset_config_path = (
        evaluation_root / "config" / "datasets" / f"{args.dataset}.yaml"
    )
    if not dataset_config_path.exists():
        console.print(f"[red]❌ Dataset config not found: {dataset_config_path}[/red]")
        return

    dataset_config = load_yaml(str(dataset_config_path))
    console.print(f"  ✅ Loaded dataset config: {args.dataset}")

    # Override MEMORY_LANGUAGE from dataset config if specified
    if "memory_language" in dataset_config:
        os.environ["MEMORY_LANGUAGE"] = dataset_config["memory_language"]
        console.print(
            f"  🌐 Memory language: {dataset_config['memory_language']} (from dataset config)"
        )

    # Load system configuration
    system_config_path = evaluation_root / "config" / "systems" / f"{args.system}.yaml"
    if not system_config_path.exists():
        console.print(f"[red]❌ System config not found: {system_config_path}[/red]")
        return

    system_config = load_yaml(str(system_config_path))
    console.print(f"  ✅ Loaded system config: {args.system}")

    # Apply dataset-specific configuration overrides
    if (
        "dataset_overrides" in system_config
        and args.dataset in system_config["dataset_overrides"]
    ):
        overrides = system_config["dataset_overrides"][args.dataset]
        # Deep merge override configurations (supports nested field overrides)
        system_config = deep_merge_config(system_config, overrides)
        console.print(
            f"  🔧 Applied dataset overrides for {args.dataset}: {list(overrides.keys())}"
        )

    # Load dataset
    console.print(f"\n[bold cyan]Loading dataset: {args.dataset}[/bold cyan]")

    data_path = dataset_config["data"]["path"]
    if not Path(data_path).is_absolute():
        # Priority: load from evaluation/data/, fall back to project root
        eval_data_path = evaluation_root / "data" / data_path
        root_data_path = evaluation_root.parent / data_path

        if eval_data_path.exists():
            data_path = eval_data_path
            console.print(f"  📂 Using evaluation/data/{data_path}")
        elif root_data_path.exists():
            data_path = root_data_path
            console.print(f"  📂 Using project root data/{data_path}")
        else:
            console.print(
                f"[red]❌ Data not found in evaluation/data/ or project root data/[/red]"
            )
            return

    # Get max_content_length from dataset config (if specified)
    max_content_length = dataset_config.get("data", {}).get("max_content_length", None)
    if max_content_length:
        console.print(f"  ⚠️  Max content length: {max_content_length} characters")

    # Smart load with auto conversion
    dataset = load_dataset(
        args.dataset, str(data_path), max_content_length=max_content_length
    )

    console.print(
        f"  ✅ Loaded {len(dataset.conversations)} conversations, {len(dataset.qa_pairs)} QA pairs"
    )

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Generate output directory name based on run_name presence
        if args.run_name:
            output_dir = (
                evaluation_root
                / "results"
                / f"{args.dataset}-{args.system}-{args.run_name}"
            )
        else:
            output_dir = evaluation_root / "results" / f"{args.dataset}-{args.system}"

    # Create components
    console.print(f"\n[bold cyan]Initializing components...[/bold cyan]")

    # Add dataset_name to system_config for adapter initialization
    # (Used to determine num_workers based on adapter + dataset combination)
    system_config["dataset_name"] = args.dataset
    # Pass CLI switch down to adapter via config (adapters can opt-in)
    system_config["clean_groups"] = bool(args.clean_groups)

    # Create adapter (pass output_dir for persistence)
    adapter = create_adapter(
        system_config["adapter"], system_config, output_dir=output_dir
    )
    console.print(f"  ✅ Created adapter: {adapter.get_system_info()['name']}")

    # Create evaluator
    evaluator = create_evaluator(
        dataset_config["evaluation"]["type"], dataset_config["evaluation"]
    )
    console.print(f"  ✅ Created evaluator: {evaluator.get_name()}")

    # Create LLM Provider for answer generation
    llm_config = system_config.get("llm", {})
    llm_provider = LLMProvider(
        provider_type=llm_config.get("provider", "openai"),
        model=llm_config.get("model"),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
        temperature=llm_config.get("temperature", 0.0),
        max_tokens=llm_config.get("max_tokens", 32768),
    )
    console.print(f"  Created LLM provider: {llm_config.get('model')}")

    # Create pipeline
    # Read filter categories from dataset configuration
    filter_categories = dataset_config.get("evaluation", {}).get("filter_category", [])

    pipeline = Pipeline(
        adapter=adapter,
        evaluator=evaluator,
        llm_provider=llm_provider,
        output_dir=output_dir,
        filter_categories=filter_categories,
    )

    console.print(f"  ✅ Created pipeline, output: {output_dir}")
    if filter_categories:
        console.print(f"  📋 Filter categories: {filter_categories}")

    # Run pipeline
    try:
        results = await pipeline.run(
            dataset=dataset,
            stages=args.stages,
            smoke_test=args.smoke,
            smoke_messages=args.smoke_messages,
            smoke_questions=args.smoke_questions,
            from_conv=args.from_conv,
            to_conv=args.to_conv,
        )

        console.print(f"\n[bold green]✨ Evaluation completed![/bold green]")
        console.print(f"Results saved to: [cyan]{output_dir}[/cyan]\n")

    finally:
        # Cleanup resources
        # Clean up adapter session (e.g., aiohttp.ClientSession)
        if hasattr(adapter, 'close') and callable(getattr(adapter, 'close')):
            try:
                await adapter.close()
                console.print("[dim]🧹 Cleaned up adapter resources[/dim]")
            except Exception as e:
                # Cleanup failure doesn't affect main process
                console.print(f"[dim]⚠️  Failed to cleanup adapter resources: {e}[/dim]")

        # Only systems using rerank need cleanup
        systems_need_rerank = ["evermemos"]
        if args.system in systems_need_rerank:
            try:
                from agentic_layer import rerank_service

                reranker = rerank_service.get_rerank_service()
                if hasattr(reranker, 'close') and callable(getattr(reranker, 'close')):
                    await reranker.close()
                    console.print("[dim]🧹 Cleaned up rerank service resources[/dim]")
            except Exception as e:
                # Cleanup failure doesn't affect main process
                console.print(f"[dim]⚠️  Failed to cleanup rerank resources: {e}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
