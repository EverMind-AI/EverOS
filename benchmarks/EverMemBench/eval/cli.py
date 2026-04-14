"""
CLI entry point for multi-person group chat evaluation.

Supported systems: memos, mem0, memobase, evermemos, zep, llm

Stages:
    add      - Ingest conversation data into memory system
    search   - Retrieve memories for QA questions
    answer   - Generate answers using LLM
    evaluate - Assess answer quality

Usage:
    # Smoke test - add first day only
    python -m eval.cli --dataset dataset/004/dialogue.json --system memos --smoke --smoke-days 1

    # Add all days
    python -m eval.cli --dataset dataset/004/dialogue.json --system memos --stages add

    # Full pipeline: search -> answer -> evaluate
    python -m eval.cli --dataset dataset/004/dialogue.json --qa dataset/004/qa_004.json \
        --system mem0 --user-id 004 --stages search answer evaluate

    # Test all systems
    for sys in memos mem0 memobase zep evermemos; do
        python -m eval.cli --dataset dataset/004/dialogue.json --system $sys --smoke
    done
"""
import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from eval.src.core.loaders import load_groupchat_dataset
from eval.src.core.pipeline import Pipeline
from eval.src.utils.config import load_yaml, get_config_path
from eval.src.utils.logger import get_console, print_error


# Supported systems and their required environment variables
SUPPORTED_SYSTEMS = {
    "memos": ["MEMOS_API_KEY"],
    "mem0": ["MEM0_API_KEY"],
    "memobase": ["MEMOBASE_BASE_URL", "MEMOBASE_API_TOKEN"],
    "evermemos": ["EVERMEMOS_BASE_URL"],  # API key optional for local deployment
    "zep": ["ZEP_API_KEY"],
    "llm": [],  # LLM system uses LLM_API_KEY (validated separately for answer/evaluate)
}


def validate_env_vars(system_name: str) -> bool:
    """
    Validate required environment variables for a system.
    
    Args:
        system_name: System name
        
    Returns:
        True if all required env vars are set
    """
    console = get_console()
    required_vars = SUPPORTED_SYSTEMS.get(system_name, [])
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        console.print(f"\n❌ Missing environment variables for {system_name}:", style="bold red")
        for var in missing_vars:
            console.print(f"   - {var}", style="red")
        console.print("\nPlease set these in your .env file or environment.", style="dim")
        return False
    
    return True


def create_adapter(system_name: str, output_dir: Path, base_url: str = None):
    """
    Create adapter for specified system.

    Args:
        system_name: System name (memos, mem0, memobase, evermemos, zep, llm)
        output_dir: Output directory
        base_url: Optional base URL override for memory system

    Returns:
        Adapter instance
    """
    # If base_url is provided via CLI, set it as environment variable to override
    # This allows CLI arguments to satisfy env var requirements
    if base_url:
        if system_name == "evermemos":
            os.environ["EVERMEMOS_BASE_URL"] = base_url
        elif system_name == "memobase":
            os.environ["MEMOBASE_BASE_URL"] = base_url
        elif system_name == "memos":
            os.environ["MEMOS_BASE_URL"] = base_url

    # Validate environment variables first
    if not validate_env_vars(system_name):
        raise ValueError(f"Missing required environment variables for {system_name}")

    # LLM system has no system-specific yaml; all config comes from pipeline.yaml
    if system_name == "llm":
        from eval.src.adapters.llm_adapter import LLMAdapter
        return LLMAdapter({"name": "llm"}, output_dir)

    # Load system config for memory system adapters
    config_path = get_config_path(f"{system_name}.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_yaml(str(config_path))

    # Apply base_url override if provided
    # Adapters use different config keys: api_url (memos), project_url (memobase), base_url (evermemos)
    if base_url:
        config["base_url"] = base_url
        if system_name == "memos":
            config["api_url"] = base_url
        elif system_name == "memobase":
            config["project_url"] = base_url

    # Create adapter based on system
    if system_name == "memos":
        from eval.src.adapters.memos_adapter import MemosAdapter
        return MemosAdapter(config, output_dir)
    elif system_name == "mem0":
        from eval.src.adapters.mem0_adapter import Mem0Adapter
        return Mem0Adapter(config, output_dir)
    elif system_name == "memobase":
        from eval.src.adapters.memobase_adapter import MemobaseAdapter
        return MemobaseAdapter(config, output_dir)
    elif system_name == "evermemos":
        from eval.src.adapters.evermemos_adapter import EverMemosAdapter
        return EverMemosAdapter(config, output_dir)
    elif system_name == "zep":
        from eval.src.adapters.zep_adapter import ZepAdapter
        return ZepAdapter(config, output_dir)
    else:
        supported = ", ".join(SUPPORTED_SYSTEMS.keys())
        raise ValueError(f"Unknown system: {system_name}. Supported: {supported}")


def parse_args():
    """Parse command line arguments."""
    supported = list(SUPPORTED_SYSTEMS.keys())
    
    parser = argparse.ArgumentParser(
        description="Multi-Person Group Chat Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported Systems:
    memos       - Memos memory system (env: MEMOS_API_KEY, MEMOS_BASE_URL)
    mem0        - Mem0 memory system (env: MEM0_API_KEY)
    memobase    - Memobase memory system (env: MEMOBASE_PROJECT_URL, MEMOBASE_API_KEY)
    evermemos   - EverMemOS memory system (env: EVERMEMOS_BASE_URL, EVERMEMOS_API_KEY)
    zep         - Zep Graph memory system (env: ZEP_API_KEY)
    llm         - Direct LLM evaluation using full dialogue (env: LLM_API_KEY)

Examples:
    # Smoke test with first day
    python -m eval.cli --dataset dataset/004/dialogue.json --system memos --smoke

    # Add all days
    python -m eval.cli --dataset dataset/004/dialogue.json --system mem0 --stages add

    # Custom user ID
    python -m eval.cli --dataset dataset/004/dialogue.json --system zep --user-id my_test_user

    # Full evaluation pipeline
    python -m eval.cli --dataset dataset/004/dialogue.json --qa dataset/004/qa_004.json \\
        --system mem0 --user-id 004 --stages search answer evaluate --top-k 10
    
    # EverMemOS local deployment (specify port via --base-url)
    python -m eval.cli --dataset dataset/004/dialogue.json --system evermemos \\
        --user-id 004 --base-url http://0.0.0.0:19004 --stages add
    
    # Test all systems
    for sys in memos mem0 memobase zep evermemos; do
        python -m eval.cli --dataset dataset/004/dialogue.json --system $sys --smoke
    done
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset JSON file (e.g., dataset/004/dialogue.json)"
    )
    
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        choices=supported,
        help="Memory system to use"
    )
    
    # Optional arguments
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=["add"],
        choices=["add", "search", "answer", "evaluate"],
        help="Stages to run: add, search, answer, evaluate (default: add)"
    )
    
    parser.add_argument(
        "--qa",
        type=str,
        default=None,
        help="Path to QA JSON file (required for search/answer/evaluate stages)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of memories to retrieve for search (default: from system config)"
    )
    
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="User ID for memory system (default: auto-generated)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/results",
        help="Output directory (default: eval/results)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Resume add from this date (inclusive), format YYYY-MM-DD (e.g., 2025-05-22). "
             "Currently implemented for memobase; other systems may ignore it."
    )
    
    # Smoke test options
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable smoke test mode"
    )
    
    parser.add_argument(
        "--smoke-days",
        type=int,
        default=1,
        help="Number of days for smoke test (default: 1)"
    )

    parser.add_argument(
        "--smoke-date",
        type=str,
        default=None,
        help="Run smoke test for a specific date (YYYY-MM-DD), e.g. 2025-01-16. "
             "If set, overrides --smoke-days."
    )
    
    parser.add_argument(
        "--qa-limit",
        type=int,
        default=None,
        help="Limit number of QA questions to process (for testing)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override base URL for memory system (e.g., http://0.0.0.0:19004 for evermemos local)"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    console = get_console()
    
    try:
        # Handle LLM system stage requirements
        # LLM answer needs dialogue loaded (add) and formatted (search)
        # LLM evaluate-only can work from saved answer results
        if args.system == "llm":
            if "answer" in args.stages:
                if "add" not in args.stages:
                    args.stages = ["add"] + args.stages
                if "search" not in args.stages:
                    idx = min(
                        args.stages.index("answer") if "answer" in args.stages else len(args.stages),
                        args.stages.index("evaluate") if "evaluate" in args.stages else len(args.stages)
                    )
                    args.stages.insert(idx, "search")
                console.print("\n[yellow]ℹ️  LLM system: using full dialogue as context (no memory retrieval)[/yellow]")

        # Validate --dataset for stages that need it
        if "add" in args.stages and not args.dataset:
            print_error("--dataset argument required for add stage")
            sys.exit(1)

        # Load dataset only when needed
        dataset = None
        if args.dataset:
            dataset = load_groupchat_dataset(args.dataset)

        # Generate user_id if not provided
        user_id = args.user_id
        if user_id is None:
            import time
            if args.dataset:
                dataset_num = Path(args.dataset).parent.name
            else:
                dataset_num = "unknown"
            timestamp = int(time.time())
            user_id = f"groupchat_{dataset_num}_{args.system}_{timestamp}"

        # Create output directory
        # Memory systems: {base}/{system}/
        # LLM system:     {base}/llm/{answer_model}/
        if args.system == "llm":
            pipeline_cfg = load_yaml(str(get_config_path("pipeline.yaml")))
            answer_model = pipeline_cfg.get("answer", {}).get("model", "unknown")
            output_dir = Path(args.output_dir) / "llm" / answer_model
        else:
            output_dir = Path(args.output_dir) / args.system
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create adapter
        adapter = create_adapter(args.system, output_dir, base_url=args.base_url)

        # Create pipeline
        pipeline = Pipeline(adapter, output_dir, system_name=args.system)

        # Determine smoke test settings
        smoke_days = None
        smoke_date = None
        if args.smoke:
            if args.smoke_date:
                smoke_date = args.smoke_date
                smoke_days = None
            else:
                smoke_days = args.smoke_days

        # Validate QA path for search/answer/evaluate stages
        if any(s in args.stages for s in ["search", "answer", "evaluate"]):
            if not args.qa:
                print_error("--qa argument required for search/answer/evaluate stages")
                sys.exit(1)

        # Validate LLM API key for answer/evaluate stages
        if any(s in args.stages for s in ["answer", "evaluate"]):
            if not os.environ.get("LLM_API_KEY"):
                print_error("LLM_API_KEY environment variable required for answer/evaluate stages")
                console.print("Please set LLM_API_KEY in your .env file (OpenRouter API key)", style="dim")
                sys.exit(1)

        # Run pipeline
        results = await pipeline.run(
            dataset=dataset,
            user_id=user_id,
            stages=args.stages,
            smoke_days=smoke_days,
            smoke_date=smoke_date,
            start_date=args.start_date,
            qa_path=args.qa,
            top_k=args.top_k,
            qa_limit=args.qa_limit,
        )
        
        # Exit with appropriate code
        if "add" in results:
            add_result = results["add"]
            if not add_result.success:
                sys.exit(1)
        
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)
    except ValueError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
