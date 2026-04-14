#!/usr/bin/env python3
"""
HyperMem Full Evaluation Pipeline

Executes the Stage 1-5 evaluation pipeline in order:
1. Stage 1: Hypergraph Extraction
2. Stage 2: Index Building
3. Stage 3: Hypergraph Retrieval
4. Stage 4: Response Generation
5. Stage 5: Evaluation

Usage:
    python eval.py                    # Run the full pipeline
    python eval.py --start 3          # Start from stage 3
    python eval.py --end 3            # Run up to stage 3
    python eval.py --start 2 --end 4  # Run stages 2-4
    python eval.py --stages 1 3 5     # Run only the specified stages
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Get the current directory
EVAL_DIR = Path(__file__).parent.absolute()


def get_stage_info():
    """Get information for each stage"""
    return {
        1: {
            "name": "Memory Extraction",
            "script": "stage1_memory_extraction.py",
            "description": "Detect episode boundaries in conversations and extract Episodes",
        },
        2: {
            "name": "Hypergraph Extraction",
            "script": "stage2_hypergraph_extraction.py",
            "description": "Extract hypergraph structure from Episodes (Episode -> Topic -> Fact)",
        },
        3: {
            "name": "Index Building",
            "script": "stage3_hypergraph_index.py",
            "description": "Build BM25 and vector retrieval indexes",
        },
        4: {
            "name": "Hypergraph Retrieval",
            "script": "stage4_hypergraph_retrieval.py",
            "description": "Perform hypergraph retrieval to obtain relevant memories",
        },
        5: {
            "name": "Response Generation",
            "script": "stage5_response.py",
            "description": "Generate answers based on retrieval results",
        },
        6: {
            "name": "Evaluation",
            "script": "stage6_eval.py",
            "description": "Evaluate the quality of generated answers",
        },
    }


def run_stage(stage_num: int, stage_info: dict) -> bool:
    """
    Run a single stage

    Args:
        stage_num: Stage number
        stage_info: Stage information

    Returns:
        Whether the stage succeeded
    """
    script_path = EVAL_DIR / stage_info["script"]
    
    if not script_path.exists():
        print(f"  ❌ Script not found: {script_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"📍 Stage {stage_num}: {stage_info['name']}")
    print(f"   {stage_info['description']}")
    print(f"   Script: {stage_info['script']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Clear pycache before running to avoid stale config
        import shutil
        pycache_dir = EVAL_DIR.parent / "__pycache__"
        if pycache_dir.exists():
            shutil.rmtree(pycache_dir)

        # Print resolved experiment name for debugging
        env_name = os.environ.get("HYPERMEM_EXPERIMENT_NAME", "(default)")
        print(f"   Env: HYPERMEM_EXPERIMENT_NAME={env_name}")

        # Run the script using subprocess, inheriting env vars
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(EVAL_DIR.parent.parent),
            env=os.environ.copy(),
            check=True,
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ Stage {stage_num} completed (elapsed: {elapsed:.1f}s)")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Stage {stage_num} failed (elapsed: {elapsed:.1f}s)")
        print(f"   Error code: {e.returncode}")
        return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Stage {stage_num} exception (elapsed: {elapsed:.1f}s)")
        print(f"   Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="HyperMem Full Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py                    # Run full pipeline (stage 1-5)
  python eval.py --start 3          # Start running from stage 3
  python eval.py --end 3            # Run up to stage 3
  python eval.py --start 2 --end 4  # Run stages 2-4
  python eval.py --stages 1 3 5     # Run only specified stages
  python eval.py --list             # List all stages
        """
    )
    
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=1,
        help="Starting stage (default: 1)"
    )
    
    parser.add_argument(
        "--end", "-e", 
        type=int,
        default=6,
        help="Ending stage (default: 6)"
    )
    
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        help="Specify stages to run (overrides --start/--end)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available stages"
    )
    
    parser.add_argument(
        "--continue-on-error", "-c",
        action="store_true",
        help="Continue running subsequent stages even if a stage fails"
    )
    
    args = parser.parse_args()
    
    stage_info = get_stage_info()
    
    # List all stages
    if args.list:
        print("\n📋 Available Stages:")
        print("-" * 60)
        for num, info in stage_info.items():
            print(f"  Stage {num}: {info['name']}")
            print(f"           {info['description']}")
            print(f"           Script: {info['script']}")
            print()
        return
    
    # Determine which stages to run
    if args.stages:
        stages_to_run = sorted([s for s in args.stages if s in stage_info])
    else:
        stages_to_run = [s for s in range(args.start, args.end + 1) if s in stage_info]
    
    if not stages_to_run:
        print("❌ No valid stages to run")
        return
    
    # Print execution plan
    print("\n" + "=" * 60)
    print("🚀 HyperMem Evaluation Pipeline")
    print("=" * 60)
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Execution plan: Stage {', '.join(map(str, stages_to_run))}")
    print("=" * 60)
    
    # Run each stage
    total_start = time.time()
    results = {}
    
    for stage_num in stages_to_run:
        success = run_stage(stage_num, stage_info[stage_num])
        results[stage_num] = success
        
        if not success and not args.continue_on_error:
            print(f"\n⚠️ Stage {stage_num} failed, aborting subsequent stages")
            print("   Use --continue-on-error to continue running subsequent stages")
            break
    
    # Print summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("📊 Execution Summary")
    print("=" * 60)
    print(f"⏱️  Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print()
    
    success_count = sum(1 for r in results.values() if r)
    fail_count = sum(1 for r in results.values() if not r)
    
    for stage_num, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   Stage {stage_num}: {status}")
    
    print()
    print(f"   Succeeded: {success_count}/{len(results)}")

    if fail_count > 0:
        print(f"   Failed: {fail_count}/{len(results)}")
        sys.exit(1)
    else:
        print("\n🎉 All stages completed successfully!")


if __name__ == "__main__":
    main()

