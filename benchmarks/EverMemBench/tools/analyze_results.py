#!/usr/bin/env python3
"""
Analyze evaluation results by question_id categories.

Supports single-file analysis and multi-batch aggregation.

Usage:
    # Single file
    python tools/analyze_results.py eval/results/evermemos/evaluation_results_004.json

    # Multi-batch (scan directory for evaluation_results_*.json)
    python tools/analyze_results.py --results-dir eval/results/evermemos/

    # Multi-batch (derive directory from system name)
    python tools/analyze_results.py --system evermemos

    # Common options
    python tools/analyze_results.py --system evermemos -o report.json -q
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


def parse_question_id(question_id: str) -> tuple[str, str]:
    """
    Parse question_id to extract major and minor categories.
    Example: "MA_U_Top004_031" -> ("MA", "U")
    Example: "P_Skill_Top004_001" -> ("P", "Skill")
    """
    parts = question_id.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "Unknown", "Unknown"


def calculate_category_stats(detailed_results: list[dict]) -> dict:
    """Calculate accuracy stats by major/minor/combined/hierarchical categories."""
    stats = {
        "major": defaultdict(lambda: {"total": 0, "correct": 0}),
        "minor": defaultdict(lambda: {"total": 0, "correct": 0}),
        "combined": defaultdict(lambda: {"total": 0, "correct": 0}),
        "hierarchical": defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0})),
    }

    for item in detailed_results:
        question_id = item.get("question_id", "")
        is_correct = item.get("is_correct", False)
        major, minor = parse_question_id(question_id)
        combined_key = f"{major}_{minor}"

        stats["major"][major]["total"] += 1
        stats["minor"][minor]["total"] += 1
        stats["combined"][combined_key]["total"] += 1
        stats["hierarchical"][major][minor]["total"] += 1

        if is_correct:
            stats["major"][major]["correct"] += 1
            stats["minor"][minor]["correct"] += 1
            stats["combined"][combined_key]["correct"] += 1
            stats["hierarchical"][major][minor]["correct"] += 1

    return stats


def format_category_results(stats: dict) -> dict:
    """Format category statistics with accuracy percentages."""
    formatted = {}

    for category_type, data in stats.items():
        if category_type == "hierarchical":
            formatted[category_type] = {}
            for major, minor_data in sorted(data.items()):
                formatted[category_type][major] = {}
                for minor, values in sorted(minor_data.items()):
                    total = values["total"]
                    correct = values["correct"]
                    accuracy = correct / total if total > 0 else 0
                    formatted[category_type][major][minor] = {
                        "total": total,
                        "correct": correct,
                        "accuracy": round(accuracy, 4),
                        "accuracy_percent": f"{accuracy * 100:.2f}%",
                    }
        else:
            formatted[category_type] = {}
            for key, values in sorted(data.items()):
                total = values["total"]
                correct = values["correct"]
                accuracy = correct / total if total > 0 else 0
                formatted[category_type][key] = {
                    "total": total,
                    "correct": correct,
                    "accuracy": round(accuracy, 4),
                    "accuracy_percent": f"{accuracy * 100:.2f}%",
                }

    return formatted


def print_report(
    formatted: dict,
    overall: dict,
    *,
    batch_stats: dict | None = None,
    system_name: str = "",
):
    """
    Print formatted report.

    In multi-file mode (batch_stats provided), shows per-batch summary and
    question type breakdown. In single-file mode, skips those sections.
    """
    multi = batch_stats is not None

    title = f"{system_name} Evaluation Results" if system_name else "Evaluation Results"
    if multi:
        title += f" - Aggregated from {len(batch_stats)} Batches"

    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

    # Per-batch summary (multi-file only)
    if multi:
        print("\n  Per-Batch Summary:")
        print("-" * 60)
        print(f"{'Batch':<10} {'Total':>10} {'Correct':>10} {'Accuracy':>15}")
        print("-" * 60)

        for batch_id, stats in sorted(batch_stats.items()):
            acc_pct = f"{stats['accuracy'] * 100:.2f}%"
            print(f"{batch_id:<10} {stats['total_questions']:>10} {stats['correct']:>10} {acc_pct:>15}")

        print("-" * 60)
        print(f"{'Total':<10} {overall['total_questions']:>10} {overall['correct']:>10} {overall['accuracy_percent']:>15}")

    # By question type
    if overall.get("accuracy_by_type"):
        print("\n  By Question Type:")
        print("-" * 60)
        print(f"{'Type':<20} {'Total':>10} {'Correct':>10} {'Accuracy':>15}")
        print("-" * 60)

        for qtype, data in sorted(overall["accuracy_by_type"].items()):
            acc_pct = f"{data['accuracy'] * 100:.2f}%"
            print(f"{qtype:<20} {data['total']:>10} {data['correct']:>10} {acc_pct:>15}")

    # Overall (single-file mode shows it here since there's no batch table)
    if not multi:
        print(f"\n  Overall: {overall['total_questions']} questions, "
              f"{overall['correct']} correct, {overall['accuracy_percent']}")

    # Major category accuracy
    print("\n  Major Category Accuracy:")
    print("-" * 60)
    print(f"{'Category':<10} {'Total':>10} {'Correct':>10} {'Accuracy':>15}")
    print("-" * 60)

    for key, values in sorted(formatted["major"].items()):
        print(f"{key:<10} {values['total']:>10} {values['correct']:>10} {values['accuracy_percent']:>15}")

    # Minor category accuracy
    print("\n  Minor Category Accuracy:")
    print("-" * 60)
    print(f"{'Category':<10} {'Total':>10} {'Correct':>10} {'Accuracy':>15}")
    print("-" * 60)

    for key, values in sorted(formatted["minor"].items()):
        print(f"{key:<10} {values['total']:>10} {values['correct']:>10} {values['accuracy_percent']:>15}")

    # Hierarchical (major -> minor)
    print("\n  Hierarchical (Major -> Minor) Accuracy:")
    print("-" * 65)

    for major, minor_data in sorted(formatted["hierarchical"].items()):
        major_total = sum(v["total"] for v in minor_data.values())
        major_correct = sum(v["correct"] for v in minor_data.values())
        major_acc = major_correct / major_total if major_total > 0 else 0

        print(f"\n  {major} (Total: {major_total}, Correct: {major_correct}, {major_acc*100:.2f}%)")
        print(f"  {'-' * 55}")
        print(f"  {'Minor':<12} {'Total':>10} {'Correct':>10} {'Accuracy':>15}")
        print(f"  {'-' * 55}")

        for minor, values in sorted(minor_data.items()):
            print(f"  {minor:<12} {values['total']:>10} {values['correct']:>10} {values['accuracy_percent']:>15}")

    # Combined category accuracy
    print("\n\n  Combined (Major_Minor) Accuracy:")
    print("-" * 60)
    print(f"{'Category':<15} {'Total':>10} {'Correct':>10} {'Accuracy':>15}")
    print("-" * 60)

    for key, values in sorted(formatted["combined"].items()):
        print(f"{key:<15} {values['total']:>10} {values['correct']:>10} {values['accuracy_percent']:>15}")

    print("\n" + "=" * 70)


def run_single_file(input_path: Path) -> dict:
    """Analyze a single evaluation results file. Returns output_data dict."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    detailed_results = data.get("detailed_results", [])
    if not detailed_results:
        raise ValueError(f"No 'detailed_results' found in {input_path}")

    total_questions = data.get("total_questions", len(detailed_results))
    total_correct = data.get("correct", sum(1 for r in detailed_results if r.get("is_correct")))
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

    # Aggregate by question type
    accuracy_by_type = {}
    if data.get("accuracy_by_type"):
        accuracy_by_type = data["accuracy_by_type"]
    else:
        type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        for item in detailed_results:
            qtype = item.get("question_type", "unknown")
            type_stats[qtype]["total"] += 1
            if item.get("is_correct"):
                type_stats[qtype]["correct"] += 1
        for qtype, s in type_stats.items():
            accuracy_by_type[qtype] = {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0,
            }

    overall = {
        "total_questions": total_questions,
        "correct": total_correct,
        "accuracy": overall_accuracy,
        "accuracy_percent": f"{overall_accuracy * 100:.2f}%",
        "accuracy_by_type": accuracy_by_type,
    }

    stats = calculate_category_stats(detailed_results)
    formatted = format_category_results(stats)

    return {
        "source_file": str(input_path),
        "overall": overall,
        "accuracy_by_major_category": formatted["major"],
        "accuracy_by_minor_category": formatted["minor"],
        "accuracy_by_hierarchical": formatted["hierarchical"],
        "accuracy_by_combined_category": formatted["combined"],
        "_formatted": formatted,
        "_overall": overall,
    }


def run_multi_file(results_dir: Path, system_name: str = "") -> dict:
    """Aggregate evaluation results from all matching files in a directory."""
    files = sorted(results_dir.glob("evaluation_results_*.json"))
    if not files:
        raise ValueError(f"No evaluation_results_*.json found in {results_dir}")

    all_results = {}
    for fpath in files:
        # Extract batch id from filename: evaluation_results_004.json -> 004
        batch_id = fpath.stem.replace("evaluation_results_", "")
        with open(fpath, "r", encoding="utf-8") as f:
            all_results[batch_id] = json.load(f)
        print(f"  Loaded: {fpath.name} ({all_results[batch_id].get('total_questions', 0)} questions)")

    # Per-batch stats
    batch_stats = {}
    for batch_id, data in all_results.items():
        batch_stats[batch_id] = {
            "total_questions": data.get("total_questions", 0),
            "correct": data.get("correct", 0),
            "accuracy": data.get("accuracy", 0),
            "accuracy_by_type": data.get("accuracy_by_type", {}),
        }

    # Overall stats
    total_questions = sum(d.get("total_questions", 0) for d in all_results.values())
    total_correct = sum(d.get("correct", 0) for d in all_results.values())
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

    # Aggregate by question type
    type_agg = defaultdict(lambda: {"total": 0, "correct": 0})
    for data in all_results.values():
        for qtype, stats in data.get("accuracy_by_type", {}).items():
            type_agg[qtype]["total"] += stats.get("total", 0)
            type_agg[qtype]["correct"] += stats.get("correct", 0)

    accuracy_by_type = {}
    for qtype, s in type_agg.items():
        accuracy_by_type[qtype] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0,
        }

    overall = {
        "total_questions": total_questions,
        "correct": total_correct,
        "accuracy": overall_accuracy,
        "accuracy_percent": f"{overall_accuracy * 100:.2f}%",
        "accuracy_by_type": accuracy_by_type,
    }

    # Category stats from all detailed results
    all_detailed = []
    for data in all_results.values():
        all_detailed.extend(data.get("detailed_results", []))

    stats = calculate_category_stats(all_detailed)
    formatted = format_category_results(stats)

    return {
        "source_directory": str(results_dir),
        "system": system_name,
        "batches_found": sorted(all_results.keys()),
        "overall": overall,
        "per_batch": batch_stats,
        "accuracy_by_major_category": formatted["major"],
        "accuracy_by_minor_category": formatted["minor"],
        "accuracy_by_hierarchical": formatted["hierarchical"],
        "accuracy_by_combined_category": formatted["combined"],
        "_formatted": formatted,
        "_overall": overall,
        "_batch_stats": batch_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze evaluation results by question_id categories"
    )
    parser.add_argument(
        "input_file",
        type=str,
        nargs="?",
        default=None,
        help="Single evaluation results JSON file",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing evaluation_results_*.json files",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Memory system name, derives results-dir as eval/results/{system}/",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON report path (optional)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - don't print to console",
    )

    args = parser.parse_args()

    # Validate mutually exclusive inputs
    if args.input_file and (args.results_dir or args.system):
        parser.error("positional input_file is mutually exclusive with --results-dir / --system")
    if args.results_dir and args.system:
        parser.error("--results-dir and --system are mutually exclusive")
    if not args.input_file and not args.results_dir and not args.system:
        parser.error("provide a file, --results-dir, or --system")

    # Single-file mode
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: file not found - {input_path}")
            return 1

        output_data = run_single_file(input_path)

        if not args.quiet:
            print_report(output_data["_formatted"], output_data["_overall"])

    # Multi-file mode
    else:
        if args.results_dir:
            results_dir = Path(args.results_dir)
            system_name = results_dir.name
        else:
            results_dir = Path(f"eval/results/{args.system}")
            system_name = args.system

        if not results_dir.exists():
            print(f"Error: directory not found - {results_dir}")
            return 1

        print(f"Loading evaluation results from {results_dir}...")
        output_data = run_multi_file(results_dir, system_name=system_name)

        if not args.quiet:
            print_report(
                output_data["_formatted"],
                output_data["_overall"],
                batch_stats=output_data["_batch_stats"],
                system_name=system_name,
            )

    # Save JSON report (strip internal keys)
    if args.output:
        save_data = {k: v for k, v in output_data.items() if not k.startswith("_")}
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
