import json
import argparse
from collections import defaultdict
import numpy as np


def compute_accuracy_from_judgments(file_path):
    """
    Computes accuracies, mean, and standard deviation from a JSON file with LLM judgments.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_judgments = 0
    # Determine the number of judgments from the first record
    for user_id, results in data.items():
        if results:
            first_item = results[0]
            if 'llm_judgments' in first_item and isinstance(
                first_item['llm_judgments'], dict
            ):
                num_judgments = len(first_item['llm_judgments'])
            break

    if num_judgments == 0:
        print("No judgments found in the file.")
        return

    judgment_correct = [0] * num_judgments
    total_questions = 0

    category_correct = [defaultdict(int) for _ in range(num_judgments)]
    category_total = defaultdict(int)

    # First pass to get total questions
    for user_id, results in data.items():
        total_questions += len(results)
        for item in results:
            category = item.get("category")
            category_total[category] += 1

    for user_id, results in data.items():
        for item in results:
            judgments = item.get("llm_judgments", {})
            category = item.get("category")
            for i in range(num_judgments):
                if judgments.get(f"judgment_{i+1}", False):
                    judgment_correct[i] += 1
                    category_correct[i][category] += 1

    accuracies = [
        (correct / total_questions) * 100 if total_questions > 0 else 0
        for correct in judgment_correct
    ]

    mean_accuracy = np.mean(accuracies)
    std_dev_accuracy = np.std(accuracies)

    category_accuracies = []
    for i in range(num_judgments):
        cat_acc = {}
        for cat, total in category_total.items():
            correct = category_correct[i][cat]
            cat_acc[cat] = (correct / total) * 100 if total > 0 else 0
        category_accuracies.append(cat_acc)

    return (
        accuracies,
        mean_accuracy,
        std_dev_accuracy,
        category_accuracies,
        category_correct,
        category_total,
    )


if __name__ == "__main__":
    # The path to the results file is hardcoded here.
    # You can change this path if you need to point to a different file.
    results_file_path = "/Users/admin/Documents/Projects/b001-memsys_/evaluation/locomo_evaluation/results/locomo_evaluation_nemori/nemori_locomo_judged.json"

    try:
        (
            accuracies,
            mean_acc,
            std_dev_acc,
            category_accs,
            category_correct,
            category_total,
        ) = compute_accuracy_from_judgments(results_file_path)

        print("--- Overall Accuracy ---")
        for i, acc in enumerate(accuracies):
            print(f"Trial {i+1} Accuracy: {acc:.2f}%")

        print(f"\nMean Accuracy: {mean_acc:.2f}%")
        print(f"Standard Deviation of Accuracy: {std_dev_acc:.2f}")

        print("\n--- Accuracy by Category ---")
        for i, cat_acc in enumerate(category_accs):
            print(f"\nTrial {i+1}:")
            for category, acc in sorted(cat_acc.items()):
                total = category_total.get(category, 0)
                correct = category_correct[i].get(category, 0)
                print(f"  Category {category}: {acc:.2f}% ({correct}/{total})")

    except FileNotFoundError:
        print(f"Error: The file '{results_file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{results_file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
