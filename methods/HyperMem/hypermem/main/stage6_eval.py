import asyncio
import json
import logging
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import sys
import time
import traceback
from pathlib import Path

import nltk
import numpy as np
import transformers
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hypermem.config import ExperimentConfig

console = Console()

logging.basicConfig(level=logging.CRITICAL)
transformers.logging.set_verbosity_error()

# Download necessary NLTK resources
try:
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    print("NLTK resources downloaded successfully.")
except Exception as e:
    print(f"Warning: Failed to download NLTK resources: {e}")

try:
    sentence_model_name = "Qwen/Qwen3-Embedding-0.6B"
    # sentence_model = SentenceTransformer(sentence_model_name)
    print(f"SentenceTransformer model : {sentence_model_name} loaded successfully.")
except Exception as e:
    print(f"Failed to load SentenceTransformer model: {e}")
    sentence_model = None


class LLMGrade(BaseModel):
    llm_judgment: str = Field(description="CORRECT or WRONG")
    llm_reasoning: str = Field(description="Explain why the answer is correct or incorrect.")


async def locomo_grader(llm_client, question: str, gold_answer: str, response: str, max_retries: int = 5) -> bool:
    """
    Use LLM to grade the generated answer and determine whether it is correct.

    Args:
        llm_client: Async OpenAI client
        question: Question text
        gold_answer: Ground truth answer
        response: Generated answer
        max_retries: Maximum number of retries, default is 5

    Returns:
        bool: Whether the answer is correct
    """
    system_prompt = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """

    accuracy_prompt = f"""
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """

    # Retry mechanism: handle temporary API failures (e.g., 401 User not found, network fluctuations, etc.)
    last_exception = None
    for attempt in range(max_retries):
        try:
            api_response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": accuracy_prompt},
                ],
                temperature=0,
            )
            message_content = api_response.choices[0].message.content
            
            # Try to parse JSON directly
            try:
                label = json.loads(message_content)["label"]
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from text
                # Match {"label": "..."} or ```json\n{"label": "..."}\n``` format
                json_match = re.search(r'\{[^{}]*"label"\s*:\s*"[^"]+"\s*[^{}]*\}', message_content)
                if json_match:
                    label = json.loads(json_match.group())["label"]
                else:
                    # If still not found, try to find CORRECT or WRONG directly in the text
                    if "CORRECT" in message_content.upper() and "WRONG" not in message_content.upper():
                        label = "CORRECT"
                    elif "WRONG" in message_content.upper() and "CORRECT" not in message_content.upper():
                        label = "WRONG"
                    else:
                        raise ValueError(f"Unable to extract label from response: {message_content[:200]}")
            
            parsed = LLMGrade(llm_judgment=label, llm_reasoning="")
            return parsed.llm_judgment.strip().lower() == "correct"
        except Exception as e:
            last_exception = e
            # If not the last retry, wait and retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s, 4s, 8s
                console.print(f"[yellow]  API call error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}[/yellow]")
                await asyncio.sleep(wait_time)
            continue
    
    # All retries failed, raise the last exception
    raise last_exception


def calculate_rouge_scores(gold_answer, response):
    metrics = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(gold_answer, response)
        metrics["rouge1_f"] = rouge_scores["rouge1"].fmeasure
        metrics["rouge2_f"] = rouge_scores["rouge2"].fmeasure
        metrics["rougeL_f"] = rouge_scores["rougeL"].fmeasure
    except Exception as e:
        print(f"Failed to calculate ROUGE scores: {e}")
    return metrics


def calculate_bleu_scores(gold_tokens, response_tokens):
    metrics = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    try:
        smoothing = SmoothingFunction().method1
        weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]

        for i, weight in enumerate(weights, 1):
            metrics[f"bleu{i}"] = sentence_bleu(
                [gold_tokens], response_tokens, weights=weight, smoothing_function=smoothing
            )
    except ZeroDivisionError:
        pass
    except Exception as e:
        print(f"Failed to calculate BLEU scores: {e}")

    return metrics


def calculate_meteor_score(gold_tokens, response_tokens):
    try:
        return meteor_score([gold_tokens], response_tokens)
    except Exception as e:
        print(f"Failed to calculate METEOR score: {e}")
        return 0.0


def calculate_semantic_similarity(gold_answer, response):
    global sentence_model

    try:
        if sentence_model is None:
            sentence_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

        gold_embedding = sentence_model.encode([gold_answer], show_progress_bar=False)[0]
        response_embedding = sentence_model.encode([response], show_progress_bar=False)[0]
        return 1 - cosine(gold_embedding, response_embedding)
    except Exception as e:
        print(f"Failed to calculate semantic similarity: {e}")
        return 0.0


def calculate_f1_score(gold_tokens, response_tokens):
    try:
        gold_set = set(gold_tokens)
        response_set = set(response_tokens)

        if len(gold_set) == 0 or len(response_set) == 0:
            return 0.0

        precision = len(gold_set.intersection(response_set)) / len(response_set)
        recall = len(gold_set.intersection(response_set)) / len(gold_set)

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    except Exception as e:
        print(f"Failed to calculate F1 score: {e}")
        return 0.0


def calculate_nlp_metrics(gold_answer, response, context, options=None):
    if options is None:
        options = ["lexical", "semantic"]

    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""

    metrics = {"context_tokens": len(nltk.word_tokenize(context)) if context else 0}

    if "lexical" in options:
        gold_tokens = nltk.word_tokenize(gold_answer.lower())
        response_tokens = nltk.word_tokenize(response.lower())

        metrics["lexical"] = {}
        metrics["lexical"]["f1"] = calculate_f1_score(gold_tokens, response_tokens)
        metrics["lexical"].update(calculate_rouge_scores(gold_answer, response))
        metrics["lexical"].update(calculate_bleu_scores(gold_tokens, response_tokens))
        metrics["lexical"]["meteor"] = calculate_meteor_score(gold_tokens, response_tokens)

    if "semantic" in options:
        metrics["semantic"] = {}
        metrics["semantic"]["similarity"] = calculate_semantic_similarity(gold_answer, response)
        _, _, f1 = bert_score(
            [gold_answer], [response], lang="en", rescale_with_baseline=True, verbose=False
        )
        metrics["semantic"]["bert_f1"] = f1.item() if f1 is not None else 0.0

    return metrics


def convert_numpy_types(obj):
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


async def process_group_responses(group_id, group_responses, oai_client, options, num_runs: int, progress: Progress = None, task_id: int = None):
    graded_responses = []

    # Process responses with asyncio for concurrent API calls
    for idx, response in enumerate(group_responses):
        if progress and task_id is not None:
            progress.update(task_id, completed=idx)
        
        question = response.get("question")
        answer = response.get("answer")
        ground_truth = response.get("golden_answer")
        category = response.get("category")

        context = response.get("search_context", "")
        response_duration_ms = response.get("response_duration_ms", 0.0)
        search_duration_ms = response.get("search_duration_ms", 0.0)

        if ground_truth is None:
            continue

        grading_tasks = [
            locomo_grader(oai_client, question, ground_truth, answer) for _ in range(num_runs)
        ]
        judgments = await asyncio.gather(*grading_tasks)
        judgments_dict = {f"judgment_{i + 1}": j for i, j in enumerate(judgments)}

        # nlp_metrics = calculate_nlp_metrics(ground_truth, answer, context, options)
        nlp_metrics = {}
        graded_response = {
            "question": question,
            "answer": answer,
            "golden_answer": ground_truth,
            "category": category,
            "llm_judgments": judgments_dict,
            "nlp_metrics": nlp_metrics,
            "response_duration_ms": response_duration_ms,
            "search_duration_ms": search_duration_ms,
            "total_duration_ms": response_duration_ms + search_duration_ms,
        }
        graded_responses.append(graded_response)
    
    if progress and task_id is not None:
        progress.update(task_id, completed=len(group_responses))

    return group_id, graded_responses


async def process_single_group(group_id, group_responses, oai_client, options, num_runs, progress: Progress = None, task_id: int = None):
    try:
        start_time = time.time()
        if progress and task_id is not None:
            progress.start_task(task_id)
        
        result = await process_group_responses(
            group_id, group_responses, oai_client, options, num_runs, progress, task_id
        )
        
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        
        if progress and task_id is not None:
            progress.update(task_id, status=f"[green]Done ({elapsed_time}s)[/green]")
        
        return result
    except Exception as e:
        console.print(f"[red][X] Conversation {group_id}: Evaluation failed - {type(e).__name__}: {e}[/red]")
        console.print(f"[red]    Full stack trace:[/red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        if progress and task_id is not None:
            progress.update(task_id, status="[red]Failed[/red]")
        return group_id, []


async def main():
    # --- Configuration ---
    frame = "cot"
    config = ExperimentConfig()
    version = config.experiment_name
    num_runs = 3
    options = ["lexical", "semantic"]
    max_workers = 10

    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print(f"[bold cyan]Stage 6: LoCoMo Evaluation ({frame}, version: {version})[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    console.print(f"[bold]Evaluation rounds:[/bold] {num_runs}")
    console.print(f"[bold]Concurrency:[/bold] {max_workers}\n")

    # --- Path Setup ---
    results_dir = config.experiment_dir()
    response_path = results_dir / "responses.json"
    judged_path = results_dir / "judged.json"

    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Client Setup ---
    llm_config = config.llm_config["openai"]
    oai_client = AsyncOpenAI(
        api_key=llm_config["api_key"], base_url=llm_config["base_url"]
    )

    # --- Data Loading ---
    try:
        with open(response_path, encoding="utf-8") as file:
            locomo_responses = json.load(file)
    except FileNotFoundError:
        console.print(f"[red][X] Response file not found: {response_path}[/red]")
        return

    # --- Evaluation ---
    num_users = 10
    all_grades = {}

    total_responses_count = sum(
        len(locomo_responses.get(f"locomo_exp_user_{i}", [])) for i in range(num_users)
    )
    console.print(f"[bold]Total responses:[/bold] {total_responses_count} (from {num_users} conversations)\n")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_workers)

    async def process_with_semaphore(group_id: str, group_responses: list, task_id: int, progress: Progress):
        """Processing function with semaphore-based concurrency control"""
        async with semaphore:
            return await process_single_group(group_id, group_responses, oai_client, options, num_runs, progress, task_id)
    
    # Create progress bar for parallel processing
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed:>3}/{task.total:<3}"),  # Right-align completed count, left-align total
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("[bold]{task.fields[status]}"),
        console=console,
        transient=False
    ) as progress:
        
        # Collect all conversation data
        all_groups_data = []
        active_users = 0
        
        for group_idx in range(num_users):
            group_id = f"locomo_exp_user_{group_idx}"
            group_responses = locomo_responses.get(group_id, [])
            if not group_responses:
                console.print(f"  [yellow][!] Conversation {group_id}: No responses found[/yellow]")
                continue

            active_users += 1
            
            # Create progress task
            task_id = progress.add_task(
                f"[cyan]Conv {group_idx}[/cyan]",
                total=len(group_responses),
                status="Waiting",
                start=False
            )
            all_groups_data.append((group_id, group_responses, task_id))
        
        console.print(f"[bold]Active conversations:[/bold] {active_users}\n")
        
        # Create parallel tasks
        coroutines = [
            process_with_semaphore(group_id, group_responses, task_id, progress)
            for group_id, group_responses, task_id in all_groups_data
        ]
        
        # Process all tasks in parallel
        group_results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Collect results
        for result in group_results:
            if isinstance(result, tuple):
                group_id, graded_responses = result
                all_grades[group_id] = graded_responses
            elif isinstance(result, Exception):
                console.print(f"[red][X] Processing exception: {type(result).__name__}: {result}[/red]")
                console.print(f"[dim]{traceback.format_exception(type(result), result, result.__traceback__)}[/dim]")

    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Evaluation Complete: Calculating Final Scores[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")

    # --- Category Mapping ---
    # Category 1: Multi-Hop (multi-hop reasoning)
    # Category 2: Temporal (temporal reasoning)
    # Category 3: Open Domain (open domain/commonsense)
    # Category 4: Single-Hop (direct fact retrieval)
    category_names = {
        1: "Multi-Hop",
        2: "Temporal",
        3: "Open Domain",
        4: "Single-Hop"
    }

    # --- Score Calculation ---
    run_scores = []
    evaluated_count = 0
    
    # Statistics by category (using majority voting)
    category_stats = {cat: {"correct": 0, "total": 0} for cat in [1, 2, 3, 4]}
    overall_correct = 0
    overall_total = 0
    
    for group in all_grades.values():
        for response in group:
            category = response.get("category")
            judgments = response.get("llm_judgments", {})
            
            # Determine correctness by majority voting
            true_count = sum(1 for v in judgments.values() if v)
            is_correct = true_count >= (num_runs / 2)
            
            # Update overall statistics
            overall_total += 1
            if is_correct:
                overall_correct += 1
            
            # Update category statistics
            if category in category_stats:
                category_stats[category]["total"] += 1
                if is_correct:
                    category_stats[category]["correct"] += 1
    
    # Calculate scores for each round
    if num_runs > 0:
        for i in range(1, num_runs + 1):
            judgment_key = f"judgment_{i}"
            current_run_correct_count = 0
            current_run_total_count = 0
            for group in all_grades.values():
                for response in group:
                    if judgment_key in response["llm_judgments"]:
                        if response["llm_judgments"][judgment_key]:
                            current_run_correct_count += 1
                        current_run_total_count += 1

            if current_run_total_count > 0:
                run_accuracy = current_run_correct_count / current_run_total_count
                run_scores.append(run_accuracy)

        if current_run_total_count > 0:
            evaluated_count = current_run_total_count

    # --- Output LLM Judge results by category ---
    console.print("[bold]LLM-as-a-Judge Results by Category:[/bold]\n")
    console.print("-" * 60)
    console.print(f"{'Category':<25} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
    console.print("-" * 60)
    
    # Output in order: Single-Hop, Multi-Hop, Temporal, Open Domain
    for cat in [4, 1, 2, 3]:
        stats = category_stats[cat]
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"] * 100
            console.print(f"{category_names[cat]:<25} {stats['correct']:>10} {stats['total']:>10} {accuracy:>11.2f}%")
        else:
            console.print(f"{category_names[cat]:<25} {'N/A':>10} {0:>10} {'N/A':>12}")
    
    console.print("-" * 60)
    if overall_total > 0:
        overall_accuracy = overall_correct / overall_total * 100
        console.print(f"{'Overall':<25} {overall_correct:>10} {overall_total:>10} {overall_accuracy:>11.2f}%")
    console.print("-" * 60)
    
    # --- Output detailed statistics ---
    console.print("")
    if evaluated_count > 0:
        mean_of_scores = np.mean(run_scores)
        std_of_scores = np.std(run_scores)
        console.print(f"[bold green]LLM-as-a-Judge average score:[/bold green] {mean_of_scores:.4f}")
        console.print(f"[bold]Standard deviation:[/bold] {std_of_scores:.4f}")
        console.print(f"[bold]Evaluation details:[/bold] {num_runs} rounds of evaluation, {evaluated_count} questions total")
        console.print(f"[bold]Per-round scores:[/bold] {[round(s, 4) for s in run_scores]}")
    else:
        console.print("[yellow]No responses evaluated[/yellow]")
        console.print("[yellow]LLM-as-a-Judge score: N/A (0/0)[/yellow]")

    # --- Save Results ---
    all_grades = convert_numpy_types(all_grades)
    with open(judged_path, "w", encoding="utf-8") as f:
        json.dump(all_grades, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[bold]Saving detailed evaluation results to:[/bold] {judged_path}")
    console.print("[bold green][SUCCESS] Evaluation completed![/bold green]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")


if __name__ == "__main__":
    asyncio.run(main())
