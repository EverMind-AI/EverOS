import asyncio
import json
import logging

import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


from evaluation.src.adapters.evermemos.config import ExperimentConfig

logging.basicConfig(level=logging.CRITICAL)


class LLMGrade(BaseModel):
    llm_judgment: str = Field(description="CORRECT or WRONG")
    llm_reasoning: str = Field(
        description="Explain why the answer is correct or incorrect."
    )


async def locomo_grader(
    llm_client, question: str, gold_answer: str, response: str
) -> bool:
    system_prompt = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """

    accuracy_prompt = f"""
    Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a ’gold’ (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it’s time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """

    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": accuracy_prompt},
        ],
        temperature=0,
    )
    message_content = response.choices[0].message.content
    label = json.loads(message_content)["label"]
    parsed = LLMGrade(llm_judgment=label, llm_reasoning="")

    return parsed.llm_judgment.strip().lower() == "correct"



def convert_numpy_types(obj):
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


async def process_group_responses(
    group_id, group_responses, oai_client, num_runs: int, disable_progress: bool = False
):
    graded_responses = []

    # Process responses with asyncio for concurrent API calls
    for response in tqdm(
        group_responses, 
        desc=f"Processing {group_id}", 
        disable=disable_progress,
        leave=False  # Clear progress bar after completion
    ):
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
            locomo_grader(oai_client, question, ground_truth, answer)
            for _ in range(num_runs)
        ]
        judgments = await asyncio.gather(*grading_tasks)
        judgments_dict = {f"judgment_{i + 1}": j for i, j in enumerate(judgments)}

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

    return group_id, graded_responses


async def process_single_group(
    group_id, group_responses, oai_client, num_runs, disable_progress: bool = False
):
    try:
        start_time = time.time()
        result = await process_group_responses(
            group_id, group_responses, oai_client, num_runs, disable_progress
        )
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        # Reduced output: only print on error or important moments
        # print(f"Group {group_id} processed in {elapsed_time} seconds")
        return result
    except Exception as e:
        print(f"❌ Error processing group {group_id}: {e}")
        return group_id, []


async def main():
    # --- Configuration ---
    config = ExperimentConfig()
    version = config.experiment_name
    num_runs = 3
    max_workers = 10

    print(f"\n{'='*60}")
    print(f"🔍 Stage5: LLM-as-a-Judge Evaluation")
    print(f"{'='*60}")

    # --- Path Setup ---
    current_dir = Path(__file__).parent
    results_dir = current_dir / version
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
        with open(response_path) as file:
            locomo_responses = json.load(file)
    except FileNotFoundError:
        print(f"❌ Error: Response file not found at {response_path}")
        return

    # --- Evaluation ---
    num_users = 10
    all_grades = {}

    total_responses_count = sum(
        len(locomo_responses.get(f"locomo_exp_user_{i}", [])) for i in range(num_users)
    )
    print(f"📊 Total responses: {total_responses_count}")
    print(f"👥 Total users: {num_users}")
    print(f"🔄 Judgments per question: {num_runs}")
    print(f"⚡ Concurrent workers: {max_workers}")
    print(f"{'='*60}\n")

    # Create tasks for processing each group
    tasks = []
    active_users = 0
    for group_idx in range(num_users):
        group_id = f"locomo_exp_user_{group_idx}"
        group_responses = locomo_responses.get(group_id, [])
        if not group_responses:
            continue

        active_users += 1
        tasks.append(
            process_single_group(
                group_id, group_responses, oai_client, num_runs, disable_progress=True  # Disable individual group progress bar
            )
        )

    print(f"🚀 Starting evaluation...\n")

    semaphore = asyncio.Semaphore(max_workers)

    async def limited_task(task):
        async with semaphore:
            return await task

    limited_tasks = [limited_task(task) for task in tasks]
    
    # Add overall progress bar
    group_results = []
    for coro in tqdm(
        asyncio.as_completed(limited_tasks),
        total=len(limited_tasks),
        desc="📈 Evaluating groups",
        unit="group"
    ):
        result = await coro
        group_results.append(result)

    for group_id, graded_responses in group_results:
        all_grades[group_id] = graded_responses

    print(f"\n{'='*60}")
    print(f"✅ Evaluation Complete")
    print(f"{'='*60}")

    # --- Score Calculation ---
    run_scores = []
    evaluated_count = 0
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

    # Simplified and formatted result output
    if evaluated_count > 0:
        mean_of_scores = np.mean(run_scores)
        std_of_scores = np.std(run_scores)
        
        print(f"\n{'='*60}")
        print(f"📊 Final Results")
        print(f"{'='*60}")
        print(f"🎯 Mean Accuracy:  {mean_of_scores:.4f} ({mean_of_scores*100:.2f}%)")
        print(f"📈 Std Deviation:  {std_of_scores:.4f}")
        print(f"📝 Questions:      {evaluated_count}")
        print(f"🔄 Runs per Q:     {num_runs}")
        print(f"{'='*60}")
        print(f"Individual run scores: {[round(s, 4) for s in run_scores]}")
        print(f"{'='*60}\n")
    else:
        print("\n⚠️  No responses were evaluated")
        print("LLM-as-a-Judge score: N/A (0/0)\n")

    # --- Save Results ---
    all_grades = convert_numpy_types(all_grades)
    with open(judged_path, "w") as f:
        json.dump(all_grades, f, indent=2)
        print(f"💾 Saved detailed results to: {judged_path}")
        print(f"✅ Stage5 complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
