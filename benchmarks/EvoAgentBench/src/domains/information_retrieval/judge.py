"""LLM Judge for BrowseComp-Plus.

Ported from evoagentbench-bcp/scripts_evaluation/evaluate_run_api.py.
Uses an OpenAI-compatible API (sglang / vllm) to judge answer correctness.
"""

import logging
import re

log = logging.getLogger("evoagentbench")

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question, response=response, correct_answer=correct_answer,
    )


def parse_judge_response(judge_response: str) -> dict:
    """Parse the structured judge response into a result dict."""
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }
    if not judge_response:
        result["parse_error"] = True
        return result

    # Extract extracted_final_answer
    for pattern in [
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
        r"extracted_final_answer:\s*(.*?)(?=\n|$)",
    ]:
        m = re.search(pattern, judge_response, re.IGNORECASE | re.DOTALL)
        if m:
            result["extracted_final_answer"] = m.group(1).strip()
            break

    # Extract correct (yes/no)
    for pattern in [
        r"\*\*correct:\*\*\s*(yes|no)",
        r"\*\*correct\*\*:\s*(yes|no)",
        r"correct:\s*(yes|no)",
    ]:
        m = re.search(pattern, judge_response, re.IGNORECASE)
        if m:
            result["correct"] = m.group(1).lower() == "yes"
            break

    # Extract confidence
    for pattern in [
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?",
        r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?",
        r"confidence:\s*(\d+(?:\.\d+)?)\s*%?",
    ]:
        m = re.search(pattern, judge_response, re.IGNORECASE)
        if m:
            result["confidence"] = min(float(m.group(1)), 100)
            break

    if result["correct"] is None:
        result["parse_error"] = True

    return result


def call_judge(question: str, response: str, correct_answer: str,
               model: str, api_base: str, api_key: str = "EMPTY",
               max_tokens: int = 4096, temperature: float = 0.7,
               disable_thinking: bool = True) -> dict:
    """Call an OpenAI-compatible LLM judge and return parsed result."""
    import os
    import openai

    # Support ${ENV_VAR} syntax in api_key
    if api_key.startswith("${") and api_key.endswith("}"):
        api_key = os.environ.get(api_key[2:-1], "")

    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    prompt = create_judge_prompt(question, response, correct_answer)

    extra_body = {}
    if disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body if extra_body else None,
        )
        judge_text = resp.choices[0].message.content or ""
    except Exception as e:
        log.error(f"Judge API error: {e}")
        return {"parse_error": True, "error": str(e)}

    result = parse_judge_response(judge_text)
    result["judge_response"] = judge_text
    return result
