"""Omni-Math answer verification.

Supports two modes:
  - exact: normalize LaTeX answers and compare strings
  - llm: use an LLM judge to compare mathematical equivalence
"""

import logging
import re

log = logging.getLogger("evoagentbench")


# ---------------------------------------------------------------------------
# LaTeX / string normalization
# ---------------------------------------------------------------------------

def _extract_braced(text: str, start: int) -> tuple[str, int]:
    """Extract content within matched braces starting at text[start] == '{'.
    Returns (content, end_index) where end_index points past the closing '}'."""
    if start >= len(text) or text[start] != '{':
        return '', start
    depth = 1
    i = start + 1
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start + 1:i - 1], i


def _normalize_answer(answer: str) -> str:
    """Normalize a math answer for comparison.

    Strips LaTeX wrappers (\\boxed, $, etc.), whitespace, and common
    formatting differences so that equivalent answers match.
    """
    s = answer.strip()

    # Extract content from \boxed{...} (handles nested braces)
    marker = '\\boxed{'
    pos = s.find(marker)
    if pos != -1:
        content, _ = _extract_braced(s, pos + len(marker) - 1)
        s = content.strip()

    # Strip dollar signs and display-math delimiters
    s = re.sub(r'^\$+|\$+$', '', s)
    s = re.sub(r'^\\\[|\\\]$', '', s)
    s = re.sub(r'^\\\(|\\\)$', '', s)

    # Remove \text{...} wrappers but keep content
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)

    # Convert \frac{a}{b} to a/b (handles nested braces)
    while '\\frac{' in s:
        pos = s.find('\\frac{')
        numer, after_numer = _extract_braced(s, pos + 5)  # pos + len('\\frac') = pos+5
        if after_numer < len(s) and s[after_numer] == '{':
            denom, after_denom = _extract_braced(s, after_numer)
            s = s[:pos] + f"{numer}/{denom}" + s[after_denom:]
        else:
            break  # malformed, stop

    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    # Remove trailing periods
    s = s.rstrip('.')

    # Lowercase for case-insensitive comparison
    s = s.lower()

    return s


def _answers_match(expected: str, actual: str) -> bool:
    """Check if two math answers are equivalent after normalization."""
    norm_expected = _normalize_answer(expected)
    norm_actual = _normalize_answer(actual)

    if norm_expected == norm_actual:
        return True

    # Try numeric comparison for simple numeric answers
    try:
        val_expected = float(norm_expected.replace(',', ''))
        val_actual = float(norm_actual.replace(',', ''))
        if abs(val_expected - val_actual) < 1e-9:
            return True
    except (ValueError, OverflowError):
        pass

    return False


# ---------------------------------------------------------------------------
# LLM-based verification
# ---------------------------------------------------------------------------

_llm_client = None
_llm_base_url = None


def _llm_verify(expected: str, actual: str, problem: str,
                api_key: str, model: str = "openai/gpt-4o",
                api_base: str = "") -> tuple[bool, str]:
    """Use an LLM to judge whether two math answers are equivalent."""
    global _llm_client, _llm_base_url

    base_url = api_base or "https://openrouter.ai/api/v1"
    if _llm_client is None or _llm_base_url != base_url:
        from openai import OpenAI
        _llm_client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
        )
        _llm_base_url = base_url

    prompt = (
        "You are a math judge. Compare the student's answer with the reference answer.\n\n"
        f"**Problem:** {problem}\n\n"
        f"**Reference answer:** {expected}\n\n"
        f"**Student's final answer:** {actual}\n\n"
        "---\n\n"
        "Determine if the student's answer is equivalent to the reference answer.\n"
        "They may look different but be mathematically equivalent "
        "(e.g., 1/2 = 0.5, \u221a2/2 = 1/\u221a2).\n\n"
        "Respond with EXACTLY this format:\n\n"
        "## Student Final Answer\n"
        "<extracted answer>\n\n"
        "## Equivalence Judgement\n"
        "TRUE or FALSE\n\n"
        "## Justification\n"
        "<brief explanation>\n\n"
        "=== report over ==="
    )

    import time
    last_err = None
    for attempt in range(3):
        try:
            response = _llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Empty response from judge model")
            # Parse TRUE/FALSE from "## Equivalence Judgement" section
            upper = text.upper()
            if "EQUIVALENCE JUDGEMENT" in upper:
                after = upper.split("EQUIVALENCE JUDGEMENT", 1)[1]
                is_correct = "TRUE" in after.split("JUSTIFICATION")[0] if "JUSTIFICATION" in after else "TRUE" in after
            else:
                # Fallback: look for TRUE anywhere
                is_correct = "TRUE" in upper and "FALSE" not in upper
            return is_correct, text
        except Exception as e:
            last_err = e
            log.warning(f"LLM verification attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(1)

    # All retries exhausted, fall back to exact match
    log.warning(f"LLM verification failed after 3 attempts, falling back to exact match")
    return _answers_match(expected, actual), f"LLM fallback: {last_err}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_answer(agent_output: str) -> str:
    r"""Extract the final answer from agent output text.

    Priority order (matching original omni-math pipeline):
      1. \boxed{...} (last occurrence)
      2. **Answer:** ...
      3. Answer: ...
      4. Last non-empty line
    """
    if not agent_output:
        return ""

    # Try \boxed{...} first (last occurrence) — handles nested braces
    def _find_boxed(text):
        """Extract content from \\boxed{...} handling nested braces."""
        results = []
        i = 0
        marker = "\\boxed{"
        while i < len(text):
            pos = text.find(marker, i)
            if pos == -1:
                break
            start = pos + len(marker)
            depth = 1
            j = start
            while j < len(text) and depth > 0:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            if depth == 0:
                results.append(text[start:j - 1].strip())
            i = j
        return results

    boxed_matches = _find_boxed(agent_output)
    if boxed_matches:
        return f"\\boxed{{{boxed_matches[-1]}}}"

    # Fallback: **Answer:** pattern
    m = re.search(r'\*\*Answer:\*\*\s*(.+?)(?:\n|$)', agent_output, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback: Answer: pattern
    m = re.search(r'Answer:\s*(.+?)(?:\n|$)', agent_output, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Last non-empty line as fallback
    lines = [l.strip() for l in agent_output.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def verify_answer(task: dict, agent_output: str,
                  mode: str = "exact",
                  api_key: str = "",
                  model: str = "openai/gpt-4o",
                  api_base: str = "") -> dict:
    """Verify an agent's answer against the expected answer.

    Returns dict with: reward, correct, expected, actual, feedback.
    """
    expected = task.get("answer", "")
    actual = extract_answer(agent_output)

    if not actual:
        return {
            "reward": 0.0,
            "correct": False,
            "expected": expected,
            "actual": "",
            "feedback": "No answer extracted from agent output.",
        }

    if mode == "llm" and (api_key or api_base):
        is_correct, feedback = _llm_verify(
            expected, actual, task.get("problem", ""),
            api_key=api_key, model=model, api_base=api_base,
        )
    else:
        is_correct = _answers_match(expected, actual)
        feedback = "exact match" if is_correct else (
            f"Expected: {_normalize_answer(expected)}, "
            f"Got: {_normalize_answer(actual)}"
        )

    return {
        "reward": 1.0 if is_correct else 0.0,
        "correct": is_correct,
        "expected": expected,
        "actual": actual,
        "feedback": feedback,
    }
