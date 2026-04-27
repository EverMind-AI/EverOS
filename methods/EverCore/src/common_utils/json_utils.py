"""
JSON parsing utilities

Provides robust JSON extraction from LLM responses that may contain
markdown code blocks or extra text around JSON objects.
"""

import json
from typing import Any, Dict, Optional


def extract_outermost_json(text: str) -> Optional[str]:
    """Extract the outermost JSON object from text using bracket counting.

    Handles escaped characters and strings correctly.

    Args:
        text: Raw text potentially containing a JSON object

    Returns:
        The JSON substring, or None if no valid object found
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from an LLM response, tolerating markdown code blocks and surrounding text.

    Tries in order:
    1. Extract from ```json ... ``` code block
    2. Extract outermost { ... } via bracket counting
    3. Parse the entire response as JSON

    Args:
        response: Raw LLM response string

    Returns:
        Parsed dict, or None if parsing fails
    """
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end > start:
            try:
                return json.loads(response[start:end].strip())
            except json.JSONDecodeError:
                pass

    json_str = extract_outermost_json(response)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None
