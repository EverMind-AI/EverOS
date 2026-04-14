"""
QA data loader for evaluation.

Loads QA data from JSON files with support for two question types:
- multiple_choice: Questions with options (A/B/C/D)
- open_ended: Questions with free-text answers

Supports two QA JSON formats:

Format 1 (eval_sub_topic format):
{
  "qars": [
    {
      "id": "F_SH_Top004_001",
      "Q": "question text",
      "A": "answer text",
      "task_id": "T001",  // ignored
      "options": {"A": "option A", "B": "option B", "C": "option C", "D": "option D"}  // null for open-ended
    }
  ]
}

Format 2 (legacy format):
{
  "questions": [
    {
      "question_id": "004_mc_001",
      "question": "question text",
      "options": ["A. option", ...],
      "correct_option": "A",
      "answer": "answer text"
    }
  ]
}
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from eval.src.core.data_models import QAItem
from eval.src.utils.logger import get_console


def load_qa(qa_path: str, limit: Optional[int] = None) -> List[QAItem]:
    """
    Load QA items from a JSON file.
    
    Auto-detects file format and question type:
    - Format 1: {"qars": [...]} - eval_sub_topic format
    - Format 2: {"questions": [...]} or [...] - legacy format
    
    Question types detected by "options" field:
    - If "options" exists and not null -> multiple_choice
    - Otherwise -> open_ended
    
    Args:
        qa_path: Path to the QA JSON file
        limit: Optional limit on number of questions to load
        
    Returns:
        List of QAItem objects
        
    Raises:
        FileNotFoundError: If QA file doesn't exist
        ValueError: If QA file format is invalid
    """
    path = Path(qa_path)
    if not path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Detect format and extract questions
    if isinstance(data, dict) and "qars" in data:
        # Format 1: eval_sub_topic format with "qars" key
        questions_raw = data["qars"]
        parser = _parse_qars_item
    elif isinstance(data, dict) and "questions" in data:
        # Format 2: legacy format with "questions" key
        questions_raw = data["questions"]
        parser = _parse_legacy_item
    elif isinstance(data, list):
        # Direct list of questions (legacy)
        questions_raw = data
        parser = _parse_legacy_item
    else:
        raise ValueError(
            f"Invalid QA file format. Expected 'qars' or 'questions' key. "
            f"Got keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )
    
    # Apply limit if specified
    if limit is not None:
        questions_raw = questions_raw[:limit]
    
    qa_items = []
    for idx, q in enumerate(questions_raw):
        qa_item = parser(q, idx)
        qa_items.append(qa_item)
    
    console = get_console()
    console.print(f"Loaded {len(qa_items)} QA items from {qa_path}")
    _print_qa_summary(qa_items)
    
    return qa_items


def _parse_qars_item(q: Dict[str, Any], idx: int) -> QAItem:
    """
    Parse a single QA item from eval_sub_topic format (qars).

    Format:
    {
        "id": "F_SH_Top004_001",
        "Q": "question text",
        "A": "answer text",
        "task_id": "T001",  // ignored
        "options": {"A": "option A", "B": "option B", ...} or null,
    }

    Args:
        q: Raw question dict
        idx: Index for fallback question_id

    Returns:
        QAItem object
    """
    # Required fields (using qars format keys)
    question = q.get("Q")
    if question is None:
        raise ValueError(f"Question {idx}: missing 'Q' field")

    answer = q.get("A")
    if answer is None:
        raise ValueError(f"Question {idx}: missing 'A' field")

    # Question ID
    question_id = q.get("id", f"q_{idx:04d}")

    # Detect question type based on options field
    # options can be dict {A: ..., B: ..., C: ..., D: ...} or null
    options_raw = q.get("options")

    if options_raw and isinstance(options_raw, dict) and len(options_raw) > 0:
        # Multiple choice: options is a dict with A/B/C/D keys
        question_type = "multiple_choice"
        # Convert dict to list format: ["A. option text", "B. option text", ...]
        options = []
        for key in sorted(options_raw.keys()):
            options.append(f"{key}. {options_raw[key]}")
        # Answer is the correct option letter (A/B/C/D)
        correct_option = answer.strip().upper() if answer else ""
    else:
        # Open-ended: options is null or missing
        question_type = "open_ended"
        options = None
        correct_option = None

    return QAItem(
        question_id=question_id,
        question=question,
        answer=answer,
        question_type=question_type,
        options=options,
        correct_option=correct_option,
        metadata={},
    )


def _parse_legacy_item(q: Dict[str, Any], idx: int) -> QAItem:
    """
    Parse a single QA item from legacy format.
    
    Format:
    {
        "question_id": "004_mc_001",
        "question": "question text",
        "options": ["A. option", ...],  // optional
        "correct_option": "A",  // optional
        "answer": "answer text"
    }
    
    Args:
        q: Raw question dict
        idx: Index for default question_id generation
        
    Returns:
        QAItem object
    """
    # Required fields
    question = q.get("question")
    if question is None:
        raise ValueError(f"Question {idx}: missing 'question' field")

    answer = q.get("answer")
    if answer is None:
        raise ValueError(f"Question {idx}: missing 'answer' field")
    
    # Question ID (auto-generate if missing)
    question_id = q.get("question_id", f"q_{idx:04d}")
    
    # Detect question type based on options field
    options = q.get("options")
    if options and isinstance(options, list) and len(options) > 0:
        question_type = "multiple_choice"
        correct_option = q.get("correct_option")
        
        # Try to infer correct_option from answer if not provided
        if not correct_option and answer:
            correct_option = _infer_correct_option(answer, options)
    else:
        question_type = "open_ended"
        options = None
        correct_option = None
    
    # Collect any extra fields as metadata
    known_fields = {"question_id", "question", "answer", "options", "correct_option"}
    metadata = {k: v for k, v in q.items() if k not in known_fields}
    
    return QAItem(
        question_id=question_id,
        question=question,
        answer=answer,
        question_type=question_type,
        options=options,
        correct_option=correct_option,
        metadata=metadata,
    )


def _infer_correct_option(answer: str, options: List[str]) -> str:
    """
    Try to infer the correct option letter from the answer text.
    
    Handles formats like:
    - "A" or "B" (direct letter)
    - "A. Option text" (letter with option)
    - "Option text" (match against option content)
    
    Args:
        answer: The answer string
        options: List of option strings
        
    Returns:
        Inferred option letter (A/B/C/D) or empty string if not found
    """
    answer_stripped = answer.strip().upper()
    
    # Direct letter answer
    if len(answer_stripped) == 1 and answer_stripped in "ABCD":
        return answer_stripped
    
    # Check if answer starts with letter
    if len(answer_stripped) >= 1 and answer_stripped[0] in "ABCD":
        if len(answer_stripped) == 1 or answer_stripped[1] in ".):":
            return answer_stripped[0]
    
    # Try to match answer content against options
    answer_lower = answer.lower().strip()
    for i, opt in enumerate(options):
        # Remove option prefix like "A. " or "A) "
        opt_content = opt.strip()
        if len(opt_content) >= 2 and opt_content[0].upper() in "ABCD" and opt_content[1] in ".):":
            opt_content = opt_content[2:].strip()
        
        if opt_content.lower() == answer_lower:
            return chr(ord("A") + i)
    
    return ""


def _print_qa_summary(qa_items: List[QAItem]) -> None:
    """Print summary of loaded QA items."""
    console = get_console()
    mc_count = sum(1 for q in qa_items if q.question_type == "multiple_choice")
    oe_count = sum(1 for q in qa_items if q.question_type == "open_ended")

    console.print(f"   Multiple choice: {mc_count}")
    console.print(f"   Open-ended: {oe_count}")

