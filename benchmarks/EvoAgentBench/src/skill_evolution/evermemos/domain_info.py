"""Per-domain metadata for EverMemOS integration.

New domain? Add one entry to BENCHMARK_DESCRIPTORS — everything else adapts.
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DomainInfo:
    # --- Query extraction (for EverMemOS search) ---
    query_field: str | None = "problem"
    query_extractor: Callable | None = None  # overrides query_field

    # --- Skill injection ---
    # "prompt_append": monkey-patch build_prompt (default, works for all)
    # "task_field": set task[skill_field] and let domain handle it
    skill_injection: str = "prompt_append"
    skill_field: str = "skill_text"


BENCHMARK_DESCRIPTORS: dict[str, DomainInfo] = {
    "information_retrieval": DomainInfo(query_field="problem"),
    "software_engineering":        DomainInfo(query_field="problem_statement"),
    "knowledge_work":          DomainInfo(query_field="prompt"),
    "reasoning":        DomainInfo(
        query_field="problem",
        skill_injection="task_field",
        skill_field="skill_text",
    ),
}


def get_task_query(domain_name: str, task: dict, bench_cfg: dict = None) -> str:
    """Extract search query text from a task dict, using the registry."""
    info = BENCHMARK_DESCRIPTORS.get(domain_name, DomainInfo())

    if info.query_extractor:
        return info.query_extractor(task, bench_cfg)

    if info.query_field and info.query_field in task:
        return str(task[info.query_field])

    # Fallback: try common field names
    for key in ("problem", "query", "question", "prompt", "description"):
        if key in task:
            return str(task[key])
    return str(task.get("name", ""))
