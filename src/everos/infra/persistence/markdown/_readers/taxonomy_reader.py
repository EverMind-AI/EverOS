"""Read and auto-generate ``.taxonomy.md`` — the knowledge category taxonomy.

The taxonomy file lives at ``knowledge_dir/.taxonomy.md``. It uses YAML
frontmatter with a ``categories`` list, each entry having ``id`` and
``description`` — matching ``everalgo.types.CategorySpec`` exactly.

``ensure_taxonomy(knowledge_dir)`` creates the file from
``DEFAULT_TAXONOMY`` if it does not exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import yaml
from everalgo.types import CategorySpec

TAXONOMY_FILENAME = ".taxonomy.md"


DEFAULT_TAXONOMY: list[dict[str, str]] = [
    {
        "id": "Technology",
        "description": (
            "Computer science, software engineering, AI/ML, semiconductors, "
            "cloud computing, cybersecurity, and general IT infrastructure."
        ),
    },
    {
        "id": "Science",
        "description": (
            "Natural sciences including physics, chemistry, biology, earth sciences, "
            "astronomy, and interdisciplinary research."
        ),
    },
    {
        "id": "Medical",
        "description": (
            "Clinical medicine, disease diagnosis, drug specifications, "
            "medical devices, public health, nursing, and healthcare administration."
        ),
    },
    {
        "id": "Finance",
        "description": (
            "Securities, banking, insurance, accounting, taxation, corporate finance, "
            "macroeconomics, and fintech."
        ),
    },
    {
        "id": "Legal",
        "description": (
            "Laws, regulations, contracts, compliance, intellectual property, "
            "litigation, and legal procedures."
        ),
    },
    {
        "id": "Education",
        "description": (
            "Teaching methods, curriculum design, academic research, "
            "student assessment, e-learning, and educational policy."
        ),
    },
    {
        "id": "Business",
        "description": (
            "Corporate strategy, marketing, sales, operations management, "
            "supply chain, HR, and entrepreneurship."
        ),
    },
    {
        "id": "Engineering",
        "description": (
            "Mechanical, civil, electrical, chemical, and industrial engineering — "
            "design, manufacturing, and construction."
        ),
    },
    {
        "id": "Arts",
        "description": (
            "Visual arts, music, literature, film, theater, design, "
            "and cultural studies."
        ),
    },
    {
        "id": "Sports",
        "description": (
            "Athletic training, sports events, fitness, sports science, "
            "and sports industry management."
        ),
    },
    {
        "id": "Travel",
        "description": (
            "Tourism, hospitality, travel guides, destination reviews, "
            "and transportation logistics."
        ),
    },
    {
        "id": "Food",
        "description": (
            "Culinary arts, nutrition science, food safety, restaurant management, "
            "and food industry."
        ),
    },
    {
        "id": "Environment",
        "description": (
            "Climate change, ecology, pollution control, renewable energy, "
            "conservation, and sustainability."
        ),
    },
    {
        "id": "Politics",
        "description": (
            "Government policy, international relations, elections, "
            "public administration, and geopolitics."
        ),
    },
    {
        "id": "History",
        "description": (
            "Historical events, civilizations, archaeology, historical analysis, "
            "and historiography."
        ),
    },
    {
        "id": "Psychology",
        "description": (
            "Cognitive science, behavioral psychology, clinical psychology, "
            "mental health, and neuroscience."
        ),
    },
    {
        "id": "Agriculture",
        "description": (
            "Farming techniques, crop science, animal husbandry, agribusiness, "
            "and food production systems."
        ),
    },
    {
        "id": "RealEstate",
        "description": (
            "Property development, real estate investment, urban planning, "
            "architecture, and housing policy."
        ),
    },
    {
        "id": "Media",
        "description": (
            "Journalism, broadcasting, social media, public relations, "
            "advertising, and communications."
        ),
    },
    {
        "id": "Others",
        "description": (
            "Documents that do not clearly fit any of the above categories."
        ),
    },
]


async def parse_taxonomy(path: anyio.Path | Path) -> list[CategorySpec]:
    """Parse ``.taxonomy.md`` and return the category list.

    Args:
        path: Path to the ``.taxonomy.md`` file.

    Returns:
        Parsed category list, or an empty list when the file does not
        exist or has no categories.
    """
    apath = anyio.Path(path) if not isinstance(path, anyio.Path) else path
    if not await apath.exists():
        return []
    text = await apath.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) < 3:
        return []
    data: dict[str, Any] = yaml.safe_load(parts[1]) or {}
    raw_categories = data.get("categories") or []
    return [
        CategorySpec(id=c["id"], description=c.get("description", ""))
        for c in raw_categories
        if isinstance(c, dict) and "id" in c
    ]


async def ensure_taxonomy(knowledge_dir: anyio.Path | Path) -> Path:
    """Create ``.taxonomy.md`` from defaults if it does not exist.

    Args:
        knowledge_dir: Knowledge directory where ``.taxonomy.md`` lives.

    Returns:
        Path to the taxonomy file (whether created or pre-existing).
    """
    adir = (
        knowledge_dir
        if isinstance(knowledge_dir, anyio.Path)
        else anyio.Path(knowledge_dir)
    )
    p = adir / TAXONOMY_FILENAME
    if await p.exists():
        return Path(p)
    await adir.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.dump(
        {"kind": "knowledge_taxonomy", "categories": DEFAULT_TAXONOMY},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    await p.write_text(f"---\n{frontmatter}---\n", encoding="utf-8")
    return Path(p)
