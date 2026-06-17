"""Validate contributor-facing Markdown and use-case documentation."""

from __future__ import annotations

import re
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", ".venv", ".uv-cache"}
USE_CASE_TABLES = {
    Path("README.md"): "## Use Cases",
    Path("use-cases/README.md"): "## Use Cases",
}
PRIMARY_LINK_RE = re.compile(
    r"^\[(?:Code|Plugin|Live Demo|Learn more)\]\(([^)]+)\)",
    flags=re.M,
)
ENV_EXAMPLE = Path(".env.example")
ENV_TEMPLATE = Path("src/everos/templates/env.template")


def _markdown_files() -> list[Path]:
    return sorted(
        path
        for path in Path(".").rglob("*.md")
        if not any(part in SKIP_DIRS for part in path.parts)
    )


def _check_active_relative_links() -> list[str]:
    missing: list[str] = []
    root = Path.cwd().resolve()
    for path in _markdown_files():
        active = re.sub(r"<!--.*?-->", "", path.read_text(), flags=re.S)
        for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", active):
            link = raw.split("#", 1)[0]
            if not link or link.startswith(("http://", "https://", "mailto:")):
                continue

            target = (path.parent / link).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                missing.append(f"{path}: {raw} -> outside repository")
                continue

            if not target.exists():
                missing.append(f"{path}: {raw} -> missing")
    return missing


def _check_use_case_banner_links() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    for path, heading in USE_CASE_TABLES.items():
        text = path.read_text()
        start = text.find(heading)
        if start == -1:
            failures.append(f"{path}: missing {heading}")
            continue

        table_start = text.find("<table>", start)
        table_end = text.find("</table>", table_start)
        if table_start == -1 or table_end == -1:
            failures.append(f"{path}: missing use-case table")
            continue

        table = text[table_start:table_end]
        cells = re.findall(r"<td[^>]*>(.*?)</td>", table, flags=re.S)
        for index, cell in enumerate(cells, start=1):
            title_match = re.search(r"####\s+(.+)", cell)
            title = title_match.group(1).strip() if title_match else f"use case {index}"
            banner_match = re.search(r"\[!\[[^\]]*\]\([^)]+\)\]\(([^)]+)\)", cell)
            primary_match = PRIMARY_LINK_RE.search(cell)

            if not banner_match and primary_match:
                warnings.append(f"{path}: {title}: primary link has no linked banner")
            elif banner_match and not primary_match:
                failures.append(f"{path}: {title}: missing primary link")
            elif (
                banner_match
                and primary_match
                and banner_match.group(1) != primary_match.group(1)
            ):
                failures.append(
                    f"{path}: {title}: banner link {banner_match.group(1)} "
                    f"does not match primary link {primary_match.group(1)}"
                )

    return failures, warnings


def _check_env_example_matches_template() -> list[str]:
    if not ENV_EXAMPLE.exists():
        return [f"{ENV_EXAMPLE}: missing"]

    if ENV_EXAMPLE.read_text() != ENV_TEMPLATE.read_text():
        return [f"{ENV_EXAMPLE}: must match {ENV_TEMPLATE}"]

    return []


def main() -> int:
    failures = _check_active_relative_links()
    use_case_failures, warnings = _check_use_case_banner_links()
    failures.extend(use_case_failures)
    failures.extend(_check_env_example_matches_template())

    if warnings:
        print("\n".join(f"warning: {warning}" for warning in warnings))

    if failures:
        print("\n".join(failures))
        return 1

    print("Documentation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
