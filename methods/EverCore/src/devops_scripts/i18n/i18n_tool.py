"""
I18N Tool - Chinese to English translation and review tool for Python files.

This unified tool provides multiple functions for internationalization:
1. translate: Translate Chinese comments/logs to English in Python files
2. check: Check for remaining Chinese content in Python files
3. review: Review git commits to verify translation changes
4. hook: Pre-commit hook to check for Chinese characters in staged files/commit msg

CRITICAL RULES FOR TRANSLATION:
    1. DO NOT modify any code logic - this is the most important rule
    2. Only translate Chinese comments and Chinese log messages
    3. Never change variable names, function names, class names, etc.
    4. Never change any code structure or behavior
    5. Violations of these rules are STRICTLY FORBIDDEN

Usage:
    # Translate commands
    python -m src.devops_scripts.i18n.i18n_tool translate
    python -m src/devops_scripts.i18n.i18n_tool translate --dry-run
    python -m src/devops_scripts.i18n.i18n_tool translate --directory tests
    python -m src/devops_scripts.i18n.i18n_tool translate --directory src tests

    # Check commands
    python -m src/devops_scripts.i18n.i18n_tool check
    python -m src/devops_scripts.i18n.i18n_tool check --directory tests

    # Review commands
    python -m src/devops_scripts.i18n.i18n_tool review
    python -m src/devops_scripts.i18n.i18n_tool review --commit abc123
    python -m src/devops_scripts.i18n.i18n_tool review --commit HEAD~3..HEAD
    python -m src/devops_scripts.i18n.i18n_tool review --reset  # Clear progress and start fresh

    # Hook commands (for pre-commit)
    python -m src/devops_scripts.i18n.i18n_tool hook file1.py file2.py
    python -m src/devops_scripts.i18n.i18n_tool hook --commit-msg .git/COMMIT_EDITMSG
"""

from __future__ import annotations

import os
import sys
import re
import asyncio
import json
import subprocess
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum

# ==============================================================================
# Path Configuration (no project dependencies needed)
# ==============================================================================

# Add src to path for imports (only when needed by other commands)
SRC_DIR = Path(__file__).parent.parent.parent
PROJECT_DIR = SRC_DIR.parent


def _setup_project_imports():
    """Setup project imports when needed (lazy loading)."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


# ==============================================================================
# Common Configuration
# ==============================================================================

# Progress files to track which files have been processed
TRANSLATION_PROGRESS_FILE = Path(__file__).parent / ".translation_progress.json"
REVIEW_PROGRESS_FILE = Path(__file__).parent / ".review_progress.json"
# Maximum file size to process (in bytes) - about 100KB
MAX_FILE_SIZE = 100 * 1024

# Directories to skip (relative to SRC_DIR)
SKIP_DIRECTORIES = ["memory_layer/prompts"]

# Files to skip (relative to SRC_DIR)
SKIP_FILES = [
    # "memory_layer/memory_extractor/profile_memory/conversation.py",
    # "common_utils/text_utils.py",
    # "core/oxm/es/analyzer.py",
    # "memory_layer/memory_extractor/profile_memory/types.py",
    # "memory_layer/memory_extractor/profile_memory/value_helpers.py",
    # This tool itself contains Chinese examples in prompts, skip it
    "devops_scripts/i18n/i18n_tool.py"
]

# ==============================================================================
# Hook Configuration (for pre-commit)
# ==============================================================================

# File patterns to skip in hook check (glob patterns, relative to project root)
# All files are checked by default, add patterns here to skip
HOOK_SKIP_PATTERNS = [
    # i18n tool itself contains CJK examples in prompts
    "src/devops_scripts/i18n/i18n_tool.py",
    # sensitive info tool contains CJK examples in AI prompts
    "src/devops_scripts/sensitive_info/sensitive_info_tool.py",
    # Prompt files may contain CJK
    "src/memory_layer/prompts/*",
    # Test files that specifically test i18n handling
    "**/test_*i18n*.py",
    # Documentation files (may contain CJK for localization)
    "*.md",
    "*.rst",
    # Lock files and generated files
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    # Data files
    "*.json",
    "*.yaml",
    "*.yml",
    "*.toml",
    "*.xml",
    "*.csv",
    # Binary and media files
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
    # Compiled files
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.exe",
    # Git files
    ".git/*",
    ".gitignore",
    ".gitattributes",
]

# Environment variable to skip hook checks
HOOK_SKIP_ENV_VAR = "SKIP_I18N_CHECK"

# Keywords in commit message to skip check
HOOK_SKIP_COMMIT_MSG_KEYWORDS = [
    "skip-i18n-check",
    "#skip-i18n-check",
    "[skip-i18n]",
    "[no-i18n-check]",
]

# Inline comment to skip i18n check for a specific line
# Usage: code_with_chinese()  #skip-i18n-check
HOOK_SKIP_LINE_COMMENT = "#skip-i18n-check"

# File-level skip marker - add this at the top of a file to skip the entire file
# Usage: Add "# skip-i18n-file" or "#skip-i18n-file" in the first 10 lines
HOOK_SKIP_FILE_MARKERS = [
    "#skip-i18n-file",
    "# skip-i18n-file",
    "#  skip-i18n-file",
    "# -*- skip-i18n-file -*-",
]

# CJK Unicode ranges (Chinese, Japanese, Korean)
CJK_PATTERN = re.compile(
    r'['
    r'\u4e00-\u9fff'  # CJK Unified Ideographs (Chinese)
    r'\u3040-\u309f'  # Hiragana (Japanese)
    r'\u30a0-\u30ff'  # Katakana (Japanese)
    r'\uac00-\ud7af'  # Hangul Syllables (Korean)
    r'\u3400-\u4dbf'  # CJK Unified Ideographs Extension A
    r'\uf900-\ufaff'  # CJK Compatibility Ideographs
    r']'
)


# ==============================================================================
# LLM Prompts
# ==============================================================================

TRANSLATION_PROMPT = '''You are a translation assistant. Your task is to translate Chinese comments and Chinese log messages in Python code to English.

**CRITICAL RULES - MUST FOLLOW:**
1. **ABSOLUTELY DO NOT modify any code logic** - This is the most important rule. Violations are STRICTLY FORBIDDEN.
2. **ONLY translate Chinese text** in:
   - Single-line comments (# ...)
   - Multi-line docstrings (""" ... """ or \'\'\' ... \'\'\')
   - String literals used in logging (logger.info(), logger.debug(), logger.warning(), logger.error(), print(), etc.)
   - f-string literals used in logging
3. **DO NOT change:**
   - Variable names, function names, class names
   - Code structure, indentation, line breaks
   - Any Python syntax or operators
   - Non-Chinese text
   - Import statements
   - Type hints
   - Any actual code behavior
4. Keep the original formatting and indentation exactly as is
5. If there is no Chinese text to translate, return the code unchanged
6. Return ONLY the translated code, no explanations

**Example translations:**
- `# 初始化配置` → `# Initialize configuration`
- `logger.info("开始处理数据")` → `logger.info("Start processing data")`
- `"""这是一个测试函数"""` → `"""This is a test function"""`
- `print(f"处理完成，共 {{count}} 条")` → `print(f"Processing completed, total {{count}} items")`

Now translate the following Python code:

```python
{code}
```

Return the translated Python code:'''

REVIEW_PROMPT = '''You are a code review assistant. Your task is to analyze a git diff and determine if the changes are ONLY translation-related (translating Chinese comments/logs to English) or if there are actual code logic changes.

**Your task:**
Analyze the following git diff and classify it as one of:
1. **SAFE** - Changes are purely translation-related:
   - Chinese comments translated to English (# 中文注释 → # English comment)
   - Chinese log messages translated to English (logger.info("中文") → logger.info("English"))
   - Chinese docstrings translated to English
   - No actual code logic changes

2. **NEEDS_REVIEW** - Changes may include code logic modifications:
   - Variable names, function names, or class names changed
   - Code structure modified
   - Import statements added/removed/changed
   - Logic conditions or return values changed
   - Exception handling modified
   - New code added or existing code removed (beyond comments/logs)
   - Type hints changed
   - Default parameter values changed
   - Any behavioral changes

**Important:**
- Whitespace-only changes (indentation, blank lines) are SAFE
- Formatting changes that don't affect behavior are SAFE
- If you see ANY potential code logic change, classify as NEEDS_REVIEW
- When in doubt, classify as NEEDS_REVIEW

**Response format (MUST follow exactly):**
First line: SAFE or NEEDS_REVIEW
Second line onwards: Brief explanation of your reasoning (max 2-3 sentences)

**Git diff to analyze:**
```diff
{diff}
```

**Your analysis:**'''


# ==============================================================================
# Review Result Types
# ==============================================================================


class ReviewResult(Enum):
    """Review result status."""

    SAFE = "safe"
    NEEDS_REVIEW = "needs_review"
    ERROR = "error"


@dataclass
class FileReviewResult:
    """Result of reviewing a single file."""

    file_path: str
    result: ReviewResult
    reason: str
    diff_summary: str = ""


# ==============================================================================
# Common Utilities
# ==============================================================================


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))


def _load_env_and_get_llm_provider():
    """Load environment and create LLM provider (lazy loading)."""
    _setup_project_imports()

    from dotenv import load_dotenv
    from memory_layer.llm import OpenAIProvider

    env_file_path = PROJECT_DIR / ".env"
    if env_file_path.exists():
        load_dotenv(env_file_path)
        print(f"Loaded environment from {env_file_path}")

    return OpenAIProvider(
        model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.1,
    )


def resolve_directories(dir_names: list[str] | None) -> list[Path]:
    """Resolve directory names to absolute paths."""
    if not dir_names:
        return [SRC_DIR]

    directories = []
    for dir_name in dir_names:
        dir_path = Path(dir_name)
        if not dir_path.is_absolute():
            dir_path = PROJECT_DIR / dir_name
        directories.append(dir_path)
    return directories


def print_header(title: str):
    """Print a section header."""
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def print_summary_header():
    """Print summary section header."""
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)


# ==============================================================================
# File Operations
# ==============================================================================


def should_skip_directory(dir_path: Path, src_dir: Path) -> bool:
    """Check if a directory should be skipped based on SKIP_DIRECTORIES config."""
    try:
        rel_path = dir_path.relative_to(src_dir)
        rel_path_str = str(rel_path).replace('\\', '/')
        for skip_dir in SKIP_DIRECTORIES:
            if rel_path_str == skip_dir or rel_path_str.startswith(skip_dir + '/'):
                return True
    except ValueError:
        pass
    return False


def should_skip_file(file_path: Path, src_dir: Path) -> bool:
    """Check if a file should be skipped based on SKIP_FILES config."""
    try:
        rel_path = file_path.relative_to(src_dir)
        rel_path_str = str(rel_path).replace('\\', '/')
        return rel_path_str in SKIP_FILES
    except ValueError:
        pass
    return False


def get_python_files(target_dir: Path) -> list[Path]:
    """Get all Python files under the target directory."""
    python_files = []
    skipped_dirs = []
    skipped_files = []

    for root, dirs, files in os.walk(target_dir):
        root_path = Path(root)

        # Skip __pycache__ and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

        # Skip configured directories (only for src directory)
        if target_dir == SRC_DIR:
            dirs_to_remove = []
            for d in dirs:
                dir_path = root_path / d
                if should_skip_directory(dir_path, SRC_DIR):
                    dirs_to_remove.append(d)
                    skipped_dirs.append(dir_path)
            for d in dirs_to_remove:
                dirs.remove(d)

        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                if target_dir == SRC_DIR and should_skip_file(file_path, SRC_DIR):
                    skipped_files.append(file_path)
                else:
                    python_files.append(file_path)

    if skipped_dirs:
        print(f"Skipped directories: {[str(d) for d in skipped_dirs]}")
    if skipped_files:
        print(f"Skipped files: {[str(f) for f in skipped_files]}")

    return python_files


def get_python_files_from_directories(directories: list[Path]) -> list[Path]:
    """Get all Python files from multiple directories."""
    all_files = []
    for target_dir in directories:
        if not target_dir.exists():
            print(f"Warning: Directory {target_dir} does not exist, skipping")
            continue
        print(f"Scanning directory: {target_dir}")
        files = get_python_files(target_dir)
        all_files.extend(files)
        print(f"  Found {len(files)} Python files")
    return all_files


# ==============================================================================
# Progress Tracking - Translation
# ==============================================================================


def load_translation_progress() -> dict:
    """Load translation progress from file."""
    if TRANSLATION_PROGRESS_FILE.exists():
        try:
            with open(TRANSLATION_PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"processed": [], "errors": []}
    return {"processed": [], "errors": []}


def save_translation_progress(progress: dict):
    """Save translation progress to file."""
    with open(TRANSLATION_PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def clear_translation_progress():
    """Clear translation progress file."""
    if TRANSLATION_PROGRESS_FILE.exists():
        TRANSLATION_PROGRESS_FILE.unlink()


# ==============================================================================
# Progress Tracking - Review
# ==============================================================================


def load_review_progress() -> dict:
    """Load review progress from file."""
    if REVIEW_PROGRESS_FILE.exists():
        try:
            with open(REVIEW_PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"commit_range": "", "safe": [], "needs_review": [], "errors": []}
    return {"commit_range": "", "safe": [], "needs_review": [], "errors": []}


def save_review_progress(progress: dict):
    """Save review progress to file."""
    with open(REVIEW_PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def clear_review_progress():
    """Clear review progress file."""
    if REVIEW_PROGRESS_FILE.exists():
        REVIEW_PROGRESS_FILE.unlink()


# ==============================================================================
# Git Operations
# ==============================================================================


def run_git_command(args: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    """Run a git command and return success status and output."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd or PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Git command timed out"
    except Exception as e:
        return False, str(e)


def get_changed_files_from_git(commit_ref: str) -> tuple[bool, list[str]]:
    """Get list of changed Python files in a commit or commit range.

    Args:
        commit_ref: A single commit (e.g., "HEAD", "abc123") or a range (e.g., "HEAD~3..HEAD")
    """
    if ".." in commit_ref:
        # It's a range, use git diff
        success, output = run_git_command(
            ["diff", "--name-only", "--diff-filter=ACMR", commit_ref, "--", "*.py"]
        )
    else:
        # Single commit, use git show to get files changed in that specific commit
        success, output = run_git_command(
            ["show", "--name-only", "--format=", commit_ref, "--", "*.py"]
        )
    if not success:
        return False, [output]
    files = [f.strip() for f in output.split("\n") if f.strip()]
    return True, files


def get_file_diff(
    commit_ref: str, file_path: str, context_lines: int = 3
) -> tuple[bool, str]:
    """Get the diff for a specific file in a commit or commit range.

    Args:
        commit_ref: A single commit or a range
        file_path: Path to the file
        context_lines: Number of context lines around changes (default: 3, use 0 for minimal)
    """
    if ".." in commit_ref:
        # It's a range, use git diff
        success, output = run_git_command(
            ["diff", f"-U{context_lines}", commit_ref, "--", file_path]
        )
    else:
        # Single commit, use git show to get the diff for that specific commit
        success, output = run_git_command(
            ["show", f"-U{context_lines}", "--format=", commit_ref, "--", file_path]
        )
    return success, output


def get_commit_info(commit_ref: str = "HEAD") -> tuple[bool, dict]:
    """Get information about a commit."""
    success, hash_output = run_git_command(["rev-parse", "--short", commit_ref])
    if not success:
        return False, {"error": hash_output}

    success, message = run_git_command(["log", "-1", "--format=%s", commit_ref])
    message = message if success else "Unknown"

    success, author = run_git_command(["log", "-1", "--format=%an <%ae>", commit_ref])
    author = author if success else "Unknown"

    success, date = run_git_command(["log", "-1", "--format=%ci", commit_ref])
    date = date if success else "Unknown"

    return True, {
        "hash": hash_output,
        "message": message,
        "author": author,
        "date": date,
    }


# ==============================================================================
# Translation Functions
# ==============================================================================


def filter_files_with_chinese(
    python_files: list[Path], progress: dict
) -> tuple[list[Path], int, int]:
    """Pre-filter files to only include those with Chinese characters."""
    files_to_process = []
    skipped_no_chinese = 0
    skipped_already_done = 0

    print("Pre-scanning files for Chinese content...")
    for file_path in python_files:
        file_str = str(file_path)

        try:
            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                if file_str not in progress.get("processed", []):
                    files_to_process.append(file_path)
                else:
                    skipped_already_done += 1
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if contains_chinese(content):
                if file_str in progress.get("processed", []):
                    progress["processed"].remove(file_str)
                    print(f"  [RE-PROCESS] {file_path} - Still has Chinese content")
                files_to_process.append(file_path)
            else:
                skipped_no_chinese += 1
                if file_str not in progress.get("processed", []):
                    progress["processed"].append(file_str)
                else:
                    skipped_already_done += 1
        except Exception as e:
            print(f"  Warning: Could not pre-scan {file_path}: {e}")
            files_to_process.append(file_path)

    save_translation_progress(progress)
    print(
        f"Pre-scan complete: {len(files_to_process)} files with Chinese to translate, "
        f"{skipped_no_chinese} without Chinese (skipped), {skipped_already_done} already done"
    )

    return files_to_process, skipped_no_chinese, skipped_already_done


async def translate_file(
    provider: OpenAIProvider,
    file_path: Path,
    semaphore: asyncio.Semaphore,
    progress: dict,
    progress_lock: asyncio.Lock,
    dry_run: bool = False,
    index: int = 0,
    total: int = 0,
) -> tuple[Path, bool, Optional[str]]:
    """Translate a single Python file."""
    file_str = str(file_path)
    progress_prefix = f"[{index}/{total}]" if total > 0 else ""

    if file_str in progress.get("processed", []):
        print(f"{progress_prefix} [ALREADY-DONE] {file_path}")
        return (file_path, True, None)

    async with semaphore:
        try:
            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                print(
                    f"{progress_prefix} [SKIP-LARGE] {file_path} - File too large ({file_size/1024:.1f}KB)"
                )
                async with progress_lock:
                    progress["processed"].append(file_str)
                    save_translation_progress(progress)
                return (file_path, True, f"Skipped: file too large")

            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            if not contains_chinese(original_content):
                print(f"{progress_prefix} [SKIP] {file_path} - No Chinese text found")
                async with progress_lock:
                    progress["processed"].append(file_str)
                    save_translation_progress(progress)
                return (file_path, True, None)

            print(
                f"{progress_prefix} [TRANSLATING] {file_path} ({file_size/1024:.1f}KB)"
            )

            prompt = TRANSLATION_PROMPT.format(code=original_content)
            translated_content = await provider.generate(prompt, temperature=0.1)

            # Clean up response
            translated_content = translated_content.strip()
            if translated_content.startswith('```python'):
                translated_content = translated_content[9:]
            if translated_content.startswith('```'):
                translated_content = translated_content[3:]
            if translated_content.endswith('```'):
                translated_content = translated_content[:-3]
            translated_content = translated_content.strip()

            if (
                not translated_content
                or len(translated_content) < len(original_content) * 0.5
            ):
                error_msg = "Translation result seems too short or empty"
                async with progress_lock:
                    progress["errors"].append({"file": file_str, "error": error_msg})
                    save_translation_progress(progress)
                return (file_path, False, error_msg)

            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(translated_content)
                print(f"{progress_prefix} [DONE] {file_path}")
            else:
                print(f"{progress_prefix} [DRY-RUN] {file_path} - Would translate")

            async with progress_lock:
                progress["processed"].append(file_str)
                save_translation_progress(progress)
            return (file_path, True, None)

        except Exception as e:
            error_msg = str(e)
            print(f"{progress_prefix} [ERROR] {file_path}: {error_msg}")
            async with progress_lock:
                progress["errors"].append({"file": file_str, "error": error_msg})
                save_translation_progress(progress)
            return (file_path, False, error_msg)


# ==============================================================================
# Review Functions
# ==============================================================================


# Maximum diff size for LLM analysis (200KB should be fine for most models)
MAX_DIFF_SIZE = 200 * 1024


async def review_file_diff(
    provider: OpenAIProvider,
    file_path: str,
    diff: str,
    semaphore: asyncio.Semaphore,
    progress: dict,
    progress_lock: asyncio.Lock,
    verbose: bool = False,
    index: int = 0,
    total: int = 0,
) -> FileReviewResult:
    """Review a single file diff using LLM."""
    progress_prefix = f"[{index}/{total}]" if total > 0 else ""

    async with semaphore:
        try:
            if len(diff) > MAX_DIFF_SIZE:
                result = FileReviewResult(
                    file_path=file_path,
                    result=ReviewResult.NEEDS_REVIEW,
                    reason=f"Diff too large for automated analysis ({len(diff)/1024:.1f}KB > {MAX_DIFF_SIZE/1024:.0f}KB limit)",
                    diff_summary=f"Diff size: {len(diff) / 1024:.1f}KB",
                )
                async with progress_lock:
                    progress["needs_review"].append(
                        {"file": file_path, "reason": result.reason}
                    )
                    save_review_progress(progress)
                print(
                    f"{progress_prefix} [NEEDS-REVIEW] {file_path} - Diff too large ({len(diff)/1024:.1f}KB)"
                )
                return result

            if not diff.strip():
                result = FileReviewResult(
                    file_path=file_path,
                    result=ReviewResult.SAFE,
                    reason="No changes in diff",
                    diff_summary="Empty diff",
                )
                async with progress_lock:
                    progress["safe"].append(file_path)
                    save_review_progress(progress)
                print(f"{progress_prefix} [SAFE] {file_path} - Empty diff")
                return result

            if verbose:
                print(f"{progress_prefix} [ANALYZING] {file_path}")
            else:
                print(f"{progress_prefix} [ANALYZING] {file_path}")

            prompt = REVIEW_PROMPT.format(diff=diff)
            response = await provider.generate(prompt, temperature=0.1)
            response = response.strip()

            lines = response.split("\n", 1)
            first_line = lines[0].strip().upper()
            reason = lines[1].strip() if len(lines) > 1 else "No explanation provided"

            if "SAFE" in first_line and "NEEDS" not in first_line:
                review_result = ReviewResult.SAFE
            else:
                review_result = ReviewResult.NEEDS_REVIEW
                if "SAFE" not in first_line and "REVIEW" not in first_line:
                    reason = f"Unclear response: {first_line}. {reason}"

            diff_lines = diff.split("\n")
            diff_summary = "\n".join(diff_lines[:10])
            if len(diff_lines) > 10:
                diff_summary += f"\n... ({len(diff_lines) - 10} more lines)"

            result = FileReviewResult(
                file_path=file_path,
                result=review_result,
                reason=reason,
                diff_summary=diff_summary,
            )

            # Save progress
            async with progress_lock:
                if review_result == ReviewResult.SAFE:
                    progress["safe"].append(file_path)
                    print(f"{progress_prefix} [SAFE] {file_path}")
                else:
                    progress["needs_review"].append(
                        {"file": file_path, "reason": reason}
                    )
                    print(f"{progress_prefix} [NEEDS-REVIEW] {file_path}")
                save_review_progress(progress)

            return result

        except Exception as e:
            result = FileReviewResult(
                file_path=file_path,
                result=ReviewResult.ERROR,
                reason=f"Error during analysis: {str(e)}",
            )
            async with progress_lock:
                progress["errors"].append({"file": file_path, "error": str(e)})
                save_review_progress(progress)
            print(f"{progress_prefix} [ERROR] {file_path}: {e}")
            return result


# ==============================================================================
# Command: translate
# ==============================================================================


async def cmd_translate(
    directories: list[Path],
    max_concurrency: int = 5,
    dry_run: bool = False,
    specific_files: list[str] | None = None,
    reset_progress: bool = False,
):
    """Execute the translate command."""
    print_header("Chinese to English Translation")

    print("CRITICAL RULES:")
    print("  1. DO NOT modify any code logic")
    print("  2. Only translate Chinese comments and log messages")
    print("  3. Preserve all code structure and behavior")
    print()
    print(f"Target directories: {[str(d) for d in directories]}")
    print()

    if reset_progress:
        clear_translation_progress()
        print("Progress cleared, starting fresh")
    progress = load_translation_progress()
    if progress.get("processed"):
        print(
            f"Resuming from previous run: {len(progress['processed'])} files already processed"
        )

    provider = _load_env_and_get_llm_provider()

    if specific_files:
        python_files = [Path(f) for f in specific_files]
    else:
        python_files = get_python_files_from_directories(directories)

    python_files.sort()
    total_files = len(python_files)
    print(f"Found {total_files} Python files in total")
    print(f"Max concurrency: {max_concurrency}")
    print(f"Max file size: {MAX_FILE_SIZE/1024:.1f}KB")
    print(f"Dry run: {dry_run}")
    print()

    files_to_process, skipped_no_chinese, skipped_already_done = (
        filter_files_with_chinese(python_files, progress)
    )

    if not files_to_process:
        print("No files with Chinese content to process!")
        return 0

    print()
    print(f"Files to translate: {len(files_to_process)}")
    print()

    semaphore = asyncio.Semaphore(max_concurrency)
    progress_lock = asyncio.Lock()

    tasks = [
        translate_file(
            provider,
            file_path,
            semaphore,
            progress,
            progress_lock,
            dry_run,
            index=idx + 1,
            total=len(files_to_process),
        )
        for idx, file_path in enumerate(files_to_process)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = 0
    error_count = 0
    errors = []

    for result in results:
        if isinstance(result, Exception):
            error_count += 1
            errors.append(str(result))
        else:
            file_path, success, error_msg = result
            if success:
                success_count += 1
            else:
                error_count += 1
                errors.append(f"{file_path}: {error_msg}")

    print_summary_header()
    print(f"Total Python files found: {total_files}")
    print(f"Files skipped (no Chinese): {skipped_no_chinese}")
    print(f"Files skipped (already done): {skipped_already_done}")
    print(f"Files translated this run: {len(files_to_process)}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print(
        f"Total processed (including previous runs): {len(progress.get('processed', []))}"
    )

    if errors:
        print()
        print("Errors encountered:")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")

    print()
    print(f"Progress saved to: {TRANSLATION_PROGRESS_FILE}")
    print("Run with --reset to start fresh")
    return 0 if error_count == 0 else 1


# ==============================================================================
# Command: check
# ==============================================================================


def cmd_check(directories: list[Path], specific_files: list[str] | None = None) -> int:
    """Execute the check command."""
    print_header("Chinese Content Check")

    if specific_files:
        python_files = [Path(f) for f in specific_files]
    else:
        python_files = get_python_files_from_directories(directories)

    python_files.sort()
    total_files = len(python_files)
    print(f"Scanning {total_files} Python files for Chinese content...")
    print()

    files_with_chinese = []
    files_checked = 0

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            files_checked += 1

            if contains_chinese(content):
                lines_with_chinese = []
                for line_num, line in enumerate(content.split('\n'), 1):
                    if contains_chinese(line):
                        lines_with_chinese.append((line_num, line.strip()[:80]))

                files_with_chinese.append(
                    {
                        'path': file_path,
                        'lines': lines_with_chinese,
                        'total_chinese_lines': len(lines_with_chinese),
                    }
                )

        except Exception as e:
            print(f"  [ERROR] Could not read {file_path}: {e}")

    print("=" * 70)
    print("Check Results")
    print("=" * 70)
    print(f"Total files checked: {files_checked}")
    print(f"Files with Chinese: {len(files_with_chinese)}")
    print()

    if files_with_chinese:
        print("Files containing Chinese characters:")
        print("-" * 70)
        for file_info in files_with_chinese:
            print(f"\n📄 {file_info['path']}")
            print(f"   ({file_info['total_chinese_lines']} lines with Chinese)")
            for line_num, line_content in file_info['lines'][:5]:
                print(f"   Line {line_num}: {line_content}...")
            if len(file_info['lines']) > 5:
                print(f"   ... and {len(file_info['lines']) - 5} more lines")
        print()
        print("-" * 70)
        print(f"❌ Found {len(files_with_chinese)} files with Chinese content")
        print("   Run 'translate' command to translate them")
    else:
        print("✅ No Chinese content found in any Python files!")

    return len(files_with_chinese)


# ==============================================================================
# Command: review
# ==============================================================================


async def cmd_review(
    commit_ref: str = "HEAD",
    max_concurrency: int = 5,
    verbose: bool = False,
    dry_run: bool = False,
    reset_progress: bool = False,
) -> int:
    """Execute the review command."""
    print_header("Translation Changes Review")

    # Determine if it's a single commit or a range
    is_range = ".." in commit_ref

    if is_range:
        # It's a range like HEAD~3..HEAD
        display_ref = commit_ref.split("..")[-1]
    else:
        # Single commit reference
        display_ref = commit_ref

    success, commit_info = get_commit_info(display_ref)
    if success:
        print(f"Reviewing commit: {commit_info['hash']}")
        print(f"  Message: {commit_info['message']}")
        print(f"  Author: {commit_info['author']}")
        print(f"  Date: {commit_info['date']}")

    if is_range:
        print(f"Commit range: {commit_ref}")
    else:
        print(f"Single commit: {commit_ref}")
    print()

    # Handle progress
    if reset_progress:
        clear_review_progress()
        print("Progress cleared, starting fresh")

    progress = load_review_progress()

    # Check if we're resuming a different commit
    if progress.get("commit_ref") and progress.get("commit_ref") != commit_ref:
        print(f"Previous review was for different commit: {progress['commit_ref']}")
        print("Clearing progress and starting fresh for new commit")
        clear_review_progress()
        progress = load_review_progress()

    progress["commit_ref"] = commit_ref

    print("Getting changed Python files...")
    success, files = get_changed_files_from_git(commit_ref)
    if not success:
        print(f"Error getting changed files: {files[0] if files else 'Unknown error'}")
        return 1

    if not files:
        print("No Python files changed in this commit range.")
        return 0

    # Filter out already processed files
    already_processed = set(progress.get("safe", []))
    already_processed.update(item["file"] for item in progress.get("needs_review", []))
    already_processed.update(item["file"] for item in progress.get("errors", []))

    files_to_process = [f for f in files if f not in already_processed]

    print(f"Found {len(files)} changed Python file(s) total:")
    for f in files:
        status = ""
        if f in progress.get("safe", []):
            status = " [already: SAFE]"
        elif f in [item["file"] for item in progress.get("needs_review", [])]:
            status = " [already: NEEDS-REVIEW]"
        elif f in [item["file"] for item in progress.get("errors", [])]:
            status = " [already: ERROR]"
        print(f"  - {f}{status}")
    print()

    if already_processed:
        print(
            f"Resuming from previous run: {len(already_processed)} files already processed"
        )
        print(f"Files remaining to process: {len(files_to_process)}")
        print()

    if dry_run:
        print("[DRY-RUN] Skipping LLM analysis")
        return 0

    if not files_to_process:
        print("All files already processed!")
    else:
        provider = _load_env_and_get_llm_provider()

        print("Analyzing changes with LLM...")
        print()
        semaphore = asyncio.Semaphore(max_concurrency)
        progress_lock = asyncio.Lock()
        tasks = []

        for idx, file_path in enumerate(files_to_process):
            # Try to get diff with default context (3 lines)
            success, diff = get_file_diff(commit_ref, file_path, context_lines=3)

            # If diff is too large, try with minimal context (0 lines)
            if success and len(diff) > MAX_DIFF_SIZE:
                print(
                    f"  [INFO] {file_path}: diff too large ({len(diff)/1024:.1f}KB), retrying with minimal context..."
                )
                success, diff = get_file_diff(commit_ref, file_path, context_lines=0)
                if success:
                    print(f"  [INFO] {file_path}: reduced to {len(diff)/1024:.1f}KB")

            if not success:

                async def make_error_result(
                    fp=file_path, err=diff, prog=progress, lock=progress_lock
                ):
                    async with lock:
                        prog["errors"].append(
                            {"file": fp, "error": f"Error getting diff: {err}"}
                        )
                        save_review_progress(prog)
                    return FileReviewResult(
                        file_path=fp,
                        result=ReviewResult.ERROR,
                        reason=f"Error getting diff: {err}",
                    )

                tasks.append(make_error_result())
            else:
                tasks.append(
                    review_file_diff(
                        provider,
                        file_path,
                        diff,
                        semaphore,
                        progress,
                        progress_lock,
                        verbose,
                        index=idx + 1,
                        total=len(files_to_process),
                    )
                )

        await asyncio.gather(*tasks, return_exceptions=True)

    # Reload progress to get final results
    progress = load_review_progress()

    # Build result lists from progress
    safe_files = [
        FileReviewResult(file_path=f, result=ReviewResult.SAFE, reason="")
        for f in progress.get("safe", [])
    ]
    needs_review_files = [
        FileReviewResult(
            file_path=item["file"],
            result=ReviewResult.NEEDS_REVIEW,
            reason=item["reason"],
        )
        for item in progress.get("needs_review", [])
    ]
    error_files = [
        FileReviewResult(
            file_path=item["file"],
            result=ReviewResult.ERROR,
            reason=item.get("error", "Unknown error"),
        )
        for item in progress.get("errors", [])
    ]

    print()
    print("=" * 70)
    print("Review Results")
    print("=" * 70)
    print()

    if needs_review_files:
        print("🔴 FILES NEEDING MANUAL REVIEW (possible code changes):")
        print("-" * 70)
        for r in needs_review_files:
            print(f"\n📄 {r.file_path}")
            print(f"   Reason: {r.reason}")
        print()

    if safe_files:
        print("🟢 SAFE FILES (translation only, no review needed):")
        print("-" * 70)
        for r in safe_files:
            print(f"  ✓ {r.file_path}")
        print()

    if error_files:
        print("⚠️  ERROR FILES (could not analyze):")
        print("-" * 70)
        for r in error_files:
            print(f"  ✗ {r.file_path}: {r.reason}")
        print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total files in commit: {len(files)}")
    print(f"  🟢 Safe (no review needed): {len(safe_files)}")
    print(f"  🔴 Needs review: {len(needs_review_files)}")
    print(f"  ⚠️  Errors: {len(error_files)}")
    print()
    print(f"Progress saved to: {REVIEW_PROGRESS_FILE}")
    print("Run with --reset to start fresh")
    print()

    if needs_review_files:
        print("⚠️  Please manually review the files marked with 🔴")
        print("   These files may contain unintended code changes.")
        return 1
    else:
        print("✅ All changes appear to be translation-only.")
        return 0


# ==============================================================================
# Command: hook (pre-commit hook)
# ==============================================================================


def _hook_should_skip_file(file_path: str) -> bool:
    """Check if a file should be skipped based on HOOK_SKIP_PATTERNS.

    Only applies skip patterns to files within the project directory.
    """
    # Try to get relative path from project root
    try:
        abs_path = Path(file_path).resolve()
        rel_path = str(abs_path.relative_to(PROJECT_DIR)).replace("\\", "/")
    except ValueError:
        # File is outside project directory, don't skip
        return False

    file_name = Path(file_path).name

    for pattern in HOOK_SKIP_PATTERNS:
        pattern = pattern.replace("\\", "/")

        # Pattern like "*.md" - match by extension
        if pattern.startswith("*."):
            if fnmatch(file_name, pattern):
                return True
            continue

        # Pattern like "**/*.py" or "**/test_*.py" - recursive glob
        if pattern.startswith("**/"):
            if fnmatch(rel_path, pattern) or fnmatch(file_name, pattern[3:]):
                return True
            continue

        # Pattern like "dir/*" or "dir/**" - directory prefix
        if pattern.endswith("/*") or pattern.endswith("/**"):
            dir_prefix = pattern.rstrip("/*")
            if rel_path.startswith(dir_prefix + "/"):
                return True
            continue

        # Exact file/path match
        if fnmatch(rel_path, pattern):
            return True

    return False


def _hook_contains_cjk(text: str) -> bool:
    """Check if text contains CJK (Chinese/Japanese/Korean) characters."""
    return bool(CJK_PATTERN.search(text))


def _hook_line_has_skip_comment(line: str) -> bool:
    """Check if a line has the skip-i18n-check inline comment.

    The comment can appear anywhere in the line (typically at the end).
    Whitespace around the comment marker is ignored.
    """
    # Normalize the line for checking (remove spaces around #)
    # Match patterns like: #skip-i18n-check, # skip-i18n-check, #  skip-i18n-check
    normalized = line.lower().replace(" ", "")
    return "#skip-i18n-check" in normalized


def _hook_file_has_skip_marker(content: str) -> bool:
    """Check if file has a skip-i18n-file marker in the first 10 lines.

    The marker can be:
    - #skip-i18n-file
    - # skip-i18n-file
    - # -*- skip-i18n-file -*-
    """
    lines = content.split("\n")[:10]  # Only check first 10 lines
    for line in lines:
        line_lower = line.lower().strip()
        for marker in HOOK_SKIP_FILE_MARKERS:
            if marker in line_lower:
                return True
    return False


def _hook_find_cjk_lines(content: str) -> list[tuple[int, str]]:
    """Find all lines containing CJK characters.

    Lines with #skip-i18n-check comment are skipped.
    If file has #skip-i18n-file marker in first 10 lines, entire file is skipped.

    Returns:
        List of tuples: (line_number, line_content)
    """
    # Check for file-level skip marker
    if _hook_file_has_skip_marker(content):
        return []

    cjk_lines = []
    for line_num, line in enumerate(content.split("\n"), 1):
        # Skip lines with inline skip comment
        if _hook_line_has_skip_comment(line):
            continue
        if _hook_contains_cjk(line):
            display_line = line.strip()[:100]
            if len(line.strip()) > 100:
                display_line += "..."
            cjk_lines.append((line_num, display_line))
    return cjk_lines


def _hook_get_relative_path(file_path: str) -> str:
    """Get the relative path from project root."""
    try:
        abs_path = Path(file_path).resolve()
        return str(abs_path.relative_to(PROJECT_DIR))
    except ValueError:
        return file_path


def _hook_format_translation_command(files: list[str]) -> str:
    """Format the i18n_tool.py translation command for the given files."""
    rel_files = [_hook_get_relative_path(f) for f in files]

    if len(rel_files) == 1:
        return f"python -m src.devops_scripts.i18n.i18n_tool translate --files {rel_files[0]}"
    else:
        files_str = " ".join(rel_files)
        return (
            f"python -m src.devops_scripts.i18n.i18n_tool translate --files {files_str}"
        )


def _hook_check_files(
    files: list[str],
) -> tuple[bool, dict[str, list[tuple[int, str]]]]:
    """Check files for CJK characters.

    Returns:
        Tuple of (has_errors, files_with_cjk)
        files_with_cjk is a dict: {file_path: [(line_num, content), ...]}
    """
    files_with_cjk = {}

    for file_path in files:
        # Check if file should be skipped
        if _hook_should_skip_file(file_path):
            continue

        # Check if file exists
        if not Path(file_path).exists():
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            cjk_lines = _hook_find_cjk_lines(content)
            if cjk_lines:
                files_with_cjk[file_path] = cjk_lines

        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)

    return bool(files_with_cjk), files_with_cjk


def _hook_check_commit_message(msg_file: str) -> tuple[bool, list[tuple[int, str]]]:
    """Check commit message for CJK characters.

    Returns:
        Tuple of (has_cjk, cjk_lines)
    """
    try:
        with open(msg_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for skip keywords
        content_lower = content.lower()
        for keyword in HOOK_SKIP_COMMIT_MSG_KEYWORDS:
            if keyword.lower() in content_lower:
                return False, []

        cjk_lines = _hook_find_cjk_lines(content)
        return bool(cjk_lines), cjk_lines

    except Exception as e:
        print(f"Warning: Could not read commit message file: {e}", file=sys.stderr)
        return False, []


def _hook_print_error_report(
    files_with_cjk: dict[str, list[tuple[int, str]]],
    commit_msg_cjk: list[tuple[int, str]] | None = None,
):
    """Print a detailed error report."""
    print("\n" + "=" * 70, file=sys.stderr)
    print("❌ NON-ENGLISH CHARACTERS DETECTED (CJK)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    if files_with_cjk:
        print("\n📄 Files containing non-English characters:", file=sys.stderr)
        print("-" * 70, file=sys.stderr)

        for file_path, cjk_lines in files_with_cjk.items():
            rel_path = _hook_get_relative_path(file_path)
            print(f"\n  {rel_path} ({len(cjk_lines)} lines)", file=sys.stderr)
            for line_num, content in cjk_lines[:5]:
                # Use file:line format for clickable terminal links
                print(f"    {rel_path}:{line_num}: {content}", file=sys.stderr)
            if len(cjk_lines) > 5:
                print(f"    ... and {len(cjk_lines) - 5} more lines", file=sys.stderr)

    if commit_msg_cjk:
        print("\n💬 Commit message contains non-English characters:", file=sys.stderr)
        print("-" * 70, file=sys.stderr)
        for line_num, content in commit_msg_cjk:
            print(f"    Line {line_num}: {content}", file=sys.stderr)

    print("\n" + "=" * 70, file=sys.stderr)
    print("HOW TO FIX", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    if files_with_cjk:
        files_list = list(files_with_cjk.keys())
        cmd = _hook_format_translation_command(files_list)
        print(f"\n1. Translate the files using:", file=sys.stderr)
        print(f"   {cmd}", file=sys.stderr)
        print("\n   Or for dry-run first:", file=sys.stderr)
        print(f"   {cmd} --dry-run", file=sys.stderr)

    if commit_msg_cjk:
        print("\n2. Rewrite your commit message in English", file=sys.stderr)

    print("\n" + "-" * 70, file=sys.stderr)
    print("TO SKIP THIS CHECK (use sparingly):", file=sys.stderr)
    print("-" * 70, file=sys.stderr)
    print(
        "  • Add '# skip-i18n-file' in first 10 lines to skip entire file",
        file=sys.stderr,
    )
    print(
        "  • Add inline comment to skip specific line: #skip-i18n-check",
        file=sys.stderr,
    )
    print('    Example: if "中文" in text:  #skip-i18n-check', file=sys.stderr)
    print(f"  • Set environment variable: {HOOK_SKIP_ENV_VAR}=1", file=sys.stderr)
    print("  • Add to commit message: [skip-i18n] or #skip-i18n-check", file=sys.stderr)
    print(
        "  • Add file patterns to HOOK_SKIP_PATTERNS in i18n_tool.py", file=sys.stderr
    )
    print("\n" + "=" * 70 + "\n", file=sys.stderr)


def cmd_hook(files: list[str], commit_msg: bool = False) -> int:
    """Execute the hook command for pre-commit.

    Args:
        files: List of files to check, or commit message file if commit_msg=True
        commit_msg: If True, check commit message instead of files

    Returns:
        0 if no CJK found, 1 if CJK found
    """
    # Check for skip environment variable
    if os.environ.get(HOOK_SKIP_ENV_VAR, "").lower() in ("1", "true", "yes"):
        print(f"Skipping i18n check ({HOOK_SKIP_ENV_VAR} is set)")
        return 0

    has_error = False
    files_with_cjk = {}
    commit_msg_cjk = None

    if commit_msg:
        # Check commit message
        if files:
            msg_file = files[0]
            has_cjk, commit_msg_cjk = _hook_check_commit_message(msg_file)
            if has_cjk:
                has_error = True
    else:
        # Check staged files
        if files:
            has_error, files_with_cjk = _hook_check_files(files)

    if has_error:
        _hook_print_error_report(files_with_cjk, commit_msg_cjk)
        return 1

    return 0


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="I18N Tool - Chinese to English translation and review tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Translate command
    translate_parser = subparsers.add_parser(
        "translate", help="Translate Chinese comments/logs to English in Python files"
    )
    translate_parser.add_argument(
        "--directory",
        "-d",
        nargs="*",
        help="Directories to scan (relative to project root). Default: src",
    )
    translate_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrent translations (default: 5)",
    )
    translate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write changes, just show what would be done",
    )
    translate_parser.add_argument(
        "--files", nargs="*", help="Specific files to translate (optional)"
    )
    translate_parser.add_argument(
        "--reset", action="store_true", help="Clear previous progress and start fresh"
    )

    # Check command
    check_parser = subparsers.add_parser(
        "check", help="Check for remaining Chinese content in Python files"
    )
    check_parser.add_argument(
        "--directory",
        "-d",
        nargs="*",
        help="Directories to scan (relative to project root). Default: src",
    )
    check_parser.add_argument(
        "--files", nargs="*", help="Specific files to check (optional)"
    )

    # Review command
    review_parser = subparsers.add_parser(
        "review", help="Review git commits to verify translation changes"
    )
    review_parser.add_argument(
        "--commit",
        "-c",
        default="HEAD",
        help="Git commit or range to review (default: HEAD)",
    )
    review_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrent LLM calls (default: 5)",
    )
    review_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including diff previews",
    )
    review_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list changed files without LLM analysis",
    )
    review_parser.add_argument(
        "--reset", action="store_true", help="Clear previous progress and start fresh"
    )

    # Hook command (for pre-commit)
    hook_parser = subparsers.add_parser(
        "hook", help="Pre-commit hook to check for non-English (CJK) characters"
    )
    hook_parser.add_argument(
        "--commit-msg",
        action="store_true",
        help="Check commit message instead of files",
    )
    hook_parser.add_argument(
        "files",
        nargs="*",
        help="Files to check (for pre-commit) or commit message file (for commit-msg)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "hook":
        # Hook command doesn't need project dependencies
        exit_code = cmd_hook(files=args.files, commit_msg=args.commit_msg)
        sys.exit(exit_code)

    if args.command == "translate":
        directories = resolve_directories(args.directory)
        exit_code = asyncio.run(
            cmd_translate(
                directories=directories,
                max_concurrency=args.max_concurrency,
                dry_run=args.dry_run,
                specific_files=args.files,
                reset_progress=args.reset,
            )
        )
        sys.exit(exit_code)

    elif args.command == "check":
        directories = resolve_directories(args.directory)
        exit_code = cmd_check(directories=directories, specific_files=args.files)
        sys.exit(0 if exit_code == 0 else 1)

    elif args.command == "review":
        exit_code = asyncio.run(
            cmd_review(
                commit_ref=args.commit,
                max_concurrency=args.max_concurrency,
                verbose=args.verbose,
                dry_run=args.dry_run,
                reset_progress=args.reset,
            )
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
