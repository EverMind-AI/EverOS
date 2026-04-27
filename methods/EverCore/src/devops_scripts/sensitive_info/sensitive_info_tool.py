"""
Sensitive Information Detection Tool - AI-Powered Pre-commit Hook

This tool uses LLM to intelligently scan files for sensitive information 
before committing to GitHub:
1. Credentials: API keys, passwords, secrets, tokens
2. Internal network config: Private IPs, internal domains  
3. Personal data: Real names, phone numbers, emails, ID numbers
4. Test data that may point to real users

Usage:
    # Check command (AI-powered scan)
    python -m devops_scripts.sensitive_info.sensitive_info_tool check
    python -m devops_scripts.sensitive_info.sensitive_info_tool check --directory tests
    python -m devops_scripts.sensitive_info.sensitive_info_tool check --files file1.py file2.py

    # Hook command (for pre-commit, uses AI)
    python -m devops_scripts.sensitive_info.sensitive_info_tool hook file1.py file2.py
"""

from __future__ import annotations

import os
import sys
import asyncio
import json
from fnmatch import fnmatch
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ==============================================================================
# Path Configuration
# ==============================================================================

SRC_DIR = Path(__file__).parent.parent.parent
PROJECT_DIR = SRC_DIR.parent


def _setup_project_imports():
    """Setup project imports when needed (lazy loading)."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


# ==============================================================================
# Configuration
# ==============================================================================

# Maximum file size to process (100KB for AI analysis)
MAX_FILE_SIZE = 100 * 1024

# Maximum content size to send to LLM at once
MAX_CHUNK_SIZE = 30000

# Directories to always skip
SKIP_DIRECTORIES = [
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
    # Prompt directories (contain example text for LLM prompts, not real data)
    "prompts",
]

# Files to skip by pattern
SKIP_FILE_PATTERNS = [
    # Lock files
    "*.lock",
    "package-lock.json",
    "yarn.lock",
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
    "*.whl",
    # Compiled files
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.exe",
    # Git files
    ".gitignore",
    ".gitattributes",
    # This tool itself
    "sensitive_info_tool.py",
    # Template files (expected to have placeholders)
    "env.template",
    "*.template",
    # Stopword files
    "*stopwords*.txt",
    # i18n tool
    "i18n_tool.py",
]

# Environment variable to skip hook
HOOK_SKIP_ENV_VAR = "SKIP_SENSITIVE_CHECK"

# Inline comment to skip check for a specific line
SKIP_LINE_COMMENT = "#skip-sensitive-check"

# File-level skip marker - add this at the top of a file to skip the entire file
# Usage: Add "# skip-sensitive-file" in the first 10 lines
SKIP_FILE_MARKERS = [
    "#skip-sensitive-file",
    "# skip-sensitive-file",
    "#  skip-sensitive-file",
    "# -*- skip-sensitive-file -*-",
]

# Keywords in commit message to skip check
SKIP_COMMIT_MSG_KEYWORDS = [
    "skip-sensitive-check",
    "#skip-sensitive-check",
    "[skip-sensitive]",
    "[no-sensitive-check]",
]

# Concurrency limit for LLM calls
MAX_CONCURRENCY = 5


# ==============================================================================
# Result Types
# ==============================================================================


class Severity(Enum):
    """Severity levels for sensitive information."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SensitiveIssue:
    """A single sensitive information issue found by AI."""

    line_number: int
    line_content: str
    issue_type: str
    severity: Severity
    description: str
    suggestion: str


@dataclass
class FileCheckResult:
    """Result of checking a single file."""

    file_path: str
    issues: list[SensitiveIssue] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""
    ai_analysis: str = ""


# ==============================================================================
# LLM Prompt
# ==============================================================================

ANALYSIS_PROMPT_TEMPLATE = '''You are a security expert reviewing code before it's published to GitHub.
Your goal is to find REAL sensitive data that could cause harm if leaked, NOT placeholder/test data.

## CRITICAL: Check Configuration Names Carefully

For EVERY line that sets a default value for a configuration (e.g., os.getenv("X", "default"), get_env("X", "default")):
- Split the default value by "_" or "-"  
- Check: Is each word a COMMON English word or standard tech term?
- If ANY word is NOT recognizable as a dictionary word, flag it as MEDIUM
- This applies to: Kafka topics, queue names, bucket names, service names, group IDs, etc.

## Severity Levels

### HIGH severity (must fix before commit)
- REAL API keys: long random strings that look like actual production keys
- REAL passwords: complex passwords that appear to be actual credentials
- Private keys, certificates, tokens that look real
- Real personal data: actual phone numbers, ID cards, real names of specific real individuals

### MEDIUM severity (should review)
- Internal IPs/domains that appear to be REAL infrastructure configuration
  
  Principle: Ask "Does this look like a placeholder/example, or a real server address?"
  - Placeholder IPs have predictable patterns: x.x.0.1, x.x.1.1, x.x.0.0, x.x.255.255
  - Real IPs have arbitrary middle/end segments that look like actual assignments
  - Example: 192.168.1.1 is clearly a placeholder (common router default)
  - Example: 192.168.47.83 looks like a real assigned IP (arbitrary numbers)

- Internal configuration names that reveal real infrastructure
  
  PRINCIPLE: Configuration names should only contain DICTIONARY WORDS or INDUSTRY-STANDARD terms.
  
  Flag as MEDIUM if a configuration name contains:
  - Abbreviations that are NOT industry-standard (API, HTTP, DB, SQL, JSON are standard)
  - Words that are NOT in an English dictionary
  - Words that look like they could be project names, product names, or company abbreviations
  
  The test: Can you find this word in a standard English dictionary or official technical documentation?
  If not, it's likely internal terminology that reveals infrastructure details.
  
  NOTE: Even in os.getenv("VAR", "fallback") - if the fallback looks specific, flag it!
  
- Internal domains that look real (specific hostnames, not generic like "example.internal")
- Data that MIGHT be real but you're not sure
- Potential real user references that aren't obviously test data

### LOW severity (just a reminder, okay to commit)
- Simple/obvious placeholder passwords: "123456", "123", "password", "admin", "test", "root", "memsys123"
  These are clearly NOT real secrets - no one uses these as actual passwords
- Documentation examples showing connection formats
- Anything that looks like intentional test/demo data

## Key Principle: Does it look like a REAL secret?

Ask yourself: Would a real person use this as their actual password/key?
- "MyR3alP@ssw0rd!2024" → Looks real → HIGH
- "123456" or "password" → Obviously placeholder → LOW or SAFE
- "sk-proj-abc123xyz..." (40+ chars) → Looks like real API key → HIGH
- "123" or "xxxx" → Obviously fake → SAFE

## What is SAFE (DO NOT flag)

Apply this principle: "Does it look intentionally fake/placeholder, or accidentally real?"

- Placeholder patterns: strings with "xxxx", "your-", "${VAR}", "{{...}}", "<placeholder>"
- IPs that are OBVIOUSLY examples: 
  Pattern: ends with .0.1, .1.1, .0.0, .255.255, or is localhost/127.0.0.1
  These are universally recognized defaults, not real infrastructure
- Generic/example domains: contains "example", "test", "demo", "sample", "foo", "bar", or is localhost
- Generic configuration names: "test-topic", "my-queue", "example-db", "default", "sample-bucket"
  These use common placeholder words, not project-specific terms
- Test names: Names that are culturally known as placeholder names (like John Doe in English)
- Test emails: Uses obviously fake domains or placeholder usernames
- Test phone numbers: Sequential digits, repeated patterns, or known test numbers
- Environment variable reads WITHOUT fallback: os.getenv("SECRET") with no default is correct practice
- Documentation showing formats/examples
- Test file data with obviously fictional content

## Response Format

JSON only (no markdown):
{"status": "ISSUES_FOUND", "issues": [{"line_number": 42, "line_content": "content", "issue_type": "Type", "severity": "HIGH|MEDIUM|LOW", "description": "why", "suggestion": "fix"}]}

Or: {"status": "SAFE", "issues": []}

## File to Analyze

File path: __FILE_PATH__

```
__CONTENT__
```

IMPORTANT: When reviewing, you MUST check these in order:
1. API keys and passwords - flag if they look REAL (complex, not "123456")
2. IP addresses in defaults - flag if they look ASSIGNED (not .0.1 or .1.1 patterns)
3. Configuration names (Kafka topics, queues, buckets, etc.) - flag if they contain NON-DICTIONARY words
   For each configuration name, split by "_" or "-" and check: is each part a common English word?
   If you find a word that is NOT in a standard dictionary, it's likely an internal codename - flag as MEDIUM.

Be conservative on passwords (simple ones like "123456" are LOW), but be STRICT on configuration names.'''


def build_analysis_prompt(file_path: str, content: str) -> str:
    """Build the analysis prompt with file content."""
    return ANALYSIS_PROMPT_TEMPLATE.replace("__FILE_PATH__", file_path).replace(
        "__CONTENT__", content
    )


# ==============================================================================
# Utility Functions
# ==============================================================================


def should_skip_file(file_path: str) -> tuple[bool, str]:
    """Check if a file should be skipped.

    Returns:
        Tuple of (should_skip, reason)
    """
    path = Path(file_path)
    file_name = path.name

    # Check directory patterns
    for part in path.parts:
        if part in SKIP_DIRECTORIES:
            return True, f"In skipped directory: {part}"

    # Check file patterns
    for pattern in SKIP_FILE_PATTERNS:
        if fnmatch(file_name, pattern):
            return True, f"Matches skip pattern: {pattern}"

    return False, ""


def get_relative_path(file_path: str) -> str:
    """Get relative path from project root."""
    try:
        return str(Path(file_path).relative_to(PROJECT_DIR))
    except ValueError:
        return file_path


# ==============================================================================
# File Discovery
# ==============================================================================


def get_files_from_directory(directory: Path) -> list[Path]:
    """Get all files under a directory recursively."""
    files = []

    for root, dirs, filenames in os.walk(directory):
        # Remove skipped directories
        dirs[:] = [
            d for d in dirs if d not in SKIP_DIRECTORIES and not d.startswith(".")
        ]

        for filename in filenames:
            file_path = Path(root) / filename
            files.append(file_path)

    return files


def get_files_from_directories(directories: list[Path]) -> list[Path]:
    """Get all files from multiple directories."""
    all_files = []
    for directory in directories:
        if directory.exists():
            files = get_files_from_directory(directory)
            all_files.extend(files)
    return all_files


def resolve_directories(dir_names: list[str] | None) -> list[Path]:
    """Resolve directory names to absolute paths."""
    if not dir_names:
        return [
            SRC_DIR,
            PROJECT_DIR / "tests",
            PROJECT_DIR / "data",
            PROJECT_DIR / "demo",
        ]

    directories = []
    for dir_name in dir_names:
        dir_path = Path(dir_name)
        if not dir_path.is_absolute():
            dir_path = PROJECT_DIR / dir_name
        directories.append(dir_path)
    return directories


# ==============================================================================
# AI Analysis
# ==============================================================================


def _load_llm_provider():
    """Load LLM provider."""
    _setup_project_imports()

    from dotenv import load_dotenv
    from memory_layer.llm import OpenAIProvider

    env_file_path = PROJECT_DIR / ".env"
    if env_file_path.exists():
        load_dotenv(env_file_path)

    return OpenAIProvider(
        model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.1,
    )


def parse_ai_response(response: str) -> tuple[str, list[SensitiveIssue]]:
    """Parse AI response and extract issues."""
    issues = []

    # Try to extract JSON from response
    try:
        # Find JSON in response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            status = data.get("status", "UNKNOWN")

            for issue_data in data.get("issues", []):
                severity_str = issue_data.get("severity", "MEDIUM").upper()
                severity = (
                    Severity.HIGH
                    if severity_str == "HIGH"
                    else (Severity.MEDIUM if severity_str == "MEDIUM" else Severity.LOW)
                )

                issues.append(
                    SensitiveIssue(
                        line_number=issue_data.get("line_number", 0),
                        line_content=issue_data.get("line_content", "")[:100],
                        issue_type=issue_data.get("issue_type", "Unknown"),
                        severity=severity,
                        description=issue_data.get("description", ""),
                        suggestion=issue_data.get("suggestion", ""),
                    )
                )

            return status, issues
    except json.JSONDecodeError:
        pass

    # If JSON parsing fails, try to determine status from text
    response_upper = response.upper()
    if "SAFE" in response_upper and "ISSUES" not in response_upper:
        return "SAFE", []

    return "PARSE_ERROR", []


async def analyze_file_with_ai(
    provider,
    file_path: str,
    semaphore: asyncio.Semaphore,
    index: int = 0,
    total: int = 0,
) -> FileCheckResult:
    """Analyze a single file using AI."""
    progress_prefix = f"[{index}/{total}]" if total > 0 else ""
    rel_path = get_relative_path(file_path)

    # Check if should skip
    should_skip, reason = should_skip_file(file_path)
    if should_skip:
        return FileCheckResult(file_path=file_path, skipped=True, skip_reason=reason)

    # Check file exists and size
    path = Path(file_path)
    if not path.exists():
        return FileCheckResult(
            file_path=file_path, skipped=True, skip_reason="File not found"
        )

    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE:
        return FileCheckResult(
            file_path=file_path,
            skipped=True,
            skip_reason=f"File too large ({file_size / 1024:.1f}KB > {MAX_FILE_SIZE / 1024:.0f}KB)",
        )

    # Read file content
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        return FileCheckResult(file_path=file_path, error=str(e))

    # Skip empty files
    if not content.strip():
        return FileCheckResult(
            file_path=file_path, skipped=True, skip_reason="Empty file"
        )

    # Check for file-level skip marker in first 10 lines
    first_lines = content.split("\n")[:10]
    for line in first_lines:
        line_lower = line.lower().strip()
        for marker in SKIP_FILE_MARKERS:
            if marker in line_lower:
                return FileCheckResult(
                    file_path=file_path,
                    skipped=True,
                    skip_reason="File has #skip-sensitive-file marker",
                )

    # Filter out lines with skip comment
    lines = content.split("\n")
    filtered_lines = []
    for i, line in enumerate(lines, 1):
        if SKIP_LINE_COMMENT.lower() not in line.lower().replace(" ", ""):
            filtered_lines.append(f"{i:6d}|{line}")

    numbered_content = "\n".join(filtered_lines)

    # Truncate if too long
    if len(numbered_content) > MAX_CHUNK_SIZE:
        numbered_content = numbered_content[:MAX_CHUNK_SIZE] + "\n... [truncated]"

    async with semaphore:
        try:
            print(f"{progress_prefix} [ANALYZING] {rel_path}")

            prompt = build_analysis_prompt(rel_path, numbered_content)

            response = await provider.generate(prompt, temperature=0.1)

            status, issues = parse_ai_response(response)

            if status == "SAFE" or not issues:
                print(f"{progress_prefix} [SAFE] {rel_path}")
            else:
                high_count = sum(1 for i in issues if i.severity == Severity.HIGH)
                med_count = sum(1 for i in issues if i.severity == Severity.MEDIUM)
                print(
                    f"{progress_prefix} [ISSUES] {rel_path} - {high_count} high, {med_count} medium"
                )

            return FileCheckResult(
                file_path=file_path,
                issues=issues,
                ai_analysis=response[:500] if len(response) > 500 else response,
            )

        except Exception as e:
            print(f"{progress_prefix} [ERROR] {rel_path}: {e}")
            return FileCheckResult(file_path=file_path, error=str(e))


# ==============================================================================
# Output Functions
# ==============================================================================


def print_header(title: str):
    """Print a section header."""
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def print_results(results: list[FileCheckResult], verbose: bool = False):
    """Print check results in a formatted way."""
    files_with_issues = [r for r in results if r.issues]
    files_skipped = [r for r in results if r.skipped]
    files_with_errors = [r for r in results if r.error]
    files_clean = [r for r in results if not r.issues and not r.skipped and not r.error]

    # Group issues by severity
    high_severity = []
    medium_severity = []
    low_severity = []

    for result in files_with_issues:
        for issue in result.issues:
            entry = (result.file_path, issue)
            if issue.severity == Severity.HIGH:
                high_severity.append(entry)
            elif issue.severity == Severity.MEDIUM:
                medium_severity.append(entry)
            else:
                low_severity.append(entry)

    if high_severity:
        print("\n🔴 HIGH SEVERITY ISSUES:")
        print("-" * 70)
        for file_path, issue in high_severity:
            rel_path = get_relative_path(file_path)
            print(f"\n  📄 {rel_path}:{issue.line_number}")
            print(f"     [{issue.issue_type}] {issue.description}")
            if issue.line_content:
                print(f"     Line: {issue.line_content}")
            if issue.suggestion:
                print(f"     Fix: {issue.suggestion}")

    if medium_severity:
        print("\n🟡 MEDIUM SEVERITY ISSUES:")
        print("-" * 70)
        for file_path, issue in medium_severity:
            rel_path = get_relative_path(file_path)
            print(f"\n  📄 {rel_path}:{issue.line_number}")
            print(f"     [{issue.issue_type}] {issue.description}")
            if issue.suggestion:
                print(f"     Fix: {issue.suggestion}")

    if low_severity and verbose:
        print("\n🟢 LOW SEVERITY ISSUES (review recommended):")
        print("-" * 70)
        for file_path, issue in low_severity:
            rel_path = get_relative_path(file_path)
            print(f"\n  📄 {rel_path}:{issue.line_number}")
            print(f"     [{issue.issue_type}] {issue.description}")

    if files_with_errors and verbose:
        print("\n⚠️  FILES WITH ERRORS:")
        print("-" * 70)
        for result in files_with_errors:
            print(f"  ✗ {get_relative_path(result.file_path)}: {result.error}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total files scanned: {len(results)}")
    print(f"  Files analyzed by AI: {len(files_clean) + len(files_with_issues)}")
    print(f"  Files with issues: {len(files_with_issues)}")
    print(f"    - High severity: {len(high_severity)}")
    print(f"    - Medium severity: {len(medium_severity)}")
    print(f"    - Low severity: {len(low_severity)}")
    print(f"  Files skipped: {len(files_skipped)}")
    print(f"  Files clean: {len(files_clean)}")
    if files_with_errors:
        print(f"  Files with errors: {len(files_with_errors)}")

    if files_with_issues:
        print("\n" + "-" * 70)
        print("HOW TO FIX:")
        print("-" * 70)
        print("  1. Remove or replace sensitive data with placeholders")
        print("  2. Use environment variables for secrets")
        print("  3. Add files to .gitignore if they contain sensitive data")
        print("  4. Add #skip-sensitive-check comment for false positives")

    return len(high_severity) + len(medium_severity)


# ==============================================================================
# Command: check
# ==============================================================================


async def cmd_check_async(
    directories: list[Path],
    specific_files: list[str] | None = None,
    verbose: bool = False,
    max_concurrency: int = MAX_CONCURRENCY,
) -> int:
    """Execute the check command with AI analysis."""
    print_header("AI-Powered Sensitive Information Check")

    if specific_files:
        files = [Path(f) for f in specific_files]
        print(f"Checking {len(files)} specified file(s)...")
    else:
        print(f"Scanning directories: {[str(d) for d in directories]}")
        files = get_files_from_directories(directories)
        print(f"Found {len(files)} file(s) to check...")

    # Filter out files that should be skipped (before AI analysis)
    files_to_analyze = []
    skipped_results = []

    for file_path in files:
        should_skip, reason = should_skip_file(str(file_path))
        if should_skip:
            skipped_results.append(
                FileCheckResult(
                    file_path=str(file_path), skipped=True, skip_reason=reason
                )
            )
        else:
            # Check file size
            if file_path.exists():
                file_size = file_path.stat().st_size
                if file_size > MAX_FILE_SIZE:
                    skipped_results.append(
                        FileCheckResult(
                            file_path=str(file_path),
                            skipped=True,
                            skip_reason=f"File too large ({file_size / 1024:.1f}KB)",
                        )
                    )
                elif file_size == 0:
                    skipped_results.append(
                        FileCheckResult(
                            file_path=str(file_path),
                            skipped=True,
                            skip_reason="Empty file",
                        )
                    )
                else:
                    files_to_analyze.append(file_path)

    print(f"Files to analyze with AI: {len(files_to_analyze)}")
    print(f"Files skipped: {len(skipped_results)}")
    print(f"Max concurrency: {max_concurrency}")
    print()

    if not files_to_analyze:
        print("No files to analyze!")
        return 0

    # Load LLM provider
    print("Loading LLM provider...")
    provider = _load_llm_provider()
    print()

    # Analyze files concurrently
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        analyze_file_with_ai(
            provider,
            str(file_path),
            semaphore,
            index=idx + 1,
            total=len(files_to_analyze),
        )
        for idx, file_path in enumerate(files_to_analyze)
    ]

    analyzed_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    all_results = skipped_results.copy()
    for result in analyzed_results:
        if isinstance(result, Exception):
            all_results.append(FileCheckResult(file_path="unknown", error=str(result)))
        else:
            all_results.append(result)

    issue_count = print_results(all_results, verbose)

    if issue_count > 0:
        print(f"\n❌ Found {issue_count} potential sensitive information issue(s)")
        return 1
    else:
        print("\n✅ No sensitive information detected!")
        return 0


def cmd_check(
    directories: list[Path],
    specific_files: list[str] | None = None,
    verbose: bool = False,
    max_concurrency: int = MAX_CONCURRENCY,
) -> int:
    """Execute the check command."""
    return asyncio.run(
        cmd_check_async(
            directories=directories,
            specific_files=specific_files,
            verbose=verbose,
            max_concurrency=max_concurrency,
        )
    )


# ==============================================================================
# Command: hook (pre-commit hook)
# ==============================================================================


async def cmd_hook_async(files: list[str]) -> int:
    """Execute the hook command for pre-commit with AI analysis."""
    if not files:
        return 0

    # Filter files
    files_to_check = []
    for file_path in files:
        should_skip, _ = should_skip_file(file_path)
        if not should_skip and Path(file_path).exists():
            file_size = Path(file_path).stat().st_size
            if 0 < file_size <= MAX_FILE_SIZE:
                files_to_check.append(file_path)

    if not files_to_check:
        return 0

    # Load LLM provider
    provider = _load_llm_provider()

    # Analyze files
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = [
        analyze_file_with_ai(provider, file_path, semaphore)
        for file_path in files_to_check
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect issues by severity
    blocking_issues = []  # HIGH and MEDIUM - will block commit
    warning_issues = []  # LOW - just warn, don't block

    for result in results:
        if isinstance(result, FileCheckResult) and result.issues:
            for issue in result.issues:
                if issue.severity in (Severity.HIGH, Severity.MEDIUM):
                    blocking_issues.append((result.file_path, issue))
                elif issue.severity == Severity.LOW:
                    warning_issues.append((result.file_path, issue))

    # Show warnings for LOW severity (but don't block)
    if warning_issues:
        print("\n" + "=" * 70, file=sys.stderr)
        print("⚠️  LOW SEVERITY REMINDERS (commit allowed)", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        for file_path, issue in warning_issues:
            rel_path = get_relative_path(file_path)
            print(
                f"   🟢 {rel_path}:{issue.line_number} [{issue.issue_type}]",
                file=sys.stderr,
            )
            print(f"      {issue.description}", file=sys.stderr)

        print("-" * 70, file=sys.stderr)
        print("These are just reminders. Commit will proceed.", file=sys.stderr)

    # Block on HIGH/MEDIUM severity
    if blocking_issues:
        print("\n" + "=" * 70, file=sys.stderr)
        print("❌ SENSITIVE INFORMATION DETECTED - COMMIT BLOCKED", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        for file_path, issue in blocking_issues:
            rel_path = get_relative_path(file_path)
            severity_icon = "🔴" if issue.severity == Severity.HIGH else "🟡"
            print(
                f"\n   {severity_icon} {rel_path}:{issue.line_number} [{issue.issue_type}]",
                file=sys.stderr,
            )
            print(f"      {issue.description}", file=sys.stderr)
            if issue.suggestion:
                print(f"      Fix: {issue.suggestion}", file=sys.stderr)

        print("\n" + "-" * 70, file=sys.stderr)
        print("TO FIX:", file=sys.stderr)
        print("  1. Remove or replace sensitive data", file=sys.stderr)
        print("  2. Use environment variables for secrets", file=sys.stderr)
        print("  3. Add #skip-sensitive-check to skip specific lines", file=sys.stderr)
        print(
            f"  4. Set {HOOK_SKIP_ENV_VAR}=1 to skip this check entirely",
            file=sys.stderr,
        )
        print("=" * 70 + "\n", file=sys.stderr)

        return 1

    return 0


def cmd_hook(files: list[str], commit_msg: bool = False) -> int:
    """Execute the hook command for pre-commit.

    Args:
        files: List of files to check
        commit_msg: If True, this is a commit-msg hook

    Returns:
        0 if no issues found, 1 if issues found
    """
    # Check for skip environment variable
    if os.environ.get(HOOK_SKIP_ENV_VAR, "").lower() in ("1", "true", "yes"):
        print(f"Skipping sensitive check ({HOOK_SKIP_ENV_VAR} is set)")
        return 0

    if commit_msg:
        # For commit-msg hook, check for skip keywords
        if files:
            try:
                with open(files[0], "r", encoding="utf-8") as f:
                    content = f.read().lower()
                for keyword in SKIP_COMMIT_MSG_KEYWORDS:
                    if keyword.lower() in content:
                        return 0
            except Exception:
                pass
        return 0

    return asyncio.run(cmd_hook_async(files))


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AI-Powered Sensitive Information Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Check command
    check_parser = subparsers.add_parser(
        "check", help="Check files for sensitive information using AI"
    )
    check_parser.add_argument(
        "--directory",
        "-d",
        nargs="*",
        help="Directories to scan (default: src, tests, data, demo)",
    )
    check_parser.add_argument("--files", nargs="*", help="Specific files to check")
    check_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    check_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help=f"Maximum concurrent AI calls (default: {MAX_CONCURRENCY})",
    )

    # Hook command
    hook_parser = subparsers.add_parser(
        "hook", help="Pre-commit hook to check for sensitive information"
    )
    hook_parser.add_argument(
        "--commit-msg",
        action="store_true",
        help="Check commit message instead of files",
    )
    hook_parser.add_argument("files", nargs="*", help="Files to check")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "check":
        directories = resolve_directories(args.directory)
        exit_code = cmd_check(
            directories=directories,
            specific_files=args.files,
            verbose=args.verbose,
            max_concurrency=args.max_concurrency,
        )
        sys.exit(exit_code)

    elif args.command == "hook":
        exit_code = cmd_hook(files=args.files, commit_msg=args.commit_msg)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
