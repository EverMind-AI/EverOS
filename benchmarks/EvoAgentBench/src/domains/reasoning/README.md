# Omni-Math Domain

Math competition problems. Tests whether agents can solve olympiad-level math problems.

Skill injection is handled separately in `src/skill_evolution/evermemos/` — this domain provides base evaluation only.

## Dataset

100 competition-level math problems (`test_set_100`) covering 10 subdisciplines (Algebra, Combinatorics, Geometry, Probability, Number Theory, etc.).

## Data Format

Each test problem (in `test.jsonl`) contains:
- `problem`: LaTeX-formatted math competition problem
- `answer`: Expected answer
- `solution`: Step-by-step reference solution
- `domain`: Mathematical domain classification
- `difficulty`: Rating from 1.0 to 9.0
- `source`: Competition source (e.g., `HMMT_11`, `jbmo_shortlist`)
- `problem_type`: Category (e.g., `Diophantine Equations`)
- `test_category`: Classification (`both_improved`, `both_fail`, etc.)

## Dependencies

No additional system packages required.

**Python packages:** (included in project requirements)
- `openai` (only needed for LLM verification mode)

## Environment Variables

- `OPENROUTER_API_KEY` — Required when `verify_mode: llm` and no `eval_api_base` is set

## Usage

```bash
# Run all 100 problems (uses agent from config.yaml)
python src/run.py --domain omnimath

# Run first N problems
python src/run.py --domain omnimath --split 10

# Run specific problems
python src/run.py --domain omnimath --task 2080,557
```

To switch agents, change `agent.config` in `config.yaml`.

## With Skill Injection

Skill injection is in `src/skill_evolution/evermemos/`. See [skill_evolution/evermemos/README.md](../../skill_evolution/evermemos/README.md).

```bash
python src/skill_evolution/evermemos/eval_with_skills.py --domain omnimath --split test
```

## Configuration

See `omnimath.yaml` for all settings:
- `verify_mode`: `exact` (string matching) or `llm` (LLM judge)
- `eval_model_owner` / `eval_model_name`: Judge model for LLM verification
- `eval_api_base`: Local vLLM endpoint for judge (if not using OpenRouter)
- `agent_timeout`: Seconds per problem (default: 600)

## Verification Modes

- **exact** — Normalizes LaTeX (`\frac{a}{b}` → `a/b`, strip `\boxed{}`, etc.) and compares strings. Falls back to numeric comparison. Fast, free, but brittle for equivalent forms.
- **llm** — Sends problem + both answers to a judge model which responds CORRECT/INCORRECT. Handles mathematical equivalence but costs API calls.
