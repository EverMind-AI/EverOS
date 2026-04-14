# GDPVal Domain

Evaluates agents on real-world occupational tasks from the [OpenAI GDPVal dataset](https://huggingface.co/datasets/openai/gdpval) (220 tasks across 44 occupations). Agents produce deliverable files (Excel, PDF, Word, etc.) which are scored by an LLM evaluator against occupation-specific rubrics.

## Evaluation

Evaluation follows the [ClawWork](https://github.com/HKUDS/ClawWork) LLMEvaluator approach.

## Dependencies

### Python packages

```bash
pip install datasets python-docx openpyxl pdf2image Pillow
```

### System packages (for evaluation)

The evaluator converts PDF/PPTX deliverables to images for multimodal LLM scoring. This requires:

```bash
# Required for PDF evaluation
apt install poppler-utils

# Optional, for PPTX evaluation
apt install libreoffice
```

Without `poppler-utils`, evaluation of PDF deliverables will fail. Excel, Word, and text-based deliverables are not affected.

### Environment variables

```bash
OPENROUTER_API_KEY=sk-or-v1-...   # Required for LLM evaluation (gpt-4o via OpenRouter)
```

## Data

- **Reference files**: Auto-downloaded from HuggingFace on first run and cached in `data/gdpval/reference_files/`.
- **Meta-prompts**: Per-occupation evaluation rubrics in `data/gdpval/meta_prompts/`. Included in the repo.
- **Cluster splits**: `data/gdpval/clusters_397b_*.json` define train/test splits for skill extraction experiments.

## Usage

```bash
# Run a single task
python3 run.py --task 83d10b06 --live

# Run all tasks
python3 run.py --live
```
