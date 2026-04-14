# src/skill_evolution/ — Self-Evolution Method Evaluation

This directory contains evaluation implementations for different agent self-evolution methods. Each subdirectory corresponds to a method with its own scripts, configurations, and documentation.

## Evaluation Paradigm

All methods share the same evaluation framework (`src/domains/`) and support any registered domain. Two evaluation modes are supported:

### Offline Mode

1. **Training** — Run agent on train split, collect session trajectories
2. **Extraction** — Batch-extract reusable knowledge (skills) from train sessions
3. **Evaluation** — Compare on test split: baseline (no skills) vs with skill injection

### Online Mode

Extract skills immediately after each task completes and update the knowledge base. Subsequent tasks can leverage previously accumulated skills, enabling learn-as-you-go evaluation.

## Split File Format

Two formats are supported:

**Cluster format** (for data with natural categories, e.g. BrowseComp-Plus topics):

```json
{
  "clusters": {
    "CLUSTER_A": {"train": ["id1", "id2"], "test": ["id3", "id4"]},
    "CLUSTER_B": {"train": [...], "test": [...]}
  }
}
```

**Flat format** (for data without clusters, auto-wrapped into a "default" cluster):

```json
{
  "train": ["task1", "task2"],
  "test": ["task3", "task4"]
}
```

## Methods

| Directory | Method | Mode | Description |
|-----------|--------|------|-------------|
| `evermemos/` | EverMemOS | Offline / Online | Extract skills via EverMemOS API, inject into prompt |
| More methods | Coming soon | — | — |

## Adding a New Method

1. Create a new directory under `src/skill_evolution/`
2. Implement knowledge extraction script (offline or online mode)
3. Implement knowledge injection + evaluation script
4. Write a README documenting the full workflow
5. Use the same split file to ensure fair comparison
