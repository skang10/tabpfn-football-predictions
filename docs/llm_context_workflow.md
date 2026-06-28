# LLM Context Workflow

This workflow creates reproducible LLM-derived context features for `predict.py`.
The LLM never predicts match probabilities. It only extracts structured team
context from source notes saved in this repository.

## Files

- `prompts/llm_context_extractor.md`: versioned extraction prompt.
- `schemas/llm_context.schema.json`: strict JSON schema for model output.
- `llm_sources/*.md`: one source note per fixture. These are the LLM inputs.
- `llm_context.jsonl`: generated structured context rows.
- `llm_context.csv`: CSV export used by `predict.py` by default.
- `llm_context_audit.jsonl`: prompt/schema/source hashes plus raw API response.

## Create source-note templates

```bash
uv run python scripts/generate_llm_context.py init-sources --from-date 2026-06-28
```

Each generated source note starts as `status: todo`. Fill it with short factual
pre-match notes, URLs, publication times, and retrieval times. Do not paste long
article text. Change the front matter to `status: ready` when the note is ready
for extraction.

## Generate context

```bash
OPENAI_API_KEY=... uv run python scripts/generate_llm_context.py generate --from-date 2026-06-28
```

Optional model override:

```bash
OPENAI_MODEL=gpt-5.5 uv run python scripts/generate_llm_context.py generate --from-date 2026-06-28
```

Dry run without calling the API:

```bash
uv run python scripts/generate_llm_context.py generate --from-date 2026-06-28 --dry-run --include-todo --limit 1
```

## Validate

```bash
uv run python scripts/generate_llm_context.py validate llm_context.jsonl
```

## Reproducibility rules

- Commit the prompt, schema, source notes, generated context, and audit log used
  for a submission.
- Use only pre-kickoff information in source notes.
- Store URLs and timestamps for every material source.
- If a source note has weak or missing evidence, leave the relevant scores at 0
  and let `llm_confidence` stay low.
- For historical backtests, only use source notes created from pre-kickoff
  archived material. Do not ask an LLM to reconstruct old team news from memory.

## Feature directions

`predict.py` reads the generated raw columns and computes:

- `absence_diff = away_absence_severity - home_absence_severity`
- `lineup_uncertainty_diff = away_lineup_uncertainty - home_lineup_uncertainty`
- `rotation_risk_diff = away_rotation_risk - home_rotation_risk`
- `tactical_edge_diff = home_tactical_edge - away_tactical_edge`

Positive values favor the home team.
