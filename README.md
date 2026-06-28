# TabPFN World Cup Predictions

This repository contains my reproducible submission pipeline for the Prior Labs
World Cup outcome prediction competition.

## Prediction Time

Current submission target: **2026 FIFA World Cup Round of 32**
(`2026-06-28` to `2026-07-03` fixtures).

Prediction file generated on: **2026-06-28**.

For later rounds, such as quarter-finals and beyond, update `results.csv` with
the new fixtures and rerun the command below.

## Prediction Objective

Predict the regulation-time outcome of each fixture:

- home win
- draw
- away win

The submitted probabilities are produced by a 3-class TabPFN classifier. Draw
probabilities are multiplied by `1.2` and renormalized for knockout submissions.

## Data Used

Required local data files:

- `results.csv`: historical international results and upcoming fixtures
- `goalscorers.csv`: used to correct extra-time-inflated knockout scores back to
  90-minute scores
- `llm_context.csv`: cached LLM-extracted context, required for the
  `--llm-context` run
- `llm_sources/*.md`, `prompts/`, `schemas/`, `llm_context_audit.jsonl`:
  reproducibility artifacts for the LLM context extraction

The base match data comes from the `martj42/international_results` dataset.

## Features Used

Base TabPFN features:

- Elo: `elo_diff`, `home_elo`, `away_elo`
- Recent form: `form5_diff`, `form10_diff`, `home_form5`, `away_form5`
- Rest: `home_rest`, `away_rest`
- Recent team stats: win rate, goals for/against, goal difference
- Streak and experience: `home_streak`, `away_streak`, `home_played`, `away_played`

Optional LLM context features, enabled only with `--llm-context`:

- `absence_diff`
- `lineup_uncertainty_diff`
- `rotation_risk_diff`
- `tactical_edge_diff`

The LLM does not predict probabilities. It only extracts structured team news
from cached source notes.

## Reproduce Submission

Install dependencies:

```bash
uv sync
```

Generate the current LLM-context submission:

```bash
uv run predict.py --draw-scale 1.2 --llm-context --output-dir submissions
```

Generate the base non-LLM submission:

```bash
uv run predict.py --draw-scale 1.2 --output-dir submissions
```

Regenerate LLM context from saved source notes:

```bash
uv run python scripts/generate_llm_context.py generate --from-date 2026-06-28
uv run python scripts/generate_llm_context.py validate llm_context.jsonl
```
