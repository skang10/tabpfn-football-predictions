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

The base match data comes from the `martj42/international_results` dataset.

## Features Used

Base TabPFN features:

- Elo: `elo_diff`, `home_elo`, `away_elo`
- Recent form: `form5_diff`, `form10_diff`, `home_form5`, `away_form5`
- Rest: `home_rest`, `away_rest`
- Recent team stats: win rate, goals for/against, goal difference
- Streak and experience: `home_streak`, `away_streak`, `home_played`, `away_played`

## Reproduce Submission

Install dependencies:

```bash
uv sync
```

Generate the submitted predictions:

```bash
uv run predict.py --draw-scale 1.2 --date 2026-06-28 --output-dir submissions
```
