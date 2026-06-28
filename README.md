# TabPFN World Cup Predictions

Reproducible submission pipeline for the Prior Labs World Cup outcome prediction
competition. Predicts regulation-time outcomes (home win / draw / away win) using
a 3-class TabPFN classifier.

## Data

- `results.csv` — historical international results and upcoming fixtures
  ([martj42/international_results](https://github.com/martj42/international_results))
- `goalscorers.csv` — used to correct extra-time-inflated scores back to 90-minute results

## Reproduce

```bash
uv sync
uv run predict.py --draw-scale 1.2 --date <YYYY-MM-DD> --output-dir submissions
```

---

## Submissions

### Round of 32 (2026-06-28 – 2026-07-03)

**File:** `submissions/submission_main_ds12_20260628.csv`

```bash
uv run predict.py --draw-scale 1.2 --date 2026-06-28 --output-dir submissions
```
