# TabPFN World Cup Predictions

Reproducible submission pipeline for the Prior Labs World Cup outcome prediction
competition. Predicts regulation-time outcomes (home win / draw / away win) using
a 3-class TabPFN classifier.

---

## Submissions

### Semifinal (2026-07-14 – 2026-07-15)

**Data:** `results.csv` + `goalscorers.csv` from
[martj42/international_results](https://github.com/martj42/international_results),
refreshed at submission time (`--refresh`).

```bash
uv sync
uv run predict.py --refresh --output-dir submissions
```

### Quarterfinal blind check (pre-2026-07-09)

**Data:** `results.csv` + `goalscorers.csv` from
[martj42/international_results](https://github.com/martj42/international_results),
refreshed at submission time (`--refresh`).

Retroactively reconstructs what the model would have predicted for all four
Quarterfinal matches (including France vs Morocco) using only data available
before 2026-07-09 — useful for validating a call after the fact without the
result leaking into training. `--as-of` caps the training pool to matches
strictly before the given date and treats every fixture on/after it as still
unplayed, even once a real result has since been recorded (`--date` alone
can't do this, since it only filters *which* unplayed fixtures to output):

```bash
uv sync
uv run predict.py --refresh --as-of 2026-07-09 --output-dir submissions
```

### Round of 16 (2026-07-04 – 2026-07-06)

**Data:** `results.csv` + `goalscorers.csv` from
[martj42/international_results](https://github.com/martj42/international_results),
refreshed at submission time (`--refresh`).

Draw scaling is now **conditional on match closeness** and baked into the model
(`predict_proba`), so no `--draw-scale` flag is needed:

```bash
uv sync
uv run predict.py --refresh --output-dir submissions
```

### Round of 32 (2026-06-28 – 2026-07-03)

**Data:** `results.csv` + `goalscorers.csv` from
[martj42/international_results](https://github.com/martj42/international_results)
as of 2026-06-28.

```bash
uv sync
uv run predict.py --draw-scale 1.2 --date 2026-06-28 --output-dir submissions
```
