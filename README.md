# TabPFN World Cup Predictions

Reproducible submission pipeline for the Prior Labs World Cup outcome prediction
competition. Predicts regulation-time outcomes (home win / draw / away win) using
a 3-class TabPFN classifier.

---

## Submissions

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
