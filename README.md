# TabPFN World Cup Predictions

Reproducible submission pipeline for the Prior Labs World Cup outcome prediction
competition. Predicts regulation-time outcomes (home win / draw / away win) using
a 3-class TabPFN classifier.

---

## Submissions

### Elo momentum on the R32 base (experimental)

**Data:** `results.csv` + `goalscorers.csv` from
[martj42/international_results](https://github.com/martj42/international_results),
refreshed at submission time (`--refresh`).

Adds `elo_mom5_diff` — each team's Elo rating change over its last 5 games,
already opponent-strength- and margin-adjusted since the Elo delta itself is
`importance × goal-diff multiplier × (actual − expected)` — directly on top
of the Round of 32 base model (20 features, flat `--draw-scale`), rather than
on the later `wc_recent_form`/`cond_draw_elo` stack:

```bash
uv sync
uv run predict.py --draw-scale 1.2 --refresh --output-dir predictions
```

### Round of 32 (2026-06-28 – 2026-07-03)

**Data:** `results.csv` + `goalscorers.csv` from
[martj42/international_results](https://github.com/martj42/international_results)
as of 2026-06-28.

```bash
uv sync
uv run predict.py --draw-scale 1.2 --date 2026-06-28 --output-dir submissions
```
