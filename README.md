# TabPFN World Cup Predictions

Reproducible submission pipeline for the Prior Labs World Cup outcome prediction
competition. Predicts regulation-time outcomes (home win / draw / away win) using
a 3-class TabPFN classifier.

---

## Strategy

| Round | Commit | Strategy |
|---|---|---|
| Round of 32 | `51f1fd4` | TabPFN 3-class classifier, `lightweight_max3000` feature set, flat draw multiplier (`--draw-scale 1.2`) |
| Round of 16 | `bf801c9` | + conditional draw scaling (`draw_k = 1 + 0.4·exp(−\|elo_diff\|/150)`, baked into `predict_proba`, no CLI flag) + recency/in-tournament form features (`form3_diff`, `ewform_diff`, `momentum_diff`) |
| Quarterfinal blind check | `78329cd` | Same as Round of 16, re-run with `--as-of` to reconstruct pre-result predictions |
| Semifinal | `bf801c9` | Same as Round of 16 |
| Third-place playoff | `bf801c9` | Same as Round of 16 |

## Log loss (cumulative)

Competition log loss so far, accumulated match-by-match across submitted rounds
(the Quarterfinal blind check is excluded — it's a retroactive validation, not
a leaderboard submission):

| Round | Matches | Cumulative matches | Cumulative log loss |
|---|---|---|---|
| Round of 32 | 16 | 16 | 0.7949 |
| Round of 16 | 8 | 24 | 0.8404 |
| Semifinal | 2 | 26 | 0.8634 |
| Third-place playoff | 1 | 27 | 0.8850 |

## Submissions

### Third-place playoff (2026-07-18)

**Data:** `results.csv` + `goalscorers.csv` from
[martj42/international_results](https://github.com/martj42/international_results),
refreshed at submission time (`--refresh`).

Only the third-place match (France vs England) is submitted this round. Same
strategy as Round of 16.

```bash
git checkout bf801c9
uv sync
uv run predict.py --refresh --output-dir submissions
```

### Semifinal (2026-07-14 – 2026-07-15)

**Data:** `results.csv` + `goalscorers.csv` from
[martj42/international_results](https://github.com/martj42/international_results),
refreshed at submission time (`--refresh`).

Same strategy as Round of 16.

```bash
git checkout bf801c9
uv sync
uv run predict.py --refresh --output-dir submissions
```

### Quarterfinal blind check (2026-07-09 – 2026-07-11)

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
git checkout 78329cd
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
git checkout bf801c9
uv sync
uv run predict.py --refresh --output-dir submissions
```

### Round of 32 (2026-06-28 – 2026-07-03)

**Data:** `results.csv` + `goalscorers.csv` from
[martj42/international_results](https://github.com/martj42/international_results)
as of 2026-06-28.

```bash
git checkout 51f1fd4
uv sync
uv run predict.py --draw-scale 1.2 --date 2026-06-28 --output-dir submissions
```
