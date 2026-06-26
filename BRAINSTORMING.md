# Experiment Ideas

Goal: maximise accuracy on WC2026 group stage predictions for the PriorLabs competition.
Current best: 62% accuracy, 0.916 log-loss (baseline_20260626, 48 matches).

## Prioritised Ideas

### 1. Fix draw prediction [ ]
Biggest gap — model almost never predicts draws despite ~25% base rate in WC group stage.
- Add `elo_closeness = 1 / (1 + abs(elo_diff))` — high when teams are evenly matched
- Add `home_draw_rate` / `away_draw_rate` — each team's historical draw frequency

### 2. Remove redundant features [ ]
`elo_diff` = `home_elo - away_elo + adj`, so all three together triple-weight that signal.
Same issue with `form5_diff` + `home_form5` + `away_form5`.
Keep only diffs OR only individual features, not both.

### 3. Split form by venue type [ ]
Current `form5` mixes home and away results. For a neutral-venue tournament, away form
is more predictive. Split into `home_form_home` / `home_form_away` etc.

### 4. Recency weighting [ ]
Currently `tail(MAX_TRAIN)` keeps the last 10k matches regardless of age.
Try filtering training to post-2018 or post-2022 — international football changed
significantly after COVID and the new formats.

### 5. WC experience feature [ ]
Number of World Cups a team has participated in. Captures tournament pressure handling
that form stats miss entirely.

## Completed

| experiment | accuracy | log-loss | notes |
|------------|----------|----------|-------|
| baseline_20260626 | 62% | 0.916 | Binary neutral flag, no crowd support |
| crowd_20260626 | 62% | 0.925 | Continuous home/away crowd factors for WC2026 venues |
