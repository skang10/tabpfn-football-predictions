# Experiment Ideas

Goal: maximise accuracy on WC2026 group stage predictions for the PriorLabs competition.
Current best: 62% accuracy, 0.916 log-loss (baseline_20260626, 48 matches).

## Prioritised Ideas

### 1. Draw tendency + match balance features [x]
Replace directional "who's stronger" features with features that answer "is this likely to stay unresolved?".
- Track `drew` in `res` entries → `home_draw_rate10`, `away_draw_rate10`, `combined_draw_rate10`, `draw_rate_diff`
- Add non-directional closeness: `elo_abs_diff`, `form_abs_diff`, `gd_abs_diff`, `elo_closeness`
- Add low-scoring profile: `goal_environment`, `defensive_tightness`, `attack_abs_diff`
- Remove redundant directional features: `home_elo`, `away_elo`, `home_form5`, `away_form5`, `home_winrate`, `away_winrate`, `h2h_home_winrate`

### 2. Round 3 qualification need-state [ ]
Biggest predictor in round 3 isn't ELO — it's what each team needs to qualify.
Add `home_points`, `away_points` (current WC group stage points) and derived
`qualification_need` (win/draw/irrelevant) for each team before round 3 matches.
Gentleman's agreement draws are nearly deterministic when both teams need a draw.

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
