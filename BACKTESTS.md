# Backtest Results

**Primary metric: multi-class log-loss (lower is better)** — competition scoring metric per §8.
Uniform baseline (random guessing): log-loss = 1.099.

Three fixed test sets evaluated with `uv run backtest.py` on each branch:

| ID | Test set | Matches | Training pool cutoff |
|----|----------|---------|----------------------|
| BT1 | WC 2022 group stage | 48 | 2022-11-20 |
| BT2 | WC 2022 knockout | 16 | 2022-11-20 |
| BT3 | WC 2026 group stage rounds 1–2 | 48 | 2026-06-11 |

**Bold** = best log-loss in column &nbsp;·&nbsp; <u>Underline</u> = best accuracy in column.

---

## BT1 — WC 2022 Group Stage (48 matches)

| experiment | log-loss | accuracy | commit | date | notes |
|------------|:--------:|:--------:|--------|------|-------|
| baseline | **1.1258** | <u>50.0%</u> | 3d45aeb | 2026-06-27 | TabPFN 3-class, TRAIN_START=2014, 26 features |

---

## BT2 — WC 2022 Knockout (16 matches)

| experiment | log-loss | accuracy | commit | date | notes |
|------------|:--------:|:--------:|--------|------|-------|
| baseline | **0.9405** | <u>56.2%</u> | 3d45aeb | 2026-06-27 | TabPFN 3-class, TRAIN_START=2014, 26 features |

---

## BT3 — WC 2026 Group Stage Rounds 1–2 (48 matches)

| experiment | log-loss | accuracy | commit | date | notes |
|------------|:--------:|:--------:|--------|------|-------|
| baseline | **0.9163** | <u>62.5%</u> | 3d45aeb | 2026-06-27 | TabPFN 3-class, TRAIN_START=2014, 26 features |

---

## Superseded (old schema — WC 2026 group stage, accuracy-primary)

> Different test set, accuracy as primary metric. Historical reference only.

| run | accuracy | log-loss | n | commit | date | notes |
|-----|:--------:|:--------:|:-:|--------|------|-------|
| baseline | 62% | 0.916 | 48 | 5ac2c86 | 2026-06-27 | Original feature set; 0 draws predicted |
| draw_features | 62% | 0.922 | 48 | 5efe5e5 | 2026-06-27 | Draw tendency features |
| draw_threshold | 58% | 0.922 | 48 | 2a2ffa2 | 2026-06-27 | Threshold 0.28; 12/48 draws predicted |
| goal_model | 62% | 0.935 | 48 | 3a75365 | 2026-06-27 | Poisson + two regressors |
| goal_model_dc | 62% | 0.933 | 48 | bb3bc9e | 2026-06-27 | Dixon-Coles ρ=-0.02 |
| two_stage | 62% | 0.943 | 48 | 0c8a8b3 | 2026-06-27 | Stage 1 draw/not_draw + Stage 2 |
| recency_2018 | 62% | 0.915 | 48 | 46317f4 | 2026-06-27 | TRAIN_START=2018 |
| threshold_sweep | 62% | 0.915 | 48 | 58bfdb8 | 2026-06-27 | Sweep thr∈[0.20,0.32] |
| two_stage_threshold | 60% | — | 48 | 0657c5d | 2026-06-27 | Stage 1 binary + threshold |
| draw_calibrated | 38% | 0.937 | 48 | 70b0782 | 2026-06-27 | Isotonic calibration; train split hurt |
| goal_model_v2 | 62% | 0.928 | 48 | 1a4a1fb | 2026-06-27 | Poisson + balance/draw-rate features |
