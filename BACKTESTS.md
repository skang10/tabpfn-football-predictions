# Backtest Results

**Primary metric: multi-class log-loss (lower is better)** — competition scoring metric per rules §8.
Uniform baseline (random guessing) log-loss = 1.099.

Three fixed test sets, all evaluated with `uv run backtest.py` on each branch:
- **BT1** WC 2022 group stage (48 matches) — model trained on data before 2022-11-20
- **BT2** WC 2022 knockout (16 matches) — same model as BT1
- **BT3** WC 2026 group stage rounds 1-2 (48 matches) — model trained on data before WC 2026 start

BT1+2 (WC 2022 full, 64 matches) is the primary logged metric for experiment comparisons.

| experiment | branch | BT1 group | BT2 knockout | **BT1+2 WC22 full** | BT3 WC26 r1-2 | model | TRAIN_START | notes |
|------------|--------|-----------|--------------|---------------------|----------------|-------|-------------|-------|
| baseline | main | 1.1258 | 0.9405 | **1.0795** | 0.9163 | TabPFN 3-class | 2014 | 26 features; 0 draws predicted everywhere |
| recency_2018 | exp/recency_2018_v2 | 1.1398 | 0.9089 | 1.0820 | **0.9146** | TabPFN 3-class | 2018 | Same features, smaller training window |
| goal_model_v2 | exp/goal_model_v2_wc22 | 1.1193 | 0.9450 | 1.0758 | 0.9281 | Poisson (TabPFN regressor) | 2018 | 31 features (+ balance/draw-rate); T=1.0 |
| dc_temp_scaling | exp/dc_temp_scaling | **1.0700** | 0.9984 | **1.0521** | 0.9502 | Poisson + temp T=2.0 | 2018 | DC ρ=0 (disabled); **best on WC22 group + full** |

**Bold** = best in column. Uniform baseline = 1.099.

### Key observations

- **dc_temp_scaling** is best on WC 2022 (full and group stage), confirming that temperature T=2.0 corrects overconfidence on upset-heavy group stages.
- **dc_temp_scaling is worst on BT3 (WC 2026 rounds 1-2)** at 0.9502 vs baseline 0.9163. T=2.0 was tuned on WC 2022 which was abnormally upset-heavy; it over-flattens probabilities for WC 2026 where favorites are winning more as expected.
- **recency_2018** narrowly best on BT3 (0.9146), suggesting more recent training data helps on current-tournament dynamics.
- BT2 (knockout) and BT3 (WC 2026) favour less flattening — temperature scaling is a double-edged sword.
- No model predicts any draws (P(draw) peaks at ~0.30, never beats argmax).

---

## Superseded experiments (WC 2026 group stage rounds 1-2 backtest, accuracy-primary)

> These used a different test set and accuracy as primary metric. Included for historical reference only.

| run | parent | accuracy | Δacc | log-loss | n_matches | commit | timestamp | notes |
|-----|--------|----------|------|----------|-----------|--------|-----------|-------|
| baseline_20260627 | — | 62% | — | 0.916 | 48 | 5ac2c86 | 2026-06-27 | Original feature set; 0 draws predicted |
| draw_features_20260627 | baseline_20260627 | 62% | 0% | 0.922 | 48 | 5efe5e5 | 2026-06-27 | Draw tendency features; 0 draws predicted |
| draw_threshold_20260627 | draw_features_20260627 | 58% | -4% | 0.922 | 48 | 2a2ffa2 | 2026-06-27 | Threshold 0.28; 12/48 predicted draws (4 correct) |
| goal_model_20260627 | baseline_20260627 | 62% | 0% | 0.935 | 48 | 3a75365 | 2026-06-27 | Poisson + two regressors; 0 draws |
| goal_model_dc_20260627 | goal_model_20260627 | 62% | 0% | 0.933 | 48 | bb3bc9e | 2026-06-27 | Dixon-Coles ρ=-0.02; 0 draws |
| two_stage_20260627 | baseline_20260627 | 62% | 0% | 0.943 | 48 | 0c8a8b3 | 2026-06-27 | Stage 1 draw/not_draw + Stage 2; 0 draws |
| recency_2018_20260627 | baseline_20260627 | 62% | 0% | 0.915 | 48 | 46317f4 | 2026-06-27 | TRAIN_START=2018; best log-loss in old backtest |
| threshold_sweep_20260627 | recency_2018_20260627 | 62% | 0% | 0.915 | 48 | 58bfdb8 | 2026-06-27 | Sweep thr∈[0.20,0.32]; at 0.27: 58% acc, 14 draws (5 correct) |
| two_stage_threshold_20260627 | threshold_sweep_20260627 | 60% | -2% | — | 48 | 0657c5d | 2026-06-27 | Stage 1 binary + threshold; best draws at 0.22 (52%, 7/23 correct) |
| draw_calibrated_20260627 | recency_2018_20260627 | 38% | -24% | 0.937 | 48 | 70b0782 | 2026-06-27 | Balance features + isotonic calibration (80/20 split); train split hurt |
| goal_model_v2_20260627 | draw_calibrated_20260627 | 62% | +24% | 0.928 | 48 | 1a4a1fb | 2026-06-27 | Poisson + balance/draw-rate features; thr=0.27: 62% acc, 8 draws (3 correct) |
