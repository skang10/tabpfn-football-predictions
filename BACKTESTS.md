# Backtest Results

**Primary metric: multi-class log-loss (lower is better)** — competition scoring metric per rules §8.
Backtest scope: WC 2022 full tournament (64 matches, group stage + knockouts), trained on all data before 2022-11-20.
Uniform baseline log-loss = 1.099 (0.333/0.333/0.333 for all matches).

> Prior experiments below were evaluated on WC 2026 group stage rounds 1-2 (48 matches) and used **accuracy** as primary metric — superseded.

| run | log-loss | Δloss | accuracy | n_matches | commit | timestamp | notes |
|-----|----------|-------|----------|-----------|--------|-----------|-------|
| baseline_wc2022_20260627 | 1.0795 | — | 51.6% | 64 | 4de9d1d | 2026-06-27 | 3-class TabPFN, TRAIN_START=2014; 15 actual draws / 0 predicted |
| recency_2018_v2_20260627 | 1.0820 | +0.0025 | 51.6% | 64 | 6790f90 | 2026-06-27 | TRAIN_START=2018 only; fewer historical matches hurts on WC2022 |
| goal_model_v2_wc22_20260627 | **1.0758** | **-0.0037** | 51.6% | 64 | 2c44434 | 2026-06-27 | Poisson regressors + balance features + TRAIN_START=2018; **best log-loss so far** |

---

## Superseded experiments (WC 2026 group stage backtest, accuracy-primary)

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
