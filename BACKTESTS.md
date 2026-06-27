# Backtest Results

Backtest scope: WC2026 group stage rounds 1 & 2 (48 matches), trained on all data before first WC2026 match.

| run | parent | accuracy | Δacc | log-loss | Δloss | n_matches | commit | timestamp | notes |
|-----|--------|----------|------|----------|-------|-----------|--------|-----------|-------|
| baseline_20260627 | — | 62% | — | 0.916 | — | 48 | 5ac2c86 | 2026-06-27 | Original feature set (ec1282f) with WC2026 backtest infrastructure; 0 draws predicted |
| draw_features_20260627 | baseline_20260627 | 62% | 0% | 0.922 | +0.006 | 48 | 5efe5e5 | 2026-06-27 | Draw tendency + match balance + goal environment features; pure argmax; 0 draws predicted |
| draw_threshold_20260627 | draw_features_20260627 | 58% | -4% | 0.922 | 0% | 48 | 2a2ffa2 | 2026-06-27 | Draw threshold 0.28 on draw features; 12/48 predicted draws (4 correct) |
| goal_model_20260627 | baseline_20260627 | 62% | 0% | 0.935 | +0.019 | 48 | 3a75365 | 2026-06-27 | Two TabPFN regressors (home/away goals) + Poisson simulation; 0 draws predicted |
| goal_model_dc_20260627 | goal_model_20260627 | 62% | 0% | 0.933 | -0.002 | 48 | bb3bc9e | 2026-06-27 | Dixon-Coles correction; fitted ρ=-0.02 (negligible); 0 draws predicted |
| two_stage_20260627 | baseline_20260627 | 62% | 0% | 0.943 | +0.027 | 48 | 0c8a8b3 | 2026-06-27 | Stage 1 draw/not_draw + Stage 2 home/away; max P(draw)=0.30; 0 draws predicted |
| recency_2018_20260627 | baseline_20260627 | 62% | 0% | 0.915 | -0.001 | 48 | 46317f4 | 2026-06-27 | Train only on post-2018 matches; marginally better log-loss; 0 draws predicted |
| threshold_sweep_20260627 | recency_2018_20260627 | 62% | 0% | 0.915 | 0% | 48 | 58bfdb8 | 2026-06-27 | Sweep thr∈[0.20,0.32]; at 0.27: 58% acc, 14 predicted draws (5 correct, 36% hit); at 0.32: 62%, 0 draws |
| two_stage_threshold_20260627 | threshold_sweep_20260627 | 60% | -2% | — | — | 48 | 0657c5d | 2026-06-27 | Stage 1 draw/not_draw binary + thr sweep; P(draw) lower than 3-class; best at thr=0.28 (60%, 2 draws, 0 correct) |
