# Backtest Results

Backtest scope: WC2026 group stage rounds 1 & 2 (48 matches), trained on all data before first WC2026 match.

| run | parent | accuracy | Δacc | log-loss | Δloss | n_matches | commit | timestamp | notes |
|-----|--------|----------|------|----------|-------|-----------|--------|-----------|-------|
| baseline_20260626 | — | 62% | — | 0.916 | — | 48 | 06935b4 | 2026-06-26 | Binary neutral flag, no crowd support |
| crowd_20260626 | baseline_20260626 | 62% | 0% | 0.925 | +0.009 | 48 | 1491440 | 2026-06-26 | Continuous home/away crowd factors for WC2026 venues |
| draw_features_20260626 | crowd_20260626 | 62% | 0% | 0.924 | +0.008 | 48 | c95d8e5 | 2026-06-26 | Draw rate, closeness, goal environment features; removed redundant home/away individual stats |
| draw_no_crowd_20260626 | baseline_20260626 | 62% | 0% | 0.922 | +0.006 | 48 | ff736c2 | 2026-06-26 | Draw tendency and match balance features using baseline neutral handling, no crowd support |
| neutral_draw_20260626 | draw_no_crowd_20260626 | 52% | -10% | 0.930 | +0.008 | 48 | 978122e | 2026-06-26 | Neutral-venue training only (3653 matches) + draw threshold 0.25; predicts too many draws (42% vs 29% actual) |
| threshold_draw_20260626 | draw_no_crowd_20260626 | 58% | -4% | 0.922 | 0% | 48 | e1dff85 | 2026-06-26 | Draw threshold 0.28 on all-data training; predicts 12/48 draws (4 correct), hurts accuracy vs decisive wins |
