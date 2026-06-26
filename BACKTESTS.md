# Backtest Results

Backtest scope: WC2026 group stage rounds 1 & 2 (48 matches), trained on all data before first WC2026 match.

| run | accuracy | log-loss | n_matches | commit | timestamp | notes |
|-----|----------|----------|-----------|--------|-----------|-------|
| baseline_20260626 | 62% | 0.916 | 48 | 06935b4 | 2026-06-26 | Binary neutral flag, no crowd support |
| crowd_20260626 | 62% | 0.925 | 48 | 1491440 | 2026-06-26 | Continuous home/away crowd factors for WC2026 venues |
| draw_features_20260626 | 62% | 0.924 | 48 | c95d8e5 | 2026-06-26 | Draw rate, closeness, goal environment features; removed redundant home/away individual stats |
| draw_no_crowd_20260626 | 62% | 0.922 | 48 | ff736c2 | 2026-06-26 | Draw tendency and match balance features using baseline neutral handling, no crowd support |
