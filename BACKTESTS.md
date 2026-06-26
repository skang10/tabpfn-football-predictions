# Backtest Results

Backtest scope: WC2026 group stage rounds 1 & 2 (48 matches), trained on all data before first WC2026 match.

| run | accuracy | log-loss | n_matches | commit | timestamp | notes |
|-----|----------|----------|-----------|--------|-----------|-------|
| baseline_20260626 | 62% | 0.916 | 48 | 06935b4 | 2026-06-26 | Binary neutral flag, no crowd support |
| crowd_20260626 | 62% | 0.925 | 48 | 1491440 | 2026-06-26 | Continuous home/away crowd factors for WC2026 venues |
