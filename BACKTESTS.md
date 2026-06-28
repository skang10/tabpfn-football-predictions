# Backtest Results

**Primary metric: multi-class log-loss (lower is better)** — competition scoring metric per §8.
Uniform baseline (random guessing): log-loss = 1.099.

Three fixed test sets evaluated with `uv run backtest.py`:

| ID | Test set | Matches | Training pool cutoff |
|----|----------|---------|----------------------|
| BT1 | WC 2022 group stage | 48 | 2022-11-20 |
| BT2 | WC 2022 knockout | 16 | 2022-11-20 |
| BT3 | WC 2026 group stage rounds 1–2 | 48 | 2026-06-11 |

**Bold** = best log-loss in column &nbsp;·&nbsp; <u>Underline</u> = best accuracy in column

---

## Experiment log

| run | branch | parent | BT1 ll | BT2 ll | BT3 ll | BT1 acc | BT2 acc | BT3 acc | commit | notes |
|-----|--------|--------|:------:|:------:|:------:|:-------:|:-------:|:-------:|--------|-------|
| uniform | main | — | **1.0986** | 1.0986 | 1.0986 | 37.5% | <u>62.5%</u> | 52.1% | 3262c25 | [1/3, 1/3, 1/3] every match |
| always_home | main | — | 2.8858 | 1.7396 | 2.2172 | 37.5% | <u>62.5%</u> | 52.1% | 3262c25 | [0.98, 0.01, 0.01] every match |
| elo_logistic | main | — | 1.1515 | **0.8724** | 0.9123 | 47.9% | 56.2% | <u>62.5%</u> | 3262c25 | multinomial LR on elo_diff only |
| lr_all_features | main | — | 1.1217 | 0.9326 | **0.9000** | 47.9% | 56.2% | 58.3% | 3262c25 | multinomial LR + StandardScaler, 26 features |
| tabpfn_all_features | main | — | 1.1337 | 0.9456 | 0.9121 | 47.9% | 56.2% | <u>62.5%</u> | 3262c25 | TabPFN, 26 features |
| baseline | main | — | 1.1258 | 0.9405 | 0.9163 | <u>50.0%</u> | 56.2% | <u>62.5%</u> | 39550fa | TabPFN, original 26 features, logged via backtest.py |
| wc_context_features | exp_0628/wc_context_features | main | 1.1241 | 0.9321 | 0.9259 | 47.9% | 56.2% | 58.3% | 4222893 | +abs_elo_diff, host_adv_diff, concacaf_adv_diff, same_continent_adv_diff → 30 features |
| symmetric_features | exp_0628/symmetric_features | exp_0628/wc_context_features | 1.1280 | 0.9530 | 0.9290 | 47.9% | 56.2% | 58.3% | ec65f5f | replaced 8 home/away individual features with 4 diffs → 26 features; regressed vs wc_context |

### Key observations

- **BT1 (WC22 group)**: no model beats uniform on log-loss — the group stage is near-unpredictable; wc_context_features is the best non-trivial model here
- **BT2 (WC22 KO)**: elo_logistic dominates; wc_context_features second best among feature-rich models
- **BT3 (WC26 R1-2)**: lr_all_features leads; WC context features hurt slightly vs baseline (model is confused by context features when test data is very recent)
- **Draw recall = 0%** across all runs — systematic issue, not model-specific

---

## Superseded (old schema — different test set, accuracy-primary)

> WC 2026 group stage only, accuracy as primary metric. Historical reference only.

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
