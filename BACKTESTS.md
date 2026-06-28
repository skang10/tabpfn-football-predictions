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
| lightweight_20feat | exp_0628/lightweight_20feat | exp_0628/ablation · abl_elo_form_homeaway | 1.1346 | 0.9321 | 0.9302 | <u>50.0%</u> | 56.2% | 58.3% | 6826053 | Elo + form + rest + home/away stats; no H2H / ctx / WC — best feature-rich TabPFN on BT2 |

### Key observations

- **BT1 (WC22 group)**: no model beats uniform on log-loss — the group stage is near-unpredictable; wc_context_features is the best non-trivial model here
- **BT2 (WC22 KO)**: elo_logistic dominates; wc_context_features second best among feature-rich models
- **BT3 (WC26 R1-2)**: lr_all_features leads; WC context features hurt slightly vs baseline (model is confused by context features when test data is very recent)
- **Draw recall = 0%** across all runs — systematic issue, not model-specific

---

## Feature ablation (exp_0628/ablation, TabPFN only)

Branch: `exp_0628/ablation` · Parent: `exp_0628/wc_context_features` · commit ee66915

Cumulative feature sets (each row adds to the previous), then targeted removals from `all_current`.

| ablation set | n features | BT1 ll | BT2 ll | BT3 ll | BT1 acc | BT2 acc | BT3 acc |
|--------------|:----------:|:------:|:------:|:------:|:-------:|:-------:|:-------:|
| elo_only | 1 | 1.1648 | **0.8562** | 0.9347 | 47.9% | 56.2% | <u>62.5%</u> |
| elo_features | 3 | 1.1561 | 0.9555 | 0.9369 | <u>50.0%</u> | 56.2% | <u>62.5%</u> |
| elo_form | 7 | 1.1543 | 0.9475 | 0.9410 | <u>50.0%</u> | 56.2% | <u>62.5%</u> |
| elo_form_rest | 9 | 1.1550 | 0.9383 | 0.9410 | 47.9% | 56.2% | <u>62.5%</u> |
| elo_form_homeaway | 20 | 1.1346 | 0.9321 | 0.9302 | <u>50.0%</u> | 56.2% | 58.3% |
| all_current | 26 | 1.1314 | 0.9520 | **0.9136** | 47.9% | 56.2% | <u>62.5%</u> |
| no_h2h | 22 | 1.1366 | 0.9495 | 0.9342 | 47.9% | 56.2% | 58.3% |
| no_streak | 24 | 1.1331 | 0.9530 | 0.9197 | 47.9% | 56.2% | 60.4% |
| wc_context | 30 | **1.1273** | 0.9414 | 0.9248 | 47.9% | 56.2% | 58.3% |

### Ablation findings

**BT2 (WC knockout — most important for competition):**
- `elo_only` is the single best model on BT2 (0.8562), beating every feature-rich variant
- Adding form/rest brings modest recovery; H2H + ctx (→ all_current) then *hurts* back to 0.9520
- `no_h2h` (0.9495) beats `all_current` (0.9520) — H2H features are net negative on KO stage
- WC context features (→ wc_context 0.9414) partially recover but still far from elo_only

**BT3 (WC26 group stage — most recent data):**
- `all_current` (26 features) is best (0.9136) — neither more nor fewer features helps
- Removing H2H hurts (+0.021): H2H signal is informative for group stage
- Removing streak hurts (+0.006): mild contribution
- Adding WC context hurts (+0.011): context features add noise on recent data

**BT1 (WC22 group stage):**
- More features consistently helps; wc_context best (1.1273)
- All models remain above uniform (1.0986) — group stage is the hardest BT

**Signal vs noise summary:**
| feature group | BT2 effect | BT3 effect | verdict |
|---------------|-----------|-----------|---------|
| Elo diff only | baseline best | weak | strong for KO |
| + form | neutral | neutral | modest |
| + rest | ↑ (helps) | neutral | marginal |
| + home/away stats | ↑ (helps) | ↑ (helps) | useful |
| + H2H | ↓ (hurts) | ↑ (helps) | mixed |
| + WC context | ↓ (hurts) | ↓ (hurts) | hurts on recent data |
| + streak | ↓ (hurts) | ↑ (helps) | mixed |

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
