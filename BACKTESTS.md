# Backtest Results

**Primary metric: multi-class log-loss (lower is better)** — competition scoring metric per §8.
Uniform baseline (random guessing): log-loss = 1.099.

Three fixed test sets evaluated with `uv run backtest.py`:

| ID | Test set | Matches | Training pool cutoff |
|----|----------|---------|----------------------|
| BT1 | WC 2022 group stage | 48 | 2022-11-20 |
| BT2 | WC 2022 knockout | 16 | 2022-11-20 |
| BT3 | WC 2026 group stage rounds 1–2 | 48 | 2026-06-11 |

**Bold** = best log-loss in column &nbsp;·&nbsp; <u>Underline</u> = best accuracy in column &nbsp;·&nbsp; ★ = current production model (main)

> **Production model: main** — 20 features · MAX_TRAIN=3000 · draw_k=1.2
> **Knockout submission**: `uv run predict.py --draw-scale 1.2` → BT2=**0.9020**
> Source: PR #1 (`tw_m3000`) + PR #2 (`draw_handling`)

---

## Experiment log

| run | branch | parent | BT1 ll | BT2 ll | BT3 ll | BT1 acc | BT2 acc | BT3 acc | commit | notes |
|-----|--------|--------|:------:|:------:|:------:|:-------:|:-------:|:-------:|--------|-------|
| uniform | exp_0628/baselines | — | **1.0986** | 1.0986 | 1.0986 | 37.5% | <u>62.5%</u> | 52.1% | 3262c25 | [1/3, 1/3, 1/3] every match |
| always_home | exp_0628/baselines | — | 2.8858 | 1.7396 | 2.2172 | 37.5% | <u>62.5%</u> | 52.1% | 3262c25 | [0.98, 0.01, 0.01] every match |
| elo_logistic | exp_0628/baselines | — | 1.1515 | **0.8724** | 0.9123 | 47.9% | 56.2% | <u>62.5%</u> | 3262c25 | multinomial LR on elo_diff only |
| lr_all_features | exp_0628/baselines | — | 1.1217 | 0.9326 | **0.9000** | 47.9% | 56.2% | 58.3% | 3262c25 | multinomial LR + StandardScaler, 26 features |
| tabpfn_all_features | exp_0628/baselines | — | 1.1337 | 0.9456 | 0.9121 | 47.9% | 56.2% | <u>62.5%</u> | 3262c25 | TabPFN, 26 features |
| baseline | exp_0628/baselines | — | 1.1258 | 0.9405 | 0.9163 | <u>50.0%</u> | 56.2% | <u>62.5%</u> | 39550fa | TabPFN, original 26 features, logged via backtest.py |
| wc_context_features | exp_0628/wc_context_features | main | 1.1241 | 0.9321 | 0.9259 | 47.9% | 56.2% | 58.3% | 4222893 | +abs_elo_diff, host_adv_diff, concacaf_adv_diff, same_continent_adv_diff → 30 features |
| symmetric_features | exp_0628/symmetric_features | exp_0628/wc_context_features | 1.1280 | 0.9530 | 0.9290 | 47.9% | 56.2% | 58.3% | ec65f5f | replaced 8 home/away individual features with 4 diffs → 26 features; regressed vs wc_context |
| lightweight_20feat | exp_0628/lightweight_20feat | exp_0628/ablation · abl_elo_form_homeaway | 1.1346 | 0.9321 | 0.9302 | <u>50.0%</u> | 56.2% | 58.3% | 6826053 | Elo + form + rest + home/away stats; no H2H / ctx / WC — best feature-rich TabPFN on BT2 |
| tw_m3000 | main (via PR #1) | exp_0628/lightweight_20feat | 1.1601 | 0.9075 | 0.9428 | <u>50.0%</u> | 56.2% | 64.6% | bf8d22c | 20 features + MAX_TRAIN=3000 · merged into main via PR #1 |
| tw_s2018_m10000 | exp_0628/train_window | exp_0628/lightweight_20feat | 1.1458 | 0.9191 | **0.9290** | <u>50.0%</u> | 56.2% | 58.3% | bf8d22c | 20 features + TRAIN_START=2018 + MAX_TRAIN=10000; best BT3 so far |
| draw_3class | exp_0628/draw_handling | main | 1.1606 | 0.9062 | 0.9433 | <u>50.0%</u> | 56.2% | 64.6% | b6aa62d | tabpfn_3class raw, no draw multiplier |
| draw_2stage | exp_0628/draw_handling | main | 1.1749 | 0.9102 | 0.9282 | <u>50.0%</u> | 56.2% | — | b6aa62d | two-stage: binary draw/not-draw then home/away |
| draw_3class_k12 ★ | exp_0628/draw_handling | main | — | **0.9020** | — | — | — | — | b6aa62d | tabpfn_3class + draw_k=1.2 (alpha/eps are no-ops for KO) · submit with `--draw-scale 1.2` |

### Key observations

- **BT1 (WC22 group)**: no model beats uniform on log-loss — group stage is near-unpredictable; wc_context_features is the best non-trivial model here
- **BT2 (WC22 KO)**: elo_logistic dominates; draw_3class_k12 is the best feature-rich model (0.9020)
- **BT3 (WC26 R1-2)**: lr_all_features leads; two-stage model (draw_2stage, 0.9282) best among TabPFN variants
- **Draw recall = 0%** — argmax never picks draw; draw multiplier k=1.2 improves log-loss without needing draw recall
- **Calibration (Step 10)**: eps clipping is a no-op (probabilities never near boundaries); alpha=1.0 is optimal for KO stage (no power scaling needed); only draw_k matters

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

## Training window experiment (exp_0628/train_window, 20-feature set)

Branch: `exp_0628/train_window` · Parent: `exp_0628/lightweight_20feat` · commit bf8d22c

Fixed 20-feature set. Grid: TRAIN_START ∈ {2010, 2014, 2018} × MAX_TRAIN ∈ {3000, 5000, 10000}.
`n22` / `n26` = actual matches used for BT1/BT2 and BT3 training pools respectively.

| config | start | max | n22 | n26 | BT1 ll | BT2 ll | BT3 ll | BT1 acc | BT2 acc | BT3 acc |
|--------|:-----:|----:|----:|----:|:------:|:------:|:------:|:-------:|:-------:|:-------:|
| s2010_m3000 | 2010 | 3000 | 3000 | 3000 | 1.1601 | **0.9075** | 0.9428 | <u>50.0%</u> | 56.2% | <u>64.6%</u> |
| s2010_m5000 | 2010 | 5000 | 5000 | 5000 | 1.1466 | 0.9238 | 0.9347 | <u>50.0%</u> | 56.2% | <u>64.6%</u> |
| s2010_m10000 | 2010 | 10000 | 10000 | 10000 | 1.1400 | 0.9301 | 0.9302 | <u>50.0%</u> | 56.2% | 58.3% |
| s2014_m3000 | 2014 | 3000 | 3000 | 3000 | 1.1601 | **0.9075** | 0.9428 | <u>50.0%</u> | 56.2% | <u>64.6%</u> |
| s2014_m5000 | 2014 | 5000 | 5000 | 5000 | 1.1466 | 0.9238 | 0.9347 | <u>50.0%</u> | 56.2% | <u>64.6%</u> |
| s2014_m10000 | 2014 | 10000 | 8142 | 10000 | 1.1346 | 0.9321 | 0.9302 | <u>50.0%</u> | 56.2% | 58.3% |
| s2018_m3000 | 2018 | 3000 | 3000 | 3000 | 1.1601 | **0.9075** | 0.9428 | <u>50.0%</u> | 56.2% | <u>64.6%</u> |
| s2018_m5000 | 2018 | 5000 | 4403 | 5000 | 1.1458 | 0.9191 | 0.9347 | <u>50.0%</u> | 56.2% | <u>64.6%</u> |
| s2018_m10000 | 2018 | 10000 | 4403 | 8108 | 1.1458 | 0.9191 | **0.9290** | <u>50.0%</u> | 56.2% | 58.3% |

### Training window findings

**TRAIN_START has no independent effect — MAX_TRAIN is the decisive variable:**
- s2010_m3000, s2014_m3000, s2018_m3000 produce identical results (pool always capped at 3000)
- TRAIN_START only matters when available data < MAX_TRAIN (e.g. s2018_m10000: only 4403 matches pre-WC2022)

**BT2 (knockout): MAX_TRAIN=3000 is best (0.9075)**
- 0.025 improvement over MAX_TRAIN=10000 (0.9321) — a large gap
- The most recent 3000 matches ≈ 2019–2022, better reflecting current team strength; larger pools dilute the recency signal
- s2018_m5000/10000 (0.9191) sits in between, confirming the value of recency

**BT3 (group stage): MAX_TRAIN=10000 is best (s2018_m10000 = 0.9290)**
- MAX_TRAIN=3000 is worst (0.9428); more data consistently helps for group stage
- s2018_m10000 (0.9290) slightly edges s2014_m10000 (0.9302)

**Core tension:** knockout stage favours small recent pools; group stage favours large pools.
For the competition target (knockout) → MAX_TRAIN=3000 is the single most impactful adjustment found so far.

---

## Draw handling + post-calibration (exp_0628/draw_handling · merged PR #2)

Branch: `exp_0628/draw_handling` · Parent: `main` · commit b6aa62d

### Step 9 — draw model comparison (raw, no multiplier)

| model | BT1 ll | BT2 ll | BT3 ll | notes |
|-------|:------:|:------:|:------:|-------|
| tabpfn_3class | 1.1606 | 0.9062 | 0.9433 | standard 3-class TabPFN (production base) |
| tabpfn_2stage | 1.1749 | 0.9102 | **0.9282** | binary draw/not-draw then home/away |
| poisson | 1.1041 | 0.9230 | 0.9396 | Poisson GLM on goal features |
| dixon_coles | 1.1058 | **0.9216** | 0.9315 | Poisson + τ correction for low-score cells |

### Step 9 — draw multiplier sweep (tabpfn_3class, BT2)

| draw_k | BT2 ll | Δ vs raw |
|:------:|:------:|:--------:|
| 0.9 | 0.9157 | +0.0095 |
| 1.0 | 0.9062 | 0.0000 |
| 1.1 | 0.9031 | −0.0031 |
| **1.2** | **0.9020** | **−0.0043** |
| 1.3 | 0.9034 | −0.0028 |
| 1.4 | 0.9078 | +0.0016 |

### Step 10 — post-calibration grid (BT2 target)

Grid: eps ∈ {0.003, 0.005, 0.01} × alpha ∈ {0.8, 0.9, 1.0} × draw_k ∈ {0.9, 1.0, 1.1, 1.2}

**Key findings:**
- **eps clipping**: no effect — model probabilities never approach the clip boundary
- **alpha power**: alpha=1.0 (no change) is optimal for BT2 (KO stage); alpha=0.8 helps BT1/BT3 (group stage) by −0.048 but hurts KO
- **draw_k=1.2** is the only effective lever for KO stage; confirmed across both models

| model | alpha | draw_k | eps | BT2 ll | Δ vs raw |
|-------|:-----:|:------:|:---:|:------:|:--------:|
| tabpfn_3class | 1.0 | 1.2 | any | **0.9020** | −0.0043 |
| tabpfn_2stage | 1.0 | 1.2 | any | 0.9041 | −0.0061 |

**Submission command for knockout stage:** `uv run predict.py --draw-scale 1.2`

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
