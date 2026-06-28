"""Step 9: Dedicated draw handling — draw multiplier, ensemble, diagnostics.

Three phases (run in one pass — models are trained once, predictions cached):

  Phase 1 — base models
    tabpfn_3class   3-class TabPFN (reference)
    tabpfn_2stage   stage1: draw/not_draw  stage2: home/away | not_draw
    poisson         PoissonGLM on goals → sum score probs → 1X2
    dixon_coles     Poisson + τ correction for low-score cells (MLE ρ)

  Phase 2 — draw multiplier sweep (applied to cached predictions)
    p_draw *= k  then renormalize;  k ∈ {0.9, 1.0, 1.1, 1.2, 1.3, 1.4}

  Phase 3 — ensemble (weighted average of cached predictions)
    2-model: tabpfn_2stage+dc, tabpfn_2stage+poisson, tabpfn_3class+poisson
    3-model: tabpfn_2stage+poisson+dc
    weights: 0.25/0.5/0.75 grid

  Diagnostics reported for each model:
    mean p_draw on actual draws / non-draws
    draw log-loss / non-draw log-loss
    draw probability rank distribution (1st / 2nd / 3rd most likely)

Usage:
    uv run draw_models.py [--refresh] [--log]
    uv run draw_models.py --only tabpfn_2stage poisson   # skip heavy TabPFN runs
"""
import argparse
import itertools
import json
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn_client import TabPFNClassifier

from features import (
    load_data, build_features,
    FEATURES, TRAIN_START, MAX_TRAIN,
    wc_matches, wc_group_stage, wc_knockout, wc_group_rounds,
)

EXPERIMENTS_LOG = "experiments.jsonl"
TODAY_STR       = datetime.now().strftime("%Y%m%d")
WC2022_START    = pd.Timestamp("2022-11-20")
WC2026_START    = pd.Timestamp("2026-06-11")
MAX_GOALS       = 8
MULTIPLIERS     = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]


# ══════════════════════════════════════════════════════════════════════════════
# Metrics and diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def _ll(hda: np.ndarray, outcomes) -> float:
    lex = hda[:, [2, 1, 0]]
    return log_loss(outcomes, lex, labels=["away_win", "draw", "home_win"])


def _acc(hda: np.ndarray, outcomes) -> float:
    label_arr = np.array(["home_win", "draw", "away_win"])
    return (label_arr[hda.argmax(1)] == np.array(outcomes)).mean()


def _draw_diagnostics(hda: np.ndarray, outcomes) -> dict:
    """Draw-specific signal quality metrics."""
    actual      = np.array(outcomes)
    p_draw      = hda[:, 1]
    is_draw     = actual == "draw"
    is_not_draw = ~is_draw

    draw_ll     = (-np.log(p_draw[is_draw].clip(1e-9))).mean()     if is_draw.any()     else float("nan")
    non_draw_ll = (-np.log((1 - p_draw)[is_not_draw].clip(1e-9))).mean() if is_not_draw.any() else float("nan")

    # rank of p_draw among [p_home, p_draw, p_away] for actual draws
    ranks = (hda[is_draw] >= p_draw[is_draw, None]).sum(axis=1)  # 1=highest, 3=lowest
    rank_counts = {r: int((ranks == r).sum()) for r in [1, 2, 3]}

    return {
        "n_draws":          int(is_draw.sum()),
        "n_total":          len(actual),
        "mean_p_draw_draw": float(p_draw[is_draw].mean())     if is_draw.any()     else float("nan"),
        "mean_p_draw_ndraw":float(p_draw[is_not_draw].mean()) if is_not_draw.any() else float("nan"),
        "draw_ll":          float(draw_ll),
        "non_draw_ll":      float(non_draw_ll),
        "rank1": rank_counts[1],  # draw is most likely  → correctly confident
        "rank2": rank_counts[2],  # draw is 2nd          → some signal
        "rank3": rank_counts[3],  # draw is least likely → no signal
    }


def _apply_multiplier(hda: np.ndarray, k: float) -> np.ndarray:
    out = hda.copy()
    out[:, 1] *= k
    return out / out.sum(axis=1, keepdims=True)


# ══════════════════════════════════════════════════════════════════════════════
# Base model trainers
# ══════════════════════════════════════════════════════════════════════════════

def _tabpfn_3class(pool, test) -> np.ndarray:
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[FEATURES].values, pool["outcome"].values)
    proba   = clf.predict_proba(test[FEATURES].values)
    proba  /= proba.sum(axis=1, keepdims=True)
    classes = list(clf.classes_)
    return np.column_stack([
        proba[:, classes.index("home_win")],
        proba[:, classes.index("draw")],
        proba[:, classes.index("away_win")],
    ])


def _tabpfn_2stage(pool, test) -> np.ndarray:
    X_pool = pool[FEATURES].values
    y_pool = pool["outcome"].values

    y1 = (y_pool == "draw").astype(int)
    clf1 = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf1.fit(X_pool, y1)
    p1      = clf1.predict_proba(test[FEATURES].values)
    q_draw  = p1[:, list(clf1.classes_).index(1)]

    mask_nd = y_pool != "draw"
    clf2 = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf2.fit(X_pool[mask_nd], (y_pool[mask_nd] == "home_win").astype(int))
    p2              = clf2.predict_proba(test[FEATURES].values)
    q_home_given_nd = p2[:, list(clf2.classes_).index(1)]

    hda = np.column_stack([
        (1 - q_draw) * q_home_given_nd,
        q_draw,
        (1 - q_draw) * (1 - q_home_given_nd),
    ])
    return hda / hda.sum(axis=1, keepdims=True)


_GOAL_FEATS_HOME = ["home_gf5", "home_ga5", "away_ga5", "away_gf5",
                    "elo_diff", "home_elo", "home_form5", "home_rest"]
_GOAL_FEATS_AWAY = ["away_gf5", "away_ga5", "home_ga5", "home_gf5",
                    "elo_diff", "away_elo", "away_form5", "away_rest"]


def _poisson_probs(lam_h: np.ndarray, lam_a: np.ndarray) -> np.ndarray:
    k    = np.arange(MAX_GOALS + 1)
    p_hw = np.zeros(len(lam_h))
    p_d  = np.zeros(len(lam_h))
    p_aw = np.zeros(len(lam_h))
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            if i > j:   p_hw += p
            elif i == j: p_d  += p
            else:        p_aw += p
    hda = np.column_stack([p_hw, p_d, p_aw])
    return hda / hda.sum(axis=1, keepdims=True)


def _train_poisson(pool):
    glm_h = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.1, max_iter=300))
    glm_a = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.1, max_iter=300))
    glm_h.fit(pool[_GOAL_FEATS_HOME].values, pool["home_score"].values)
    glm_a.fit(pool[_GOAL_FEATS_AWAY].values, pool["away_score"].values)
    return glm_h, glm_a


def _poisson(pool, test) -> np.ndarray:
    glm_h, glm_a = _train_poisson(pool)
    lam_h = glm_h.predict(test[_GOAL_FEATS_HOME].values).clip(0.2, 8)
    lam_a = glm_a.predict(test[_GOAL_FEATS_AWAY].values).clip(0.2, 8)
    return _poisson_probs(lam_h, lam_a)


def _dc_tau(x, y, lam_h, lam_a, rho):
    tau = np.ones(len(lam_h))
    tau[(x == 0) & (y == 0)] -= lam_h[(x == 0) & (y == 0)] * lam_a[(x == 0) & (y == 0)] * rho
    tau[(x == 1) & (y == 0)] += lam_a[(x == 1) & (y == 0)] * rho
    tau[(x == 0) & (y == 1)] += lam_h[(x == 0) & (y == 1)] * rho
    tau[(x == 1) & (y == 1)] -= rho
    return tau.clip(1e-6)


def _dc_nll(rho, pool, glm_h, glm_a):
    lam_h = glm_h.predict(pool[_GOAL_FEATS_HOME].values).clip(0.2, 8)
    lam_a = glm_a.predict(pool[_GOAL_FEATS_AWAY].values).clip(0.2, 8)
    hs, as_ = pool["home_score"].values.astype(int), pool["away_score"].values.astype(int)
    ll = (np.log(poisson.pmf(hs, lam_h).clip(1e-9))
          + np.log(poisson.pmf(as_, lam_a).clip(1e-9))
          + np.log(_dc_tau(hs, as_, lam_h, lam_a, rho)))
    return -ll.sum()


def _dixon_coles(pool, test) -> np.ndarray:
    glm_h, glm_a = _train_poisson(pool)
    rho  = minimize_scalar(_dc_nll, bounds=(-0.2, 0.2), method="bounded",
                           args=(pool, glm_h, glm_a)).x
    lam_h = glm_h.predict(test[_GOAL_FEATS_HOME].values).clip(0.2, 8)
    lam_a = glm_a.predict(test[_GOAL_FEATS_AWAY].values).clip(0.2, 8)
    p_hw = np.zeros(len(lam_h))
    p_d  = np.zeros(len(lam_h))
    p_aw = np.zeros(len(lam_h))
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            ph = (poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
                  * _dc_tau(np.full(len(lam_h), i), np.full(len(lam_h), j), lam_h, lam_a, rho))
            if i > j:    p_hw += ph
            elif i == j: p_d  += ph
            else:        p_aw += ph
    hda = np.column_stack([p_hw, p_d, p_aw])
    return hda / hda.sum(axis=1, keepdims=True)


BASE_MODELS = {
    "tabpfn_3class": _tabpfn_3class,
    "tabpfn_2stage": _tabpfn_2stage,
    "poisson":       _poisson,
    "dixon_coles":   _dixon_coles,
}

ENSEMBLES = [
    ("tabpfn_2stage", "dixon_coles"),
    ("tabpfn_2stage", "poisson"),
    ("tabpfn_3class", "poisson"),
    ("tabpfn_2stage", "poisson", "dixon_coles"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

def _sep(n=72): print("─" * n)


def _print_base_table(bt_label, cached, outcomes):
    print(f"\n{'━'*72}")
    print(f"  {bt_label}")
    print(f"{'━'*72}")
    hdr = f"  {'model':<22} {'ll':>8} {'acc':>7}  {'p_draw|draw':>11} {'p_draw|ndraw':>12} {'rank 1/2/3':>12}"
    print(hdr)
    _sep()
    for name, hda in cached.items():
        ll  = _ll(hda, outcomes)
        acc = _acc(hda, outcomes)
        d   = _draw_diagnostics(hda, outcomes)
        r   = f"{d['rank1']}/{d['rank2']}/{d['rank3']}"
        pd_draw  = f"{d['mean_p_draw_draw']:.3f}"  if not np.isnan(d['mean_p_draw_draw'])  else "—"
        pd_ndraw = f"{d['mean_p_draw_ndraw']:.3f}" if not np.isnan(d['mean_p_draw_ndraw']) else "—"
        print(f"  {name:<22} {ll:>8.4f} {acc:>7.1%}  {pd_draw:>11} {pd_ndraw:>12} {r:>12}")
    diag = _draw_diagnostics(list(cached.values())[0], outcomes)
    print(f"  actual draws: {diag['n_draws']}/{diag['n_total']}")


def _print_multiplier_table(bt_label, cached, outcomes):
    print(f"\n── Draw multiplier sweep — {bt_label}")
    names = list(cached)
    hdr   = f"  {'k':>4}" + "".join(f"  {n:>16}" for n in names)
    print(hdr)
    _sep(len(hdr))
    best = {n: (float("inf"), 1.0) for n in names}
    for k in MULTIPLIERS:
        row = f"  {k:>4.1f}"
        for name, hda in cached.items():
            ll = _ll(_apply_multiplier(hda, k), outcomes)
            row += f"  {ll:>16.4f}"
            if ll < best[name][0]:
                best[name] = (ll, k)
        print(row)
    print("  best k: " + "  ".join(f"{n}=k{best[n][1]}" for n in names))
    return best


def _print_ensemble_table(bt_label, cached, outcomes):
    print(f"\n── Ensemble — {bt_label}")
    hdr = f"  {'combination':<40} {'weights':>18}  {'ll':>8}"
    print(hdr)
    _sep(len(hdr))
    results = []
    for combo in ENSEMBLES:
        if not all(m in cached for m in combo):
            continue
        n = len(combo)
        # generate weight grids summing to 1
        if n == 2:
            weight_sets = [(w, round(1 - w, 2)) for w in [0.25, 0.5, 0.75]]
        else:
            weight_sets = [
                ws for ws in itertools.product([0.25, 0.5, 0.75], repeat=n)
                if abs(sum(ws) - 1.0) < 1e-9
            ]
        for ws in weight_sets:
            blend = sum(w * cached[m] for w, m in zip(ws, combo))
            blend /= blend.sum(axis=1, keepdims=True)
            ll = _ll(blend, outcomes)
            label = "+".join(m.replace("tabpfn_", "t_").replace("dixon_coles", "dc") for m in combo)
            w_str = "+".join(f"{w:.2f}" for w in ws)
            results.append((ll, label, w_str, blend))
            print(f"  {label:<40} {w_str:>18}  {ll:>8.4f}")
    if results:
        best = min(results, key=lambda x: x[0])
        print(f"  → best: {best[1]} @ {best[2]}  ll={best[0]:.4f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _pool(feats, cutoff):
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]
    return played[played["date"] < cutoff].tail(MAX_TRAIN)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _upsert(run_name, entry):
    existing = []
    if os.path.exists(EXPERIMENTS_LOG):
        with open(EXPERIMENTS_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
    existing = [e for e in existing if e.get("run") != run_name]
    existing.append(entry)
    with open(EXPERIMENTS_LOG, "w") as f:
        for e in existing:
            f.write(json.dumps(e) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--log",     action="store_true")
    parser.add_argument("--only",    nargs="+",
                        choices=list(BASE_MODELS),
                        help="Run only these base models (skip others)")
    args = parser.parse_args()

    df    = load_data(refresh=args.refresh)
    feats = build_features(df)

    wc22     = wc_matches(feats, 2022)
    wc26     = wc_matches(feats, 2026)
    bt_cfgs  = [
        ("BT1 — WC2022 group (48)",    _pool(feats, WC2022_START), wc_group_stage(wc22)),
        ("BT2 — WC2022 knockout (16)", _pool(feats, WC2022_START), wc_knockout(wc22)),
        ("BT3 — WC2026 R1-2 (48)",     _pool(feats, WC2026_START), wc_group_rounds(wc26, 2)),
    ]

    active_models = {k: v for k, v in BASE_MODELS.items()
                     if args.only is None or k in args.only}

    print(f"\nModels  : {list(active_models)}")
    print(f"Pool    : BT1/BT2={len(bt_cfgs[0][1])}  BT3={len(bt_cfgs[2][1])}")

    for bt_label, pool, test in bt_cfgs:
        outcomes = test["outcome"].tolist()

        # ── Phase 1: train and cache all base predictions ──────────────────
        cached = {}
        for name, fn in active_models.items():
            try:
                cached[name] = fn(pool, test)
            except Exception as e:
                print(f"  [{name}] ERROR: {e}")

        if not cached:
            continue

        # ── Phase 2: base model table + diagnostics ────────────────────────
        _print_base_table(bt_label, cached, outcomes)

        # ── Phase 3: draw multiplier sweep ─────────────────────────────────
        _print_multiplier_table(bt_label, cached, outcomes)

        # ── Phase 4: ensemble ──────────────────────────────────────────────
        _print_ensemble_table(bt_label, cached, outcomes)

    if args.log:
        entry = {
            "run":       f"draw_models_{TODAY_STR}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "commit":    git_commit(),
            "features":  FEATURES,
            "note":      "draw multiplier + ensemble sweep; see stdout for full table",
        }
        _upsert(entry["run"], entry)
        print(f"\nLogged '{entry['run']}'")


if __name__ == "__main__":
    main()
