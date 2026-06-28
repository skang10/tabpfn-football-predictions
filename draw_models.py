"""Step 9: Dedicated draw handling — compare three model tracks on BT1/BT2/BT3.

Tracks:
  tabpfn_3class   current production (reference, 3-class TabPFN)
  tabpfn_2stage   stage1: draw/not_draw → stage2: home_win/away_win | not_draw
  poisson         Poisson GLM on goals → sum score probabilities → 1X2
  dixon_coles     Poisson + Dixon-Coles τ correction for low-score cells

Two-stage combination:
  p_draw     = q_draw                               (stage 1)
  p_home_win = (1 - q_draw) * q_home_given_not_draw (stage 1 × stage 2)
  p_away_win = (1 - q_draw) * (1 - q_home_given_not_draw)

Usage:
    uv run draw_models.py [--refresh] [--log] [--only tabpfn_2stage poisson]
"""
import argparse
import json
import os
import subprocess
from collections import defaultdict
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
MAX_GOALS       = 8   # truncation for Poisson sum


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _ll(proba_hda: np.ndarray, outcomes) -> float:
    lex = proba_hda[:, [2, 1, 0]]
    return log_loss(outcomes, lex, labels=["away_win", "draw", "home_win"])


def _acc(proba_hda: np.ndarray, outcomes) -> float:
    label_arr = np.array(["home_win", "draw", "away_win"])
    pred = label_arr[proba_hda.argmax(1)]
    return (pred == np.array(outcomes)).mean()


def _draw_recall(proba_hda: np.ndarray, outcomes) -> float:
    label_arr = np.array(["home_win", "draw", "away_win"])
    pred = label_arr[proba_hda.argmax(1)]
    actual = np.array(outcomes)
    draws_actual = (actual == "draw").sum()
    if draws_actual == 0:
        return float("nan")
    return ((pred == "draw") & (actual == "draw")).sum() / draws_actual


def _pool(feats, cutoff):
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]
    return played[played["date"] < cutoff].tail(MAX_TRAIN)


def _evaluate(name, proba_hda, outcomes):
    ll  = _ll(proba_hda, outcomes)
    acc = _acc(proba_hda, outcomes)
    dr  = _draw_recall(proba_hda, outcomes)
    return {"name": name, "ll": ll, "acc": acc, "draw_recall": dr, "n": len(outcomes)}


# ── Track 1: 3-class TabPFN ────────────────────────────────────────────────────

def _tabpfn_3class(pool, test):
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[FEATURES].values, pool["outcome"].values)
    proba = clf.predict_proba(pool[FEATURES].values[:1])   # warm-up (no-op)
    proba = clf.predict_proba(test[FEATURES].values)
    proba = proba / proba.sum(axis=1, keepdims=True)
    classes = list(clf.classes_)
    hda = np.column_stack([
        proba[:, classes.index("home_win")],
        proba[:, classes.index("draw")],
        proba[:, classes.index("away_win")],
    ])
    return hda


# ── Track 2: Two-stage TabPFN ──────────────────────────────────────────────────

def _tabpfn_2stage(pool, test):
    X_pool = pool[FEATURES].values
    y_pool = pool["outcome"].values

    # Stage 1: draw vs not_draw
    y1 = (y_pool == "draw").astype(int)
    clf1 = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf1.fit(X_pool, y1)
    p1 = clf1.predict_proba(test[FEATURES].values)
    classes1 = list(clf1.classes_)
    q_draw = p1[:, classes1.index(1)]

    # Stage 2: home_win vs away_win (trained only on non-draw matches)
    mask_nd = y_pool != "draw"
    X_nd = X_pool[mask_nd]
    y_nd = (y_pool[mask_nd] == "home_win").astype(int)
    clf2 = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf2.fit(X_nd, y_nd)
    p2 = clf2.predict_proba(test[FEATURES].values)
    classes2 = list(clf2.classes_)
    q_home_given_nd = p2[:, classes2.index(1)]

    p_draw     = q_draw
    p_home_win = (1 - q_draw) * q_home_given_nd
    p_away_win = (1 - q_draw) * (1 - q_home_given_nd)

    hda = np.column_stack([p_home_win, p_draw, p_away_win])
    hda = hda / hda.sum(axis=1, keepdims=True)
    return hda


# ── Track 3a: Poisson GLM ──────────────────────────────────────────────────────

_GOAL_FEATS_HOME = [
    "home_gf5", "home_ga5", "away_ga5", "away_gf5",
    "elo_diff", "home_elo", "home_form5", "home_rest",
]
_GOAL_FEATS_AWAY = [
    "away_gf5", "away_ga5", "home_ga5", "home_gf5",
    "elo_diff", "away_elo", "away_form5", "away_rest",
]


def _poisson_probs(lam_h, lam_a, max_g=MAX_GOALS):
    """Vectorised: lam_h, lam_a are 1-D arrays of length n."""
    k = np.arange(max_g + 1)
    # P[i,j] = P(home=i) * P(away=j)  per match
    p_h = poisson.pmf(k[None, :, None], lam_h[:, None, None])   # (n, k, 1)
    p_a = poisson.pmf(k[None, None, :], lam_a[:, None, None])   # (n, 1, k)
    joint = p_h * p_a                                             # (n, k, k)
    p_home_win = joint[:, np.tril_indices(max_g + 1, -1)[0],
                         np.tril_indices(max_g + 1, -1)[1]].sum(1)
    p_away_win = joint[:, np.triu_indices(max_g + 1, 1)[0],
                         np.triu_indices(max_g + 1, 1)[1]].sum(1)
    p_draw     = np.array([joint[i].diagonal().sum() for i in range(len(lam_h))])
    hda = np.column_stack([p_home_win, p_draw, p_away_win])
    return hda / hda.sum(axis=1, keepdims=True)


def _train_poisson(pool):
    glm_h = make_pipeline(StandardScaler(),
                          PoissonRegressor(alpha=0.1, max_iter=300))
    glm_a = make_pipeline(StandardScaler(),
                          PoissonRegressor(alpha=0.1, max_iter=300))
    glm_h.fit(pool[_GOAL_FEATS_HOME].values, pool["home_score"].values)
    glm_a.fit(pool[_GOAL_FEATS_AWAY].values, pool["away_score"].values)
    return glm_h, glm_a


def _poisson_predict(glm_h, glm_a, test):
    lam_h = glm_h.predict(test[_GOAL_FEATS_HOME].values).clip(0.2, 8)
    lam_a = glm_a.predict(test[_GOAL_FEATS_AWAY].values).clip(0.2, 8)
    return _poisson_probs(lam_h, lam_a)


# ── Track 3b: Dixon-Coles ──────────────────────────────────────────────────────

def _dc_tau(x, y, lam_h, lam_a, rho):
    """Dixon-Coles τ correction for low-score cells (vectorised over matches)."""
    tau = np.ones(len(lam_h))
    mask_00 = (x == 0) & (y == 0)
    mask_10 = (x == 1) & (y == 0)
    mask_01 = (x == 0) & (y == 1)
    mask_11 = (x == 1) & (y == 1)
    tau[mask_00] = 1 - lam_h[mask_00] * lam_a[mask_00] * rho
    tau[mask_10] = 1 + lam_a[mask_10] * rho
    tau[mask_01] = 1 + lam_h[mask_01] * rho
    tau[mask_11] = 1 - rho
    return tau.clip(1e-6)


def _dc_log_likelihood(rho, pool, glm_h, glm_a):
    lam_h = glm_h.predict(pool[_GOAL_FEATS_HOME].values).clip(0.2, 8)
    lam_a = glm_a.predict(pool[_GOAL_FEATS_AWAY].values).clip(0.2, 8)
    hs = pool["home_score"].values.astype(int)
    as_ = pool["away_score"].values.astype(int)
    ll = (np.log(poisson.pmf(hs, lam_h).clip(1e-9))
          + np.log(poisson.pmf(as_, lam_a).clip(1e-9))
          + np.log(_dc_tau(hs, as_, lam_h, lam_a, rho)))
    return -ll.sum()


def _dc_probs(lam_h, lam_a, rho, max_g=MAX_GOALS):
    k = np.arange(max_g + 1)
    p_home_win = np.zeros(len(lam_h))
    p_draw     = np.zeros(len(lam_h))
    p_away_win = np.zeros(len(lam_h))
    for i in k:
        for j in k:
            ph = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            ph *= _dc_tau(
                np.full(len(lam_h), i), np.full(len(lam_h), j),
                lam_h, lam_a, rho,
            )
            if i > j:
                p_home_win += ph
            elif i == j:
                p_draw += ph
            else:
                p_away_win += ph
    hda = np.column_stack([p_home_win, p_draw, p_away_win])
    return hda / hda.sum(axis=1, keepdims=True)


def _dixon_coles_predict(pool, test):
    glm_h, glm_a = _train_poisson(pool)
    res = minimize_scalar(
        _dc_log_likelihood, bounds=(-0.2, 0.2), method="bounded",
        args=(pool, glm_h, glm_a),
    )
    rho = res.x
    lam_h = glm_h.predict(test[_GOAL_FEATS_HOME].values).clip(0.2, 8)
    lam_a = glm_a.predict(test[_GOAL_FEATS_AWAY].values).clip(0.2, 8)
    return _dc_probs(lam_h, lam_a, rho), rho


# ── Backtest runner ────────────────────────────────────────────────────────────

def _run_bt(label, pool, test, tracks):
    outcomes = test["outcome"].tolist()
    results = {}
    for name, fn in tracks.items():
        try:
            hda = fn(pool, test)
            results[name] = _evaluate(name, hda, outcomes)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            results[name] = None
    return results


def _print_table(bt_label, results_by_track):
    print(f"\n── {bt_label} ──")
    fmt = f"  {'model':<22}  {'log-loss':>9}  {'accuracy':>9}  {'draw recall':>12}"
    print(fmt)
    print("  " + "-" * (len(fmt) - 2))
    for name, r in results_by_track.items():
        if r is None:
            print(f"  {name:<22}  {'ERROR':>9}")
            continue
        dr = f"{r['draw_recall']:.1%}" if not np.isnan(r["draw_recall"]) else "—"
        print(f"  {name:<22}  {r['ll']:>9.4f}  {r['acc']:>9.1%}  {dr:>12}")


# ── Logging ────────────────────────────────────────────────────────────────────

def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--log",     action="store_true")
    parser.add_argument("--only",    nargs="+",
                        choices=["tabpfn_3class", "tabpfn_2stage", "poisson", "dixon_coles"],
                        help="Run only these tracks")
    args = parser.parse_args()

    df    = load_data(refresh=args.refresh)
    feats = build_features(df)

    # BT fixtures
    wc22      = wc_matches(feats, 2022)
    wc26      = wc_matches(feats, 2026)
    bt1_test  = wc_group_stage(wc22)
    bt2_test  = wc_knockout(wc22)
    bt3_test  = wc_group_rounds(wc26, max_round=2)

    pool_bt12 = _pool(feats, WC2022_START)
    pool_bt3  = _pool(feats, WC2026_START)

    # Track registry
    all_tracks = {
        "tabpfn_3class": lambda pool, test: _tabpfn_3class(pool, test),
        "tabpfn_2stage": lambda pool, test: _tabpfn_2stage(pool, test),
        "poisson":       lambda pool, test: _poisson_predict(*_train_poisson(pool), test),
        "dixon_coles":   lambda pool, test: _dixon_coles_predict(pool, test)[0],
    }
    if args.only:
        all_tracks = {k: v for k, v in all_tracks.items() if k in args.only}

    print(f"\nTracks: {list(all_tracks)}")
    print(f"Training pool: BT1/BT2={len(pool_bt12)}  BT3={len(pool_bt3)}")

    bt_configs = [
        ("BT1 — WC 2022 group (48)", pool_bt12, bt1_test),
        ("BT2 — WC 2022 knockout (16)", pool_bt12, bt2_test),
        ("BT3 — WC 2026 R1-2 (48)", pool_bt3,  bt3_test),
    ]

    all_results = {}
    for bt_label, pool, test in bt_configs:
        r = _run_bt(bt_label, pool, test, all_tracks)
        all_results[bt_label] = r
        _print_table(bt_label, r)

    if args.log:
        for name in all_tracks:
            run_name = f"draw_{name}_{TODAY_STR}"
            entry = {
                "run":       run_name,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "commit":    git_commit(),
                "features":  FEATURES,
            }
            for bt_label, r in all_results.items():
                bt_key = bt_label.split()[0].lower()
                if r.get(name):
                    entry[f"{bt_key}_ll"]  = round(r[name]["ll"],  4)
                    entry[f"{bt_key}_acc"] = round(r[name]["acc"], 4)
                    entry[f"{bt_key}_draw_recall"] = (
                        round(r[name]["draw_recall"], 4)
                        if not np.isnan(r[name]["draw_recall"]) else None
                    )
            _upsert(run_name, entry)
            print(f"\n  logged '{run_name}'")


if __name__ == "__main__":
    main()
