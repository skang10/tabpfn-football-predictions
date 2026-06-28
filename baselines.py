"""Baseline model comparison.

Five reference models evaluated on BT1/BT2/BT3:
  uniform             — [1/3, 1/3, 1/3] every match, no information
  always_home         — [0.980, 0.010, 0.010] every match, ignores draw/away
  elo_logistic        — multinomial logistic regression on elo_diff only
  lr_all_features     — multinomial logistic regression on all current FEATURES
  tabpfn_all_features — TabPFN on all current FEATURES

Run names written to experiments.jsonl follow the pattern:
  <model>_<YYYYMMDD>  e.g. uniform_20260628

Usage:
    uv run baselines.py [--log] [--refresh] [--only <name> ...]

    --log     append results to experiments.jsonl (skipped by default)
    --refresh re-download dataset
    --only    run only the listed baseline(s)
"""
import argparse
import json
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    load_data, build_features,
    FEATURES, TRAIN_START, MAX_TRAIN,
    wc_matches, wc_group_stage, wc_knockout, wc_group_rounds,
)
from predict import train as tabpfn_train, predict_proba as tabpfn_predict_proba
from analyze_backtest import (
    print_summary, print_diagnostics,
    plot_logloss_grouped, plot_logloss_heatmap,
    plot_draw_distribution, plot_outcome_distribution,
    plot_calibration_curve,
)

EXPERIMENTS_LOG = "experiments.jsonl"
WC2022_START    = pd.Timestamp("2022-11-20")
WC2026_START    = pd.Timestamp("2026-06-11")
TODAY_STR       = datetime.now().strftime("%Y%m%d")


# ── Model definitions ─────────────────────────────────────────────────────────

def _uniform_predict(model, X):
    return pd.DataFrame({
        "p_home_win": np.full(len(X), 1/3),
        "p_draw":     np.full(len(X), 1/3),
        "p_away_win": np.full(len(X), 1/3),
    })


def _always_home_predict(model, X):
    return pd.DataFrame({
        "p_home_win": np.full(len(X), 0.980),
        "p_draw":     np.full(len(X), 0.010),
        "p_away_win": np.full(len(X), 0.010),
    })


def _lr_predict(model, X):
    proba   = model.predict_proba(X)
    classes = list(model.classes_)
    return pd.DataFrame({
        "p_home_win": proba[:, classes.index("home_win")],
        "p_draw":     proba[:, classes.index("draw")],
        "p_away_win": proba[:, classes.index("away_win")],
    })


_ELO_IDX = FEATURES.index("elo_diff")


def _elo_lr_train(pool):
    return LogisticRegression(max_iter=1000, random_state=42).fit(
        pool[["elo_diff"]].values, pool["outcome"].values)


def _elo_lr_predict(model, X):
    return _lr_predict(model, X[:, [_ELO_IDX]])


def _lr_all_train(pool):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    ).fit(pool[FEATURES].values, pool["outcome"].values)


BASELINES = {
    "uniform":             {"train": lambda pool: None, "predict": _uniform_predict},
    "always_home":         {"train": lambda pool: None, "predict": _always_home_predict},
    "elo_logistic":        {"train": _elo_lr_train,     "predict": _elo_lr_predict},
    "lr_all_features":     {"train": _lr_all_train,     "predict": _lr_predict},
    "tabpfn_all_features": {"train": tabpfn_train,      "predict": tabpfn_predict_proba},
}


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(label, test, model, predict_fn):
    if not len(test):
        print(f"\n{label}: no matches found")
        return None

    X        = test[FEATURES].values
    proba_df = predict_fn(model, X)
    proba_df = proba_df / proba_df.sum(axis=1).values[:, None]

    proba_lex = proba_df[["p_away_win", "p_draw", "p_home_win"]].values
    ll  = log_loss(test["outcome"], proba_lex, labels=["away_win", "draw", "home_win"])

    label_arr = np.array(["home_win", "draw", "away_win"])
    proba_hda = proba_df[["p_home_win", "p_draw", "p_away_win"]].values
    pred      = label_arr[proba_hda.argmax(1)]
    acc       = accuracy_score(test["outcome"], pred)

    actual       = test["outcome"].values
    draws_actual = (actual == "draw").sum()
    draws_pred   = (pred   == "draw").sum()
    draw_recall  = (((pred == "draw") & (actual == "draw")).sum() / draws_actual
                    if draws_actual else np.nan)
    mean_p_draw  = proba_df["p_draw"].mean()

    outcome_to_p = {
        "home_win": proba_df["p_home_win"].values,
        "draw":     proba_df["p_draw"].values,
        "away_win": proba_df["p_away_win"].values,
    }
    true_probs            = np.array([outcome_to_p[o][i] for i, o in enumerate(actual)])
    true_outcome_avg_prob = true_probs.mean()

    print(f"\n── {label} ({len(test)} matches) ──")
    print(f"  log-loss              {ll:.4f}")
    print(f"  accuracy              {acc:.1%}")
    dr_str = f"{draw_recall:.1%}" if not np.isnan(draw_recall) else "—"
    print(f"  draw recall           {dr_str}  ({draws_actual} actual / {draws_pred} predicted)")
    print(f"  mean p(draw)          {mean_p_draw:.3f}")
    print(f"  true outcome avg prob {true_outcome_avg_prob:.3f}")

    _ph = proba_df["p_home_win"].round(3)
    _pd = proba_df["p_draw"].round(3)
    _pa = (1.0 - _ph - _pd).round(3)

    pm = test[["date", "home_team", "away_team", "outcome"]].copy()
    pm["predicted"]  = pred
    pm["p_home_win"] = _ph.values
    pm["p_draw"]     = _pd.values
    pm["p_away_win"] = _pa.values
    pm["p_true"]     = true_probs.round(3)
    pm["correct"]    = pm["outcome"] == pm["predicted"]

    return {
        "ll":                    ll,
        "acc":                   acc,
        "draw_recall":           draw_recall,
        "mean_p_draw":           mean_p_draw,
        "true_outcome_avg_prob": true_outcome_avg_prob,
        "pm":                    pm,
    }


def _bt_entry(result):
    if result is None:
        return None
    pm = result["pm"]
    dr = result["draw_recall"]
    return {
        "log_loss":              round(result["ll"], 4),
        "accuracy":              round(result["acc"], 4),
        "draw_recall":           round(dr, 4) if not np.isnan(dr) else None,
        "mean_p_draw":           round(result["mean_p_draw"], 4),
        "true_outcome_avg_prob": round(result["true_outcome_avg_prob"], 4),
        "n_matches":             len(pm),
        "per_match":             (pm.assign(date=pm["date"].dt.strftime("%Y-%m-%d"))
                                    .to_dict(orient="records")),
    }


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _upsert_experiment(run_name, entry):
    """Write entry to experiments.jsonl, replacing any existing row with the same run name."""
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


def log_entry(run_name, bt1, bt2, bt3):
    entry = {
        "run":       run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit":    git_commit(),
        "features":  FEATURES,
        "bt1":       _bt_entry(bt1),
        "bt2":       _bt_entry(bt2),
        "bt3":       _bt_entry(bt3),
    }
    _upsert_experiment(run_name, entry)
    print(f"  → logged as '{run_name}'")
    return entry


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",     action="store_true", help="Append results to experiments.jsonl")
    parser.add_argument("--refresh", action="store_true", help="Re-download dataset")
    parser.add_argument("--only",    nargs="+",           help="Run only these baseline(s)")
    args = parser.parse_args()

    to_run = list(BASELINES.keys())
    if args.only:
        unknown = [b for b in args.only if b not in BASELINES]
        if unknown:
            print(f"Unknown baselines: {unknown}. Available: {list(BASELINES.keys())}")
            return
        to_run = [b for b in BASELINES if b in args.only]

    df     = load_data(refresh=args.refresh)
    feats  = build_features(df)
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]
    print(f"Latest game in dataset: {df['date'].max().date()}")

    wc22_all = wc_matches(feats, 2022)
    wc26_all = wc_matches(feats, 2026)
    wc26_r12 = wc_group_rounds(wc26_all, max_round=2)

    has_wc22 = len(wc22_all) > 0
    has_wc26 = len(wc26_r12) > 0

    pool22 = played[played["date"] < WC2022_START].tail(MAX_TRAIN) if has_wc22 else None
    pool26 = played[played["date"] < WC2026_START].tail(MAX_TRAIN) if has_wc26 else None

    all_exps = []

    for name in to_run:
        cfg      = BASELINES[name]
        run_name = f"{name}_{TODAY_STR}"
        print(f"\n{'=' * 60}\n  Baseline: {run_name}\n{'=' * 60}")

        bt1 = bt2 = bt3 = None

        if has_wc22:
            print(f"\nTraining on {len(pool22)} matches (pre-WC2022) ...")
            model22 = cfg["train"](pool22)
            bt1 = evaluate("BT1 — WC 2022 group stage", wc_group_stage(wc22_all), model22, cfg["predict"])
            bt2 = evaluate("BT2 — WC 2022 knockout",    wc_knockout(wc22_all),    model22, cfg["predict"])
        else:
            print("No WC2022 data (try --refresh).")

        if has_wc26:
            print(f"\nTraining on {len(pool26)} matches (pre-WC2026) ...")
            model26 = cfg["train"](pool26)
            bt3 = evaluate("BT3 — WC 2026 group rounds 1-2", wc26_r12, model26, cfg["predict"])
        else:
            print("No WC2026 data (try --refresh).")

        if args.log:
            log_entry(run_name, bt1, bt2, bt3)

        exp = {
            "run":    run_name,
            "commit": git_commit(),
            "bt1":    _bt_entry(bt1),
            "bt2":    _bt_entry(bt2),
            "bt3":    _bt_entry(bt3),
        }
        all_exps.append(exp)

    print(f"\n{'=' * 60}\n  Summary\n{'=' * 60}")
    print_summary(all_exps)
    print_diagnostics(all_exps)

    print(f"\n{'=' * 60}\n  Generating plots\n{'=' * 60}")
    plot_logloss_grouped(all_exps)
    plot_logloss_heatmap(all_exps)
    plot_draw_distribution(all_exps)
    plot_outcome_distribution(all_exps)
    plot_calibration_curve(all_exps)
    print(f"\nAll plots saved to ./plots/")


if __name__ == "__main__":
    main()
