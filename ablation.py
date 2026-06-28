"""Feature ablation study — TabPFN with different feature subsets.

Trains TabPFN on each BT training pool using a specific feature subset
and evaluates on the fixed test sets.

Feature sets are cumulative (elo → +form → +rest → +home/away → +h2h/ctx)
plus targeted ablations from the full set.

Run names in experiments.jsonl: abl_<name>_YYYYMMDD

Usage:
    uv run ablation.py [--log] [--refresh] [--only <name> ...]
"""
import argparse
import json
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from tabpfn_client import TabPFNClassifier

from features import (
    load_data, build_features,
    TRAIN_START, MAX_TRAIN,
    wc_matches, wc_group_stage, wc_knockout, wc_group_rounds,
)

EXPERIMENTS_LOG = "experiments.jsonl"
WC2022_START    = pd.Timestamp("2022-11-20")
WC2026_START    = pd.Timestamp("2026-06-11")
TODAY_STR       = datetime.now().strftime("%Y%m%d")


# ── Feature groups ─────────────────────────────────────────────────────────────

_ELO      = ["elo_diff", "home_elo", "away_elo"]
_FORM     = ["form5_diff", "form10_diff", "home_form5", "away_form5"]
_REST     = ["home_rest", "away_rest"]
_HOMEAWAY = [
    "home_winrate", "away_winrate",
    "home_gf5", "away_gf5", "home_ga5", "away_ga5", "gd10_diff",
    "home_streak", "away_streak", "home_played", "away_played",
]
_H2H      = ["h2h_n", "h2h_home_winrate", "h2h_draw_rate", "h2h_gd"]
_CTX      = ["neutral", "importance"]
_WC       = ["abs_elo_diff", "host_adv_diff", "concacaf_adv_diff", "same_continent_adv_diff"]

_ALL_CURRENT = _ELO + _FORM + _REST + _HOMEAWAY + _H2H + _CTX  # 26 features (same as main)

ABLATION_SETS = {
    "elo_only":          ["elo_diff"],
    "elo_features":      _ELO,
    "elo_form":          _ELO + _FORM,
    "elo_form_rest":     _ELO + _FORM + _REST,
    "elo_form_homeaway": _ELO + _FORM + _REST + _HOMEAWAY,
    "all_current":       _ALL_CURRENT,
    "no_h2h":            _ELO + _FORM + _REST + _HOMEAWAY + _CTX,
    "no_streak":         [f for f in _ALL_CURRENT if "streak" not in f],
    "wc_context":        _ALL_CURRENT + _WC,
}


# ── Model ──────────────────────────────────────────────────────────────────────

def _train(pool, features):
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[features].values, pool["outcome"].values)
    return clf


def _predict_proba(model, X):
    proba   = model.predict_proba(X)
    proba   = proba / proba.sum(axis=1, keepdims=True)
    classes = list(model.classes_)
    return pd.DataFrame({
        "p_home_win": proba[:, classes.index("home_win")],
        "p_draw":     proba[:, classes.index("draw")],
        "p_away_win": proba[:, classes.index("away_win")],
    })


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(label, test, model, features):
    if not len(test):
        print(f"\n{label}: no matches found")
        return None

    proba_df = _predict_proba(model, test[features].values)
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

    wrong = pm[~pm["correct"]]
    if len(wrong):
        print(f"  wrong predictions ({len(wrong)}):")
        for r in wrong.itertuples():
            date = r.date.date() if hasattr(r.date, "date") else r.date
            print(f"    {date}  {r.home_team:>22} vs {r.away_team:<22}"
                  f"  actual={r.outcome:<9}  pred={r.predicted:<9}"
                  f"  H {r.p_home_win:.2f} D {r.p_draw:.2f} A {r.p_away_win:.2f}")

    return {
        "ll": ll, "acc": acc, "draw_recall": draw_recall,
        "mean_p_draw": mean_p_draw, "true_outcome_avg_prob": true_outcome_avg_prob,
        "pm": pm,
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


def _print_summary(results):
    header = f"  {'ablation set':<25}  {'n':>3}  {'BT1 ll':>7}  {'BT2 ll':>7}  {'BT3 ll':>7}  {'BT1 acc':>7}  {'BT2 acc':>7}  {'BT3 acc':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, n_feats, bt1, bt2, bt3 in results:
        def _ll(b):  return f"{b['log_loss']:.4f}" if b else "—"
        def _ac(b):  return f"{b['accuracy']:.1%}"  if b else "—"
        print(f"  {name:<25}  {n_feats:>3}  {_ll(bt1):>7}  {_ll(bt2):>7}  {_ll(bt3):>7}"
              f"  {_ac(bt1):>7}  {_ac(bt2):>7}  {_ac(bt3):>7}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",     action="store_true", help="Write results to experiments.jsonl")
    parser.add_argument("--refresh", action="store_true", help="Re-download dataset")
    parser.add_argument("--only",    nargs="+",           help="Run only these ablation(s)")
    args = parser.parse_args()

    to_run = list(ABLATION_SETS.keys())
    if args.only:
        unknown = [n for n in args.only if n not in ABLATION_SETS]
        if unknown:
            print(f"Unknown ablations: {unknown}. Available: {list(ABLATION_SETS.keys())}")
            return
        to_run = [n for n in ABLATION_SETS if n in args.only]

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

    summary_rows = []

    for name in to_run:
        features = ABLATION_SETS[name]
        run_name = f"abl_{name}_{TODAY_STR}"
        print(f"\n{'=' * 60}")
        print(f"  {run_name}  ({len(features)} features)")
        print(f"  {features}")
        print(f"{'=' * 60}")

        bt1 = bt2 = bt3 = None

        if has_wc22:
            print(f"\nTraining on {len(pool22)} matches (pre-WC2022) ...")
            m22 = _train(pool22, features)
            bt1 = evaluate("BT1 — WC 2022 group stage", wc_group_stage(wc22_all), m22, features)
            bt2 = evaluate("BT2 — WC 2022 knockout",    wc_knockout(wc22_all),    m22, features)

        if has_wc26:
            print(f"\nTraining on {len(pool26)} matches (pre-WC2026) ...")
            m26 = _train(pool26, features)
            bt3 = evaluate("BT3 — WC 2026 group rounds 1-2", wc26_r12, m26, features)

        e1, e2, e3 = _bt_entry(bt1), _bt_entry(bt2), _bt_entry(bt3)
        summary_rows.append((name, len(features), e1, e2, e3))

        if args.log:
            entry = {
                "run":       run_name,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "commit":    git_commit(),
                "features":  features,
                "bt1":       e1,
                "bt2":       e2,
                "bt3":       e3,
            }
            _upsert_experiment(run_name, entry)
            print(f"  → logged as '{run_name}'")

    print(f"\n{'=' * 60}\n  Ablation Summary\n{'=' * 60}")
    _print_summary(summary_rows)


if __name__ == "__main__":
    main()
