"""Baseline model (3-class TabPFN classifier) + forward predictions.

Exposes the standard interface used by backtest.py:
    train(pool)          → fitted model
    predict_proba(model, X) → DataFrame[p_home_win, p_draw, p_away_win]

Run as a script to generate forward predictions for upcoming fixtures:
    uv run predict.py [--refresh]
"""
import argparse
import json
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from tabpfn_client import TabPFNClassifier

from features import (
    load_data, build_features as _base_build_features,
    FEATURES as BASE_FEATURES, TRAIN_START, MAX_TRAIN, TODAY,
    wc_matches,
)

LLM_CONTEXT_PATH = os.environ.get("LLM_CONTEXT_PATH", "llm_context.csv")

LLM_RAW_COLUMNS = [
    "home_absence_severity",
    "away_absence_severity",
    "home_lineup_uncertainty",
    "away_lineup_uncertainty",
    "home_rotation_risk",
    "away_rotation_risk",
    "home_tactical_edge",
    "away_tactical_edge",
    "llm_confidence",
]

LLM_DIFF_FEATURES = [
    "absence_diff",
    "lineup_uncertainty_diff",
    "rotation_risk_diff",
    "tactical_edge_diff",
]

# Recency / in-tournament-condition features (last 3-10 games). Long-term Elo
# misses the physical & mental condition a team carries *into* the WC, which the
# last few results capture. Added on top of the 20-feature base; the trio wins
# BT2 and BT3 together (see sweep_recent_form.py) though each alone is noise.
RECENCY_FEATURES = ["form3_diff", "ewform_diff", "momentum_diff"]

# Elo momentum: rating change over a team's last 3/5 games. Unlike form3/ewform
# (points-only), this is opponent-strength- and margin-adjusted, since each Elo
# delta already bakes in importance * goal-diff-multiplier * (actual - expected).
# Distinguishes e.g. a narrow win over a strong side from the same scoreline
# against a weak one. See features.py::build_features.
MOMENTUM_FEATURES = ["elo_mom3_diff", "elo_mom5_diff"]
FEATURES = BASE_FEATURES + RECENCY_FEATURES + MOMENTUM_FEATURES

# Conditional draw scaling: evenly-matched teams (small |elo_diff|) draw far more
# often than mismatches, so boost p_draw more when the game is close and taper to
# ~1x for blowouts, instead of the flat draw_k=1.2 applied to every match.
#     draw_k(match) = 1 + DRAW_B * exp(-|elo_diff| / DRAW_S)
# Tuned on BT2 (WC22 knockout); see sweep_cond_draw.py. Set DRAW_B=0 to disable.
DRAW_B = 0.4
DRAW_S = 150.0
_ELO_DIFF_IDX = BASE_FEATURES.index("elo_diff")


def _conditional_draw_k(elo_diff, b=DRAW_B, s=DRAW_S):
    return 1.0 + b * np.exp(-np.abs(np.asarray(elo_diff, dtype=float)) / s)


def _empty_llm_features(index):
    cols = LLM_DIFF_FEATURES + ["llm_confidence"]
    return pd.DataFrame(0.0, index=index, columns=cols)


def _read_llm_context(path=LLM_CONTEXT_PATH):
    """Read cached LLM match context; never calls an LLM at prediction time."""
    if not path or not os.path.exists(path):
        return None

    if path.endswith(".jsonl"):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        ctx = pd.DataFrame(rows)
    else:
        ctx = pd.read_csv(path)

    required = {"date", "home_team", "away_team"}
    missing = required - set(ctx.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    ctx = ctx.copy()
    ctx["date"] = pd.to_datetime(ctx["date"]).dt.strftime("%Y-%m-%d")

    for col in LLM_RAW_COLUMNS + LLM_DIFF_FEATURES:
        if col in ctx:
            ctx[col] = pd.to_numeric(ctx[col], errors="coerce")

    if "absence_diff" not in ctx:
        ctx["absence_diff"] = (
            ctx.get("away_absence_severity", 0.0) - ctx.get("home_absence_severity", 0.0)
        )
    if "lineup_uncertainty_diff" not in ctx:
        ctx["lineup_uncertainty_diff"] = (
            ctx.get("away_lineup_uncertainty", 0.0) - ctx.get("home_lineup_uncertainty", 0.0)
        )
    if "rotation_risk_diff" not in ctx:
        ctx["rotation_risk_diff"] = (
            ctx.get("away_rotation_risk", 0.0) - ctx.get("home_rotation_risk", 0.0)
        )
    if "tactical_edge_diff" not in ctx:
        ctx["tactical_edge_diff"] = (
            ctx.get("home_tactical_edge", 0.0) - ctx.get("away_tactical_edge", 0.0)
        )
    if "llm_confidence" not in ctx:
        ctx["llm_confidence"] = 0.0

    clip_bounds = {
        "absence_diff": (-3.0, 3.0),
        "lineup_uncertainty_diff": (-3.0, 3.0),
        "rotation_risk_diff": (-3.0, 3.0),
        "tactical_edge_diff": (-3.0, 3.0),
        "llm_confidence": (0.0, 1.0),
    }
    for col, bounds in clip_bounds.items():
        ctx[col] = pd.to_numeric(ctx[col], errors="coerce").clip(*bounds).fillna(0.0)

    ctx = ctx.rename(columns={"date": "_llm_date"})
    keep = ["_llm_date", "home_team", "away_team"] + LLM_DIFF_FEATURES + ["llm_confidence"]
    return ctx[keep].drop_duplicates(["_llm_date", "home_team", "away_team"], keep="last")


def build_features(df, llm_context=False, llm_context_path=LLM_CONTEXT_PATH):
    feats = _base_build_features(df)
    if not llm_context:
        return feats

    ctx = _read_llm_context(llm_context_path)
    if ctx is None or ctx.empty:
        return feats.join(_empty_llm_features(feats.index))

    base = feats.copy()
    base["_llm_order"] = np.arange(len(base))
    base["_llm_date"] = base["date"].dt.strftime("%Y-%m-%d")
    merged = base.merge(ctx, on=["_llm_date", "home_team", "away_team"], how="left")
    merged = merged.sort_values("_llm_order").drop(columns=["_llm_order", "_llm_date"])
    merged.index = feats.index
    for col in LLM_DIFF_FEATURES + ["llm_confidence"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged


def _model_features(use_llm_context=False):
    return FEATURES + LLM_DIFF_FEATURES if use_llm_context else FEATURES


def train(pool, features=None):
    features = features or FEATURES
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[features].values, pool["outcome"].values)
    return clf


def predict_proba(model, X):
    """Return a DataFrame with columns p_home_win, p_draw, p_away_win (rows sum to 1).

    Applies conditional draw scaling (larger draw boost for evenly-matched games)
    using elo_diff, which is column `_ELO_DIFF_IDX` of X. DRAW_B=0 disables it.
    """
    proba = model.predict_proba(X)
    proba = proba / proba.sum(axis=1, keepdims=True)
    classes = list(model.classes_)
    hda = np.column_stack([
        proba[:, classes.index("home_win")],
        proba[:, classes.index("draw")],
        proba[:, classes.index("away_win")],
    ])

    if DRAW_B:
        elo_diff = np.asarray(X)[:, _ELO_DIFF_IDX]
        hda[:, 1] *= _conditional_draw_k(elo_diff)
        hda /= hda.sum(axis=1, keepdims=True)

    return pd.DataFrame({
        "p_home_win": hda[:, 0],
        "p_draw":     hda[:, 1],
        "p_away_win": hda[:, 2],
    })


def _round_probs(probs, decimals=4):
    """Round H/D/A probabilities to `decimals` places while preserving a 1.0 row sum.

    Uses the largest-remainder method so rounded rows still sum to exactly 1, and
    guarantees each outcome stays strictly between 0 and 1 (competition requirement).
    """
    scale = 10 ** decimals
    rounded = []
    for row in probs:
        units = np.floor(row * scale).astype(int)
        remainder = int(scale - units.sum())
        fractions = row * scale - units
        for idx in np.argsort(-fractions)[:remainder]:
            units[idx] += 1

        zero_idx = np.where(units == 0)[0]
        for idx in zero_idx:
            donor = int(np.argmax(units))
            if units[donor] <= 1:
                break
            units[idx] = 1
            units[donor] -= 1

        rounded.append(units / scale)
    return np.array(rounded)


def _git_branch():
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        return branch.replace("/", "_")
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh",    action="store_true", help="Re-download dataset")
    parser.add_argument("--date",       default=None,
                        help="Predict fixtures on or after this date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--draw-scale", type=float, default=1.0,
                        help="Extra flat multiplier on p_draw, then renormalize (default 1.0 = "
                             "none). Conditional draw scaling is already baked into predict_proba; "
                             "this stacks on top and is normally left at 1.0.")
    parser.add_argument("--output-dir", default="predictions",
                        help="Directory for generated submission CSVs (default: predictions).")
    parser.add_argument("--llm-context", action="store_true",
                        help="Enable cached LLM context features from llm_context.csv/jsonl.")
    parser.add_argument("--llm-context-path", default=LLM_CONTEXT_PATH,
                        help=f"Path to cached LLM context file (default: {LLM_CONTEXT_PATH}).")
    parser.add_argument("--decimals", type=int, default=4,
                        help="Decimal places for submitted probabilities (default 4). "
                             "Rows still sum to exactly 1 and stay strictly between 0 and 1.")
    args = parser.parse_args()

    from_date = pd.Timestamp(args.date) if args.date else TODAY

    df    = load_data(refresh=args.refresh)
    feats = build_features(df, llm_context=args.llm_context, llm_context_path=args.llm_context_path)
    model_features = _model_features(args.llm_context)
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]
    # Keep the fixture order from results.csv/load_data instead of re-sorting here.
    future = feats[feats["home_score"].isna() & (feats["date"] >= from_date)]

    if not len(future):
        print("No upcoming fixtures — run with --refresh to fetch latest data.")
        return

    today_str  = datetime.now().strftime("%Y%m%d")
    llm_tag    = "_llm" if args.llm_context else ""
    scale_tag  = f"_ds{args.draw_scale:.1f}".replace(".", "") if args.draw_scale != 1.0 else ""
    filename   = os.path.join(args.output_dir, f"submission_{_git_branch()}{llm_tag}{scale_tag}_{today_str}.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    model     = train(played.tail(MAX_TRAIN), features=model_features)
    proba_df  = predict_proba(model, future[model_features].values)

    # Apply draw scaling then renormalize
    hda = proba_df[["p_home_win", "p_draw", "p_away_win"]].values.copy()
    if args.draw_scale != 1.0:
        hda[:, 1] *= args.draw_scale
        hda /= hda.sum(axis=1, keepdims=True)

    label_arr = np.array(["home_win", "draw", "away_win"])
    predicted = label_arr[hda.argmax(1)]

    # Submission format: date, home_team, away_team, p_home_win, p_draw, p_away_win (sums to 1)
    rounded_hda = _round_probs(hda, decimals=args.decimals)
    ph_raw, pd_raw, pa_raw = rounded_hda[:, 0], rounded_hda[:, 1], rounded_hda[:, 2]
    out = future[["date", "home_team", "away_team"]].copy()
    out["p_home_win"] = ph_raw
    out["p_draw"]     = pd_raw
    out["p_away_win"] = pa_raw

    out.to_csv(filename, index=False, float_format=f"%.{args.decimals}f")
    print(f"\n{len(out)} fixture predictions -> {filename}\n")
    for r, pred in zip(out.itertuples(), predicted):
        print(f"  {r.date.date()}  {r.home_team:>22} vs {r.away_team:<22}"
              f"  -> {pred:<9}  H {r.p_home_win:.0%} | D {r.p_draw:.0%} | A {r.p_away_win:.0%}")


if __name__ == "__main__":
    main()
