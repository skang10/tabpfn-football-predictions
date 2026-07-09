"""Ablation: which Elo-momentum window(s) actually help, if any?

elo_momentum bundled elo_mom3_diff + elo_mom5_diff on top of wc_recent_form
without checking whether both windows pull weight. This sweep isolates each
window (plus a single-last-game delta, elo_mom1_diff) to see what's signal vs.
noise, the same way sweep_recent_form.py did for the form3/ewform/momentum trio.

BT2 (WC22 knockout) is the competition-relevant target.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from features import load_data, wc_matches, wc_group_stage, wc_knockout, wc_group_rounds, MAX_TRAIN
from predict import train, predict_proba, build_features, TRAIN_START, BASE_FEATURES, RECENCY_FEATURES

WC2022_START = pd.Timestamp("2022-11-20")
WC2026_START = pd.Timestamp("2026-06-11")

BASE = BASE_FEATURES + RECENCY_FEATURES  # = wc_recent_form's FEATURES (23)

SETS = {
    "base (wc_recent_form)": BASE,
    "+mom1":                 BASE + ["elo_mom1_diff"],
    "+mom3":                 BASE + ["elo_mom3_diff"],
    "+mom5":                 BASE + ["elo_mom5_diff"],
    "+mom1+mom3":            BASE + ["elo_mom1_diff", "elo_mom3_diff"],
    "+mom3+mom5 (current)":  BASE + ["elo_mom3_diff", "elo_mom5_diff"],
    "+mom1+mom3+mom5":       BASE + ["elo_mom1_diff", "elo_mom3_diff", "elo_mom5_diff"],
}


def ll(model, test, feats):
    proba = predict_proba(model, test[feats].values)
    lex = proba[["p_away_win", "p_draw", "p_home_win"]].values
    return log_loss(test["outcome"], lex, labels=["away_win", "draw", "home_win"])


def main():
    df = load_data()
    feats = build_features(df)
    played = feats[feats["outcome"].notna() & (feats["date"] >= TRAIN_START)]

    wc22 = wc_matches(feats, 2022)
    bt1_test, bt2_test = wc_group_stage(wc22), wc_knockout(wc22)
    pool22 = played[played["date"] < WC2022_START].tail(MAX_TRAIN)

    wc26 = wc_matches(feats, 2026)
    bt3_test = wc_group_rounds(wc26, max_round=2)
    pool26 = played[played["date"] < WC2026_START].tail(MAX_TRAIN)

    print(f"{'feature set':>24} | {'n':>3} | {'BT1':>7} {'BT2*':>7} {'BT3':>7}")
    print("-" * 60)
    rows = []
    for name, fs in SETS.items():
        m22 = train(pool22, features=fs)
        m26 = train(pool26, features=fs)
        r = (ll(m22, bt1_test, fs), ll(m22, bt2_test, fs), ll(m26, bt3_test, fs))
        rows.append((name, len(fs), *r))
        print(f"{name:>24} | {len(fs):>3} | {r[0]:>7.4f} {r[1]:>7.4f} {r[2]:>7.4f}")

    base_bt1, base_bt2, base_bt3 = rows[0][2], rows[0][3], rows[0][4]
    print("\nΔ vs base (negative = better):")
    for name, n, b1, b2, b3 in rows[1:]:
        print(f"  {name:>24}  ΔBT1={b1 - base_bt1:+.4f}  ΔBT2={b2 - base_bt2:+.4f}  ΔBT3={b3 - base_bt3:+.4f}")


if __name__ == "__main__":
    main()
