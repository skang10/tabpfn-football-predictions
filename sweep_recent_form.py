"""Ablation: do recency/momentum features (last 3-10 games) improve backtests?

Each feature set retrains TabPFN (columns differ). Conditional draw scaling from
predict_proba still applies (it reads elo_diff at column 0, kept first everywhere).
BT2 (WC22 knockout) is the competition-relevant target.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from features import (
    load_data, wc_matches, wc_group_stage, wc_knockout, wc_group_rounds, MAX_TRAIN,
    FEATURES as BASE,
)
from predict import train, predict_proba, build_features, TRAIN_START

WC2022_START = pd.Timestamp("2022-11-20")
WC2026_START = pd.Timestamp("2026-06-11")

SETS = {
    "base (control)":        BASE,
    "+form3":                BASE + ["form3_diff"],
    "+momentum":             BASE + ["momentum_diff"],
    "+ewform":               BASE + ["ewform_diff"],
    "+form3+momentum":       BASE + ["form3_diff", "momentum_diff"],
    "+ewform+momentum":      BASE + ["ewform_diff", "momentum_diff"],
    "+form3+ewform+momentum": BASE + ["form3_diff", "ewform_diff", "momentum_diff"],
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

    base_bt2 = rows[0][3]
    print("\nΔBT2 vs base (negative = better):")
    for name, n, b1, b2, b3 in rows[1:]:
        print(f"  {name:>24}  ΔBT2={b2 - base_bt2:+.4f}  ΔBT3={b3 - rows[0][4]:+.4f}")


if __name__ == "__main__":
    main()
