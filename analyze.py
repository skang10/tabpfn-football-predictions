"""Statistical analysis of World Cup draw rates to inform feature engineering."""
import pandas as pd
import numpy as np


def load_wc(path="results.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    # Only assign outcome where scores are known (avoid NaN → draw bug)
    df["outcome"] = np.where(
        df["home_score"].isna(), None,
        np.where(df["home_score"] > df["away_score"], "home_win",
        np.where(df["home_score"] < df["away_score"], "away_win", "draw"))
    )
    t = df["tournament"].str.lower()
    return df[t.str.contains("world cup") & ~t.str.contains("qualif") & df["outcome"].notna()].copy()


def tag_rounds(wc):
    """Tag each match with its group-stage round (1/2/3) or 'knockout'."""
    rounds = []
    for year, group in wc.groupby("year"):
        group = group.sort_values("date").reset_index(drop=True)
        n_group = 72 if year == 2026 else 48   # 2026: 12×6, others: 8×6
        per_round = n_group // 3
        for i, row in group.iterrows():
            if i < per_round:
                rounds.append("round_1")
            elif i < 2 * per_round:
                rounds.append("round_2")
            elif i < n_group:
                rounds.append("round_3")
            else:
                rounds.append("knockout")
    wc = wc.sort_values(["year", "date"]).reset_index(drop=True)
    wc["round"] = rounds
    return wc


def draw_rate(series):
    return (series == "draw").mean()


def main():
    wc = load_wc()
    wc = tag_rounds(wc)

    # ── 1. Draw rate by year (all matches) ───────────────────────────────────
    print("=" * 60)
    print("1. World Cup draw rate by year (all matches)")
    print("=" * 60)
    by_year = (
        wc.groupby("year")["outcome"]
        .agg(total="count", draws=lambda x: (x == "draw").sum())
        .assign(draw_rate=lambda d: (d["draws"] / d["total"]).round(3))
    )
    print(by_year.to_string())

    # ── 2. Group stage draw rate by round (1998–2026) ─────────────────────────
    print()
    print("=" * 60)
    print("2. Group stage draw rate by round (1998–2026)")
    print("=" * 60)
    gs = wc[wc["year"] >= 1998]
    by_round = (
        gs.groupby(["year", "round"])["outcome"]
        .agg(total="count", draws=lambda x: (x == "draw").sum())
        .assign(draw_rate=lambda d: (d["draws"] / d["total"]).round(3))
    )
    print(by_round.to_string())

    # ── 3. Historical baseline vs 2026 ───────────────────────────────────────
    print()
    print("=" * 60)
    print("3. Historical group stage (rounds 1+2) vs WC2026")
    print("=" * 60)
    hist = wc[wc["year"].between(1998, 2022) & wc["round"].isin(["round_1", "round_2"])]
    wc26 = wc[wc["year"] == 2026]

    hist_rate = draw_rate(hist["outcome"])
    r1_rate   = draw_rate(wc26[wc26["round"] == "round_1"]["outcome"])
    r2_rate   = draw_rate(wc26[wc26["round"] == "round_2"]["outcome"])
    r3_rate   = draw_rate(wc26[wc26["round"] == "round_3"]["outcome"])
    r12_rate  = draw_rate(wc26[wc26["round"].isin(["round_1", "round_2"])]["outcome"])

    n_r3 = len(wc26[wc26["round"] == "round_3"])
    r3_note = "(incomplete — dataset may not cover all round 3 matches)" if n_r3 < 24 else ""
    print(f"Historical rounds 1+2 draw rate (1998–2022): {hist_rate:.1%}  (n={len(hist)})")
    print(f"WC2026 round 1 draw rate:                     {r1_rate:.1%}  (n={len(wc26[wc26['round']=='round_1'])})")
    print(f"WC2026 round 2 draw rate:                     {r2_rate:.1%}  (n={len(wc26[wc26['round']=='round_2'])})")
    print(f"WC2026 round 3 draw rate:                     {r3_rate:.1%}  (n={n_r3})  {r3_note}")
    print(f"WC2026 rounds 1+2 combined:                   {r12_rate:.1%}  (n={len(wc26[wc26['round'].isin(['round_1','round_2'])])})")
    print(f"Delta vs historical:                          +{r12_rate - hist_rate:.1%}")

    # ── 4. Draw scoreline breakdown in WC2026 ────────────────────────────────
    print()
    print("=" * 60)
    print("4. WC2026 draw scoreline breakdown")
    print("=" * 60)
    draws_26 = wc26[wc26["outcome"] == "draw"]
    scorelines = draws_26["home_score"].value_counts().sort_index().rename("count")
    scorelines.index = [f"{int(s)}-{int(s)}" for s in scorelines.index]
    print(scorelines.to_string())
    print(f"Total: {len(draws_26)} draws  ({draw_rate(wc26['outcome']):.1%} of all WC2026 matches)")

    # ── 5. Outcome distribution comparison ───────────────────────────────────
    print()
    print("=" * 60)
    print("5. Outcome distribution: historical rounds 1+2 vs WC2026 rounds 1+2")
    print("=" * 60)
    comp = pd.DataFrame({
        "historical (1998–2022)": hist["outcome"].value_counts(normalize=True).round(3),
        "WC2026":                 wc26[wc26["round"].isin(["round_1","round_2"])]["outcome"].value_counts(normalize=True).round(3),
    }).fillna(0)
    print(comp.to_string())


if __name__ == "__main__":
    main()
