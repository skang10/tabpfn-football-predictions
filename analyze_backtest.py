"""Analyze and compare backtest results across experiments from experiments.jsonl."""
import json
import pandas as pd
import numpy as np

OUTCOMES = ["home_win", "draw", "away_win"]


def load_experiments(path="experiments.jsonl"):
    experiments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                experiments.append(json.loads(line))
    return experiments


def per_match_df(exp):
    return pd.DataFrame(exp["per_match"])


def prediction_distribution(df):
    return df["predicted"].value_counts(normalize=True).reindex(OUTCOMES, fill_value=0).round(3)


def actual_distribution(df):
    return df["outcome"].value_counts(normalize=True).reindex(OUTCOMES, fill_value=0).round(3)


def accuracy_by_outcome(df):
    """Accuracy broken down by actual outcome class."""
    rows = {}
    for outcome in OUTCOMES:
        subset = df[df["outcome"] == outcome]
        if len(subset):
            rows[outcome] = {
                "n":        len(subset),
                "correct":  subset["correct"].sum(),
                "accuracy": round(subset["correct"].mean(), 3),
            }
        else:
            rows[outcome] = {"n": 0, "correct": 0, "accuracy": float("nan")}
    return pd.DataFrame(rows).T


def confusion_matrix(df):
    """Rows = actual, cols = predicted."""
    return pd.crosstab(
        df["outcome"], df["predicted"],
        rownames=["actual"], colnames=["predicted"]
    ).reindex(index=OUTCOMES, columns=OUTCOMES, fill_value=0)


def print_separator(title=""):
    width = 60
    if title:
        print(f"\n{'=' * width}")
        print(f"  {title}")
        print('=' * width)
    else:
        print("-" * width)


def main():
    experiments = load_experiments()

    if not experiments:
        print("No experiments found in experiments.jsonl")
        return

    # ── Summary table ────────────────────────────────────────────────────────
    print_separator("Summary")
    summary_rows = []
    for exp in experiments:
        df = per_match_df(exp)
        pred_dist = prediction_distribution(df)
        summary_rows.append({
            "run":          exp["run"],
            "commit":       exp["commit"],
            "accuracy":     f"{exp['accuracy']:.1%}",
            "log_loss":     exp["log_loss"],
            "n":            exp["n_matches"],
            "pred_home%":   f"{pred_dist['home_win']:.0%}",
            "pred_draw%":   f"{pred_dist['draw']:.0%}",
            "pred_away%":   f"{pred_dist['away_win']:.0%}",
        })
    print(pd.DataFrame(summary_rows).to_string(index=False))

    # ── Per-experiment detail ─────────────────────────────────────────────────
    for exp in experiments:
        print_separator(f"Experiment: {exp['run']}  (commit {exp['commit']})")
        df = per_match_df(exp)

        print(f"\nFeatures ({len(exp['features'])}):")
        print("  " + ", ".join(exp["features"]))

        print("\nPrediction vs actual distribution:")
        dist = pd.DataFrame({
            "predicted": prediction_distribution(df),
            "actual":    actual_distribution(df),
        })
        dist["gap"] = (dist["predicted"] - dist["actual"]).round(3)
        print(dist.to_string())

        print("\nAccuracy by actual outcome:")
        print(accuracy_by_outcome(df).to_string())

        print("\nConfusion matrix (actual → predicted):")
        print(confusion_matrix(df).to_string())

    # ── Cross-experiment prediction distribution comparison ───────────────────
    print_separator("Prediction distribution across experiments")
    rows = []
    for exp in experiments:
        df = per_match_df(exp)
        d = prediction_distribution(df)
        rows.append({
            "run":       exp["run"],
            "home_win":  d["home_win"],
            "draw":      d["draw"],
            "away_win":  d["away_win"],
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nActual outcome distribution (same for all):")
    df0 = per_match_df(experiments[0])
    print(actual_distribution(df0).to_string())


if __name__ == "__main__":
    main()
