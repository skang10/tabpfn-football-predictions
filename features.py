"""Shared data loading, feature engineering, and WC match selectors."""
import os
import pandas as pd
import numpy as np
from collections import defaultdict

TODAY = pd.Timestamp.now().normalize()
TRAIN_START = pd.Timestamp("2014-01-01")
MAX_TRAIN = 3000
HOME_ADV = 65.0
DATA       = "results.csv"
GOALS_DATA = "goalscorers.csv"
RAW_URL    = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
GOALS_URL  = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"

FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "form5_diff", "form10_diff", "home_form5", "away_form5",
    "home_rest", "away_rest",
    "home_winrate", "away_winrate",
    "home_gf5", "away_gf5", "home_ga5", "away_ga5", "gd10_diff",
    "home_streak", "away_streak", "home_played", "away_played",
]


def importance(t):
    t = t.lower()
    if "world cup" in t and "qual" not in t:
        return 60.0
    if "confederations" in t:
        return 50.0
    if any(k in t for k in ["uefa euro", "copa am", "african cup", "asian cup",
                             "gold cup", "nations league", "oceania nations"]):
        return 45.0
    if "qualif" in t:
        return 35.0
    if "friendly" in t:
        return 20.0
    return 30.0


# ── World Cup context ─────────────────────────────────────────────────────────

_WC_HOSTS = {
    2026: {"United States", "Canada", "Mexico"},
    2022: {"Qatar"},
    2018: {"Russia"},
    2014: {"Brazil"},
    2010: {"South Africa"},
    2006: {"Germany"},
    2002: {"South Korea", "Japan"},
    1998: {"France"},
    1994: {"United States"},
    1990: {"Italy"},
    1986: {"Mexico"},
    1982: {"Spain"},
    1978: {"Argentina"},
    1974: {"West Germany"},
    1970: {"Mexico"},
    1966: {"England"},
    1962: {"Chile"},
    1958: {"Sweden"},
    1954: {"Switzerland"},
    1950: {"Brazil"},
}

_WC_HOST_CONF = {
    2026: "CONCACAF", 2022: "AFC",      2018: "UEFA",     2014: "CONMEBOL",
    2010: "CAF",      2006: "UEFA",     2002: "AFC",      1998: "UEFA",
    1994: "CONCACAF", 1990: "UEFA",     1986: "CONCACAF", 1982: "UEFA",
    1978: "CONMEBOL", 1974: "UEFA",     1970: "CONCACAF", 1966: "UEFA",
    1962: "CONMEBOL", 1958: "UEFA",     1954: "UEFA",     1950: "CONMEBOL",
}

_TEAM_CONF = {
    **{t: "UEFA" for t in [
        "France", "Germany", "West Germany", "German DR", "Yugoslavia",
        "Spain", "England", "Italy", "Portugal", "Netherlands", "Belgium",
        "Croatia", "Poland", "Switzerland", "Denmark", "Sweden", "Norway",
        "Serbia", "Ukraine", "Russia", "Soviet Union", "Austria",
        "Czech Republic", "Czechoslovakia", "Hungary", "Romania", "Bulgaria",
        "Slovakia", "Slovenia", "Greece", "Turkey", "Scotland", "Wales",
        "Israel", "Northern Ireland", "Republic of Ireland", "Finland",
        "Iceland", "North Macedonia", "Albania", "Bosnia and Herzegovina",
        "Montenegro", "Georgia", "Armenia", "Azerbaijan", "Kosovo",
        "Luxembourg", "Estonia", "Latvia", "Lithuania", "Belarus",
        "Moldova", "Faroe Islands", "Malta", "Cyprus", "Andorra",
        "Liechtenstein", "San Marino", "Gibraltar",
    ]},
    **{t: "CONMEBOL" for t in [
        "Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Peru",
        "Ecuador", "Paraguay", "Bolivia", "Venezuela",
    ]},
    **{t: "CONCACAF" for t in [
        "United States", "Canada", "Mexico", "Costa Rica", "Honduras",
        "Panama", "Jamaica", "El Salvador", "Trinidad and Tobago",
        "Guatemala", "Haiti", "Cuba", "Belize", "Nicaragua",
        "Dominican Republic", "Barbados", "Bermuda", "Guyana",
        "Grenada", "Suriname", "Curacao", "Curaçao",
    ]},
    **{t: "AFC" for t in [
        "Japan", "South Korea", "Iran", "Saudi Arabia", "Australia",
        "Qatar", "China", "Iraq", "United Arab Emirates", "UAE",
        "Oman", "Bahrain", "Kuwait", "Jordan", "Lebanon", "Syria",
        "Vietnam", "Thailand", "Indonesia", "Malaysia", "Philippines",
        "India", "Uzbekistan", "Kyrgyzstan", "Tajikistan", "Yemen",
        "Palestine", "North Korea",
    ]},
    **{t: "CAF" for t in [
        "Senegal", "Morocco", "Nigeria", "Ghana", "Cameroon", "Tunisia",
        "Algeria", "Egypt", "Ivory Coast", "Côte d'Ivoire",
        "South Africa", "Mali", "Burkina Faso",
        "DR Congo", "Democratic Republic of the Congo",
        "Zambia", "Zimbabwe", "Tanzania", "Kenya", "Ethiopia",
        "Uganda", "Angola", "Mozambique", "Namibia", "Benin",
        "Guinea", "Guinea-Bissau", "Liberia", "Sierra Leone", "Togo",
        "Gabon", "Equatorial Guinea", "Congo", "Botswana",
        "Cape Verde", "Comoros", "Gambia", "Libya", "Mauritania",
        "Niger", "Rwanda", "Sudan", "South Sudan",
    ]},
    **{t: "OFC" for t in [
        "New Zealand", "Fiji", "Papua New Guinea", "Vanuatu",
        "Solomon Islands", "Tahiti", "New Caledonia",
    ]},
}


def _wc_context(home, away, tournament, year):
    """Return (host_adv_diff, concacaf_adv_diff, same_continent_adv_diff).
    All three are 0 for non-WC matches.
    """
    t = tournament.lower()
    if "fifa world cup" not in t or "qualif" in t:
        return 0, 0, 0

    hosts     = _WC_HOSTS.get(year, set())
    host_conf = _WC_HOST_CONF.get(year)
    h_conf    = _TEAM_CONF.get(home)
    a_conf    = _TEAM_CONF.get(away)

    host_adv  = int(home in hosts)      - int(away in hosts)
    same_cont = (int(h_conf == host_conf) - int(a_conf == host_conf)) if host_conf else 0

    # CONCACAF advantage: active only when WC is hosted in CONCACAF territory
    if host_conf == "CONCACAF":
        concacaf_adv = int(h_conf == "CONCACAF") - int(a_conf == "CONCACAF")
    else:
        concacaf_adv = 0

    return host_adv, concacaf_adv, same_cont


def _fix_et_scores(df, refresh=False):
    """Replace ET-inflated scores with 90-minute scores using goalscorers data.

    Knockout matches decided by an ET goal are recorded with the post-ET score,
    but the competition rules (and our backtest labels) use 90-minute results only.
    Matches that went to penalties are already correct (still level after ET).
    """
    if refresh or not os.path.exists(GOALS_DATA):
        goals = pd.read_csv(GOALS_URL)
        goals.to_csv(GOALS_DATA, index=False)
    else:
        goals = pd.read_csv(GOALS_DATA)

    goals["date"]     = pd.to_datetime(goals["date"])
    goals["own_goal"] = goals["own_goal"].astype(str).str.upper().eq("TRUE")
    goals             = goals.dropna(subset=["minute"])

    et_keys = set(
        zip(goals.loc[goals["minute"] > 90, "date"].astype(str),
            goals.loc[goals["minute"] > 90, "home_team"],
            goals.loc[goals["minute"] > 90, "away_team"])
    )
    if not et_keys:
        return df

    g90 = goals[goals["minute"] <= 90].copy()
    g90["home_goal"] = (
        (g90["team"] == g90["home_team"]) & ~g90["own_goal"]
    ) | (
        (g90["team"] == g90["away_team"]) & g90["own_goal"]
    )
    g90["away_goal"] = (
        (g90["team"] == g90["away_team"]) & ~g90["own_goal"]
    ) | (
        (g90["team"] == g90["home_team"]) & g90["own_goal"]
    )

    scores90 = (
        g90.groupby(["date", "home_team", "away_team"])
           .agg(home_score_90=("home_goal", "sum"),
                away_score_90=("away_goal", "sum"))
           .reset_index()
    )

    df = df.copy()
    df["_key"] = list(zip(df["date"].astype(str), df["home_team"], df["away_team"]))
    mask = df["_key"].apply(lambda k: k in et_keys)

    fixed = df[mask].merge(scores90, on=["date", "home_team", "away_team"], how="left")
    df.loc[mask, "home_score"] = fixed["home_score_90"].fillna(0).values
    df.loc[mask, "away_score"] = fixed["away_score_90"].fillna(0).values

    print(f"  [ET fix] corrected {mask.sum()} match(es) to 90-minute scores")
    return df.drop(columns=["_key"])


def load_data(refresh=False):
    if refresh or not os.path.exists(DATA):
        df = pd.read_csv(RAW_URL)
        df.to_csv(DATA, index=False)
    else:
        df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = _fix_et_scores(df, refresh=refresh)
    df["outcome"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["home_win", "away_win"], default="draw")
    df.loc[df["home_score"].isna(), "outcome"] = np.nan
    df["importance"] = df["tournament"].apply(importance)
    return df


def build_features(df):
    """One chronological pass: every feature uses only matches before kickoff."""
    elo = defaultdict(lambda: 1500.0)
    res = defaultdict(list)
    last_date, h2h = {}, defaultdict(list)

    def team_feats(team):
        r = res[team]
        if not r:
            return elo[team], 1.3, 1.3, 0.33, 1.0, 1.0, 0.0, 0.0, 0
        last5, last10 = r[-5:], r[-10:]
        streak = 0
        for p, *_ in reversed(r):
            if p < 1:
                break
            streak += 1
        return (elo[team],
                np.mean([p for p, *_ in last5]), np.mean([p for p, *_ in last10]),
                np.mean([w for *_, w in last10]),
                np.mean([g for _, g, _, _ in last5]), np.mean([a for _, _, a, _ in last5]),
                np.mean([g - a for _, g, a, _ in last10]), streak, len(r))

    def h2h_feats(home, away):
        m = h2h[tuple(sorted((home, away)))]
        if not m:
            return 0, 0.5, 0.25, 0.0
        n = len(m)
        return (n,
                sum(w == home for _, _, w in m) / n,
                sum(w == "draw" for _, _, w in m) / n,
                np.mean([g if h == home else -g for h, g, _ in m]))

    rows = []
    for r in df.itertuples():
        h, a, adj = r.home_team, r.away_team, HOME_ADV * (1 - r.neutral)
        he, hf5, hf10, hwr, hgf, hga, hgd, hstk, hn = team_feats(h)
        ae, af5, af10, awr, agf, aga, agd, astk, an = team_feats(a)
        nm, h2h_wr, h2h_dr, h2h_gd = h2h_feats(h, a)
        elo_d = he + adj - ae
        host_adv, concacaf_adv, same_cont_adv = _wc_context(h, a, r.tournament, r.date.year)
        rows.append({
            "elo_diff": elo_d, "home_elo": he, "away_elo": ae,
            "abs_elo_diff": abs(elo_d),
            "form5_diff": hf5 - af5, "form10_diff": hf10 - af10,
            "home_form5": hf5, "away_form5": af5,
            "home_winrate": hwr, "away_winrate": awr,
            "home_gf5": hgf, "away_gf5": agf, "home_ga5": hga, "away_ga5": aga,
            "gd10_diff": hgd - agd, "home_streak": hstk, "away_streak": astk,
            "home_rest": min((r.date - last_date[h]).days, 90) if h in last_date else 30,
            "away_rest": min((r.date - last_date[a]).days, 90) if a in last_date else 30,
            "home_played": hn, "away_played": an,
            "h2h_n": nm, "h2h_home_winrate": h2h_wr, "h2h_draw_rate": h2h_dr, "h2h_gd": h2h_gd,
            "host_adv_diff": host_adv,
            "concacaf_adv_diff": concacaf_adv,
            "same_continent_adv_diff": same_cont_adv,
        })

        if not np.isnan(r.home_score):
            gd = r.home_score - r.away_score
            exp = 1 / (1 + 10 ** ((ae - he - adj) / 400))
            s = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)
            # goal-difference multiplier (FIFA-style)
            g = 1.0 if abs(gd) <= 1 else (1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8)
            delta = r.importance * g * (s - exp)
            elo[h] += delta
            elo[a] -= delta
            res[h].append((3 if gd > 0 else (1 if gd == 0 else 0), r.home_score, r.away_score, gd > 0))
            res[a].append((3 if gd < 0 else (1 if gd == 0 else 0), r.away_score, r.home_score, gd < 0))
            last_date[h] = last_date[a] = r.date
            h2h[tuple(sorted((h, a)))].append((h, gd, h if gd > 0 else (a if gd < 0 else "draw")))

    return df.join(pd.DataFrame(rows, index=df.index))


# ── WC match selectors ────────────────────────────────────────────────────────

def wc_matches(feats, year):
    """All played WC main-tournament matches for a given year."""
    t = feats["tournament"].str.lower()
    return feats[
        t.str.contains("world cup") &
        ~t.str.contains("qualif") &
        (feats["date"].dt.year == year) &
        feats["outcome"].notna()
    ].sort_values("date")


def wc_group_stage(all_wc):
    """Group stage only: each team's first 3 WC games."""
    team_count = defaultdict(int)
    include = []
    for idx, row in all_wc.iterrows():
        h, a = row["home_team"], row["away_team"]
        if team_count[h] < 3 and team_count[a] < 3:
            include.append(idx)
        team_count[h] += 1
        team_count[a] += 1
    return all_wc.loc[include]


def wc_knockout(all_wc):
    """Knockout stage only: matches after each team's first 3 WC games."""
    group_idx = set(wc_group_stage(all_wc).index)
    return all_wc.loc[~all_wc.index.isin(group_idx)]


def wc_group_rounds(all_wc, max_round):
    """First max_round rounds of the group stage.
    Round = 1 + max(games played by either team so far in this tournament)."""
    team_games = defaultdict(int)
    include = []
    for idx, row in all_wc.iterrows():
        h, a = row["home_team"], row["away_team"]
        if max(team_games[h], team_games[a]) + 1 <= max_round:
            include.append(idx)
        team_games[h] += 1
        team_games[a] += 1
    return all_wc.loc[include]
