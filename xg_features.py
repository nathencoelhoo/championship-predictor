"""Add rolling xG features and test whether they improve walk-forward results.

Expects data/xg.csv with columns: date, home, away, home_xg, away_xg
(one row per match; date as YYYY-MM-DD). Export from FBref or Understat.

Run: python xg_features.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from predictor import CLASSES, add_elo, add_form, book_probs, brier, load_data, make_X
from walkforward import bootstrap_diff, per_row_logloss, walk_forward

# FBref/Understat name -> football-data.co.uk name. Extend as needed:
# the script prints any names in your xG file that don't match.
XG_MAP = {
    "Sheffield Wed": "Sheffield Weds",
    "Sheffield Utd": "Sheffield United",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "West Brom": "West Brom",
    "Nott'ham Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Preston North End": "Preston",
    "Blackburn Rovers": "Blackburn",
    "Birmingham City": "Birmingham",
    "Leeds United": "Leeds",
    "Hull City": "Hull",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Cardiff City": "Cardiff",
    "Coventry City": "Coventry",
    "Norwich City": "Norwich",
    "Luton Town": "Luton",
    "Derby County": "Derby",
    "Huddersfield Town": "Huddersfield",
    "Plymouth Argyle": "Plymouth",
    "Oxford United": "Oxford",
    "Bristol City": "Bristol City",
    "Wigan Athletic": "Wigan",
    "Rotherham United": "Rotherham",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton and Hove Albion": "Brighton",
    "Manchester City": "Man City",
    "Manchester Utd": "Man United",
}


def load_xg(path="data/xg.csv"):
    x = pd.read_csv(path, parse_dates=["date"])
    for c in ["home", "away"]:
        x[c] = x[c].replace(XG_MAP)
    return x.drop_duplicates(subset=["date", "home", "away"])


def merge_xg(df, xg):
    x = xg.rename(columns={"date": "Date", "home": "HomeTeam", "away": "AwayTeam"})
    out = df.merge(x[["Date", "HomeTeam", "AwayTeam", "home_xg", "away_xg"]],
                   on=["Date", "HomeTeam", "AwayTeam"], how="left")
    out.index = df.index  # merge resets the index; keep the original

    known = set(df.HomeTeam) | set(df.AwayTeam)
    unmatched = sorted((set(xg.home) | set(xg.away)) - known)
    if unmatched:
        print("Unmatched team names (add to XG_MAP):", unmatched)
    covered = df.Date.between(xg.date.min(), xg.date.max())
    print(f"xG match rate within its date range: {out.loc[covered, 'home_xg'].notna().mean():.1%}\n")
    return out


def add_xg_form(df, window=5):
    home = pd.DataFrame({"mid": df.index, "Date": df.Date, "team": df.HomeTeam,
                         "xgf": df.home_xg, "xga": df.away_xg, "venue": "h"})
    away = pd.DataFrame({"mid": df.index, "Date": df.Date, "team": df.AwayTeam,
                         "xgf": df.away_xg, "xga": df.home_xg, "venue": "a"})
    long = pd.concat([home, away]).sort_values(["team", "Date", "mid"]).reset_index(drop=True)
    g = long.groupby("team")
    for c in ["xgf", "xga"]:
        # shift(1): a match never sees its own xG
        long[f"{c}_roll"] = g[c].transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
    feats = ["xgf_roll", "xga_roll"]
    h = long[long.venue == "h"].set_index("mid")[feats].add_prefix("home_")
    a = long[long.venue == "a"].set_index("mid")[feats].add_prefix("away_")
    return df.join(h).join(a)


def make_xg_X(df):
    X = make_X(df)
    X["xgd_diff"] = ((df.home_xgf_roll - df.home_xga_roll)
                     - (df.away_xgf_roll - df.away_xga_roll))
    X["home_xgf"], X["home_xga"] = df.home_xgf_roll, df.home_xga_roll
    X["away_xgf"], X["away_xga"] = df.away_xgf_roll, df.away_xga_roll
    return X


def main():
    df = load_data()
    df, _ = add_elo(df)
    df, _ = add_form(df)
    df = merge_xg(df, load_xg())
    df = add_xg_form(df)

    df = df.dropna(subset=["B365H", "B365D", "B365A"])
    base_cols = list(make_X(df).columns)
    X = make_xg_X(df)
    y = df.FTR.map(CLASSES)
    ok = X[base_cols].notna().all(axis=1)
    df, X, y = df[ok], X[ok], y[ok].astype(int)

    # Gradient boosting handles NaN, so seasons without xG still help training.
    make = lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)
    p_base = walk_forward(df, X, y, make, base_cols)
    p_xg = walk_forward(df, X, y, make, list(X.columns))
    p_book = book_probs(df)

    # Compare only on matches that have xG features, so the test is like-for-like.
    mask = ~np.isnan(p_base).any(axis=1) & X["xgd_diff"].notna().values
    yt = y[mask].values
    print(f"Evaluated on {mask.sum()} matches with xG available\n")
    for name, p in [("Base features", p_base), ("Base + xG", p_xg), ("Bookmaker", p_book)]:
        pm = p[mask]
        print(f"  {name:<14} logloss={log_loss(yt, pm, labels=[0, 1, 2]):.4f}  brier={brier(yt, pm):.4f}")

    lo, hi = bootstrap_diff(per_row_logloss(yt, p_xg[mask]), per_row_logloss(yt, p_base[mask]))
    print(f"\nxG minus base log loss, 95% CI: [{lo:+.4f}, {hi:+.4f}]  (negative = xG helps)")
    lo, hi = bootstrap_diff(per_row_logloss(yt, p_xg[mask]), per_row_logloss(yt, p_book[mask]))
    print(f"xG model minus bookmaker, 95% CI: [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
