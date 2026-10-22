"""EFL Championship match predictor.

Pipeline: load -> Elo -> rolling form -> models -> time-based evaluation
vs bookmaker odds -> predict a single fixture.

Data: download E1 CSVs from https://www.football-data.co.uk/englandm.php
and save as data/E1_1819.csv, data/E1_1920.csv, ... (one per season).
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = {"H": 0, "D": 1, "A": 2}


# ---------- 1. Data ----------
def load_data(pattern="data/E1_*.csv"):
    frames = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f, encoding="latin-1")
        d["season"] = f.split("_")[-1].replace(".csv", "")
        frames.append(d)
    if not frames:
        raise FileNotFoundError("No CSVs found. See the docstring for download steps.")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    cols = ["season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "B365H", "B365D", "B365A"]
    return df[cols].sort_values("Date").reset_index(drop=True)


# ---------- 2. Elo (pre-match ratings only, so no leakage) ----------
def add_elo(df, k=20, home_adv=65, regress=0.25, start=1450):
    ratings, last_season = {}, None
    h_elo, a_elo = [], []
    for r in df.itertuples():
        if last_season is not None and r.season != last_season:
            # pull ratings toward the mean between seasons
            ratings = {t: (1 - regress) * v + regress * 1500 for t, v in ratings.items()}
        last_season = r.season
        rh, ra = ratings.get(r.HomeTeam, start), ratings.get(r.AwayTeam, start)
        h_elo.append(rh)
        a_elo.append(ra)
        expected = 1 / (1 + 10 ** (-(rh + home_adv - ra) / 400))
        score = {"H": 1, "D": 0.5, "A": 0}[r.FTR]
        mult = np.log(abs(r.FTHG - r.FTAG) + 1) + 1
        delta = k * mult * (score - expected)
        ratings[r.HomeTeam] = rh + delta
        ratings[r.AwayTeam] = ra - delta
    df = df.copy()
    df["home_elo"], df["away_elo"] = h_elo, a_elo
    return df, ratings


# ---------- 3. Rolling form (shifted by 1 so a match never sees itself) ----------
def add_form(df, window=5):
    home = pd.DataFrame({"mid": df.index, "Date": df.Date, "team": df.HomeTeam,
                         "gf": df.FTHG, "ga": df.FTAG,
                         "pts": df.FTR.map({"H": 3, "D": 1, "A": 0}), "venue": "h"})
    away = pd.DataFrame({"mid": df.index, "Date": df.Date, "team": df.AwayTeam,
                         "gf": df.FTAG, "ga": df.FTHG,
                         "pts": df.FTR.map({"A": 3, "D": 1, "H": 0}), "venue": "a"})
    long = pd.concat([home, away]).sort_values(["team", "Date", "mid"]).reset_index(drop=True)
    g = long.groupby("team")
    for c in ["pts", "gf", "ga"]:
        long[f"{c}_roll"] = g[c].transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
    long["rest"] = g["Date"].diff().dt.days.clip(upper=14)
    feats = ["pts_roll", "gf_roll", "ga_roll", "rest"]
    h = long[long.venue == "h"].set_index("mid")[feats].add_prefix("home_")
    a = long[long.venue == "a"].set_index("mid")[feats].add_prefix("away_")
    return df.join(h).join(a), long


def make_X(df):
    return pd.DataFrame({
        "elo_diff": df.home_elo - df.away_elo,
        "pts_diff": df.home_pts_roll - df.away_pts_roll,
        "gf_diff": df.home_gf_roll - df.away_gf_roll,
        "ga_diff": df.home_ga_roll - df.away_ga_roll,
        "rest_diff": df.home_rest - df.away_rest,
        "home_pts": df.home_pts_roll,
        "away_pts": df.away_pts_roll,
    })


# ---------- 4. Evaluation ----------
def book_probs(df):
    inv = 1 / df[["B365H", "B365D", "B365A"]].values
    return inv / inv.sum(axis=1, keepdims=True)  # remove the overround


def brier(y, p):
    onehot = np.eye(3)[y]
    return np.mean(np.sum((p - onehot) ** 2, axis=1))


def report(name, y, p):
    print(f"{name:<22} logloss={log_loss(y, p, labels=[0, 1, 2]):.4f}  "
          f"brier={brier(y, p):.4f}  acc={accuracy_score(y, p.argmax(1)):.3f}")


def main():
    df = load_data()
    df, ratings = add_elo(df)
    df, long = add_form(df)

    df = df.dropna(subset=["B365H", "B365D", "B365A"])
    X, y = make_X(df), df.FTR.map(CLASSES)
    ok = X.notna().all(axis=1)
    df, X, y = df[ok], X[ok], y[ok].astype(int)

    # time-based split: train on past seasons, test on the latest one
    test_season = df.season.iloc[-1]
    tr, te = (df.season != test_season).values, (df.season == test_season).values
    print(f"Train: {tr.sum()} matches | Test ({test_season}): {te.sum()} matches\n")

    models = {
        "Elo only (logreg)": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                              ["elo_diff"]),
        "Logreg + form": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                          list(X.columns)),
        "Gradient boosting": (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                             max_iter=200, random_state=0),
                              list(X.columns)),
    }
    preds = {}
    for name, (m, cols) in models.items():
        m.fit(X[tr][cols], y[tr])
        preds[name] = m.predict_proba(X[te][cols])
        report(name, y[te], preds[name])
    preds["Bookmaker (B365)"] = book_probs(df[te])
    report("Bookmaker (B365)", y[te], preds["Bookmaker (B365)"])

    # calibration plot for P(home win)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    for name, p in preds.items():
        frac, mean = calibration_curve((y[te] == 0).astype(int), p[:, 0], n_bins=6)
        plt.plot(mean, frac, marker="o", label=name)
    plt.xlabel("Predicted P(home win)")
    plt.ylabel("Observed frequency")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("calibration.png", dpi=150)
    print("\nSaved calibration.png")

    return models["Gradient boosting"][0].fit(X[list(X.columns)], y), ratings, long


# ---------- 5. Predict a fixture ----------
def predict_fixture(model, ratings, long, home, away, date, window=5):
    def state(team):
        s = long[long.team == team].sort_values("Date")
        last = s.tail(window)
        return {"pts_roll": last.pts.mean(), "gf_roll": last.gf.mean(),
                "ga_roll": last.ga.mean(),
                "rest": min((pd.Timestamp(date) - s.Date.iloc[-1]).days, 14)}
    h, a = state(home), state(away)
    row = pd.DataFrame([{
        "home_elo": ratings[home], "away_elo": ratings[away],
        **{f"home_{k}": v for k, v in h.items()},
        **{f"away_{k}": v for k, v in a.items()},
    }])
    p = model.predict_proba(make_X(row))[0]
    return dict(zip(["Home", "Draw", "Away"], p.round(3)))


if __name__ == "__main__":
    model, ratings, long = main()
    # Example (use team names exactly as they appear in the CSV):
    # print(predict_fixture(model, ratings, long, "Leeds", "Sunderland", "2026-10-24"))
