"""Walk-forward validation: retrain monthly on past data only, predict the next month.

Run: python walkforward.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predictor import CLASSES, add_elo, add_form, book_probs, brier, load_data, make_X


def walk_forward(df, X, y, make_model, cols, min_train_seasons=3, min_train_rows=300):
    """Returns an (n, 3) array of out-of-sample probabilities (NaN where not predicted)."""
    seasons = sorted(df.season.unique(), key=lambda s: df.loc[df.season == s, "Date"].min())
    start = df.loc[df.season == seasons[min_train_seasons], "Date"].min()
    period = df.Date.dt.to_period("M")
    out = np.full((len(df), 3), np.nan)
    for p in sorted(period[df.Date >= start].unique()):
        te = (period == p).values
        tr = (df.Date < p.start_time).values  # strictly before the month starts
        if te.sum() == 0 or tr.sum() < min_train_rows:
            continue
        m = make_model()
        m.fit(X[tr][cols], y[tr])
        out[te] = m.predict_proba(X[te][cols])
    return out


def per_row_logloss(y, p):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def bootstrap_diff(loss_a, loss_b, n=2000, seed=0):
    """95% CI for mean(loss_a - loss_b). Negative = model a is better."""
    rng = np.random.default_rng(seed)
    d = loss_a - loss_b
    means = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def main():
    df = load_data()
    df, _ = add_elo(df)
    df, _ = add_form(df)
    df = df.dropna(subset=["B365H", "B365D", "B365A"])
    X, y = make_X(df), df.FTR.map(CLASSES)
    ok = X.notna().all(axis=1)
    df, X, y = df[ok], X[ok], y[ok].astype(int)

    all_cols = list(X.columns)
    models = {
        "Elo only": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                     ["elo_diff"]),
        "Logreg + form": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                          all_cols),
        "Grad boosting": (lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=0), all_cols),
    }

    preds = {n: walk_forward(df, X, y, mk, cols) for n, (mk, cols) in models.items()}
    preds["Bookmaker"] = book_probs(df)

    mask = ~np.isnan(preds["Grad boosting"]).any(axis=1)
    yt, seasons = y[mask].values, df.season[mask].values
    print(f"Walk-forward predictions: {mask.sum()} matches\n")

    print("Overall")
    for n, p in preds.items():
        pm = p[mask]
        print(f"  {n:<16} logloss={log_loss(yt, pm, labels=[0, 1, 2]):.4f}  brier={brier(yt, pm):.4f}")

    print("\nLog loss by season")
    rows = {}
    for n, p in preds.items():
        rows[n] = {s: log_loss(yt[seasons == s], p[mask][seasons == s], labels=[0, 1, 2])
                   for s in np.unique(seasons)}
    print(pd.DataFrame(rows).round(4).to_string())

    print("\nModel minus bookmaker log loss (95% bootstrap CI; negative = model better)")
    book_loss = per_row_logloss(yt, preds["Bookmaker"][mask])
    for n in models:
        lo, hi = bootstrap_diff(per_row_logloss(yt, preds[n][mask]), book_loss)
        print(f"  {n:<16} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
