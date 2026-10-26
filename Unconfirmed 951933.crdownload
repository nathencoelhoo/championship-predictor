"""Dixon-Coles scoreline model with time decay.

Each team has an attack and a defence strength. Expected goals:
    home = exp(attack_home + defence_away + home_advantage)
    away = exp(attack_away + defence_home)
Goals are Poisson, with the Dixon-Coles correction (rho) for the
low-scoring results 0-0, 1-0, 0-1 and 1-1. Recent matches count more
(exponential decay with a chosen half-life). Home/Draw/Away probabilities
come from summing the predicted scoreline grid.

Note: defence > 0 means a team concedes MORE (it is added to the opponent's
scoring rate).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


def _nll(params, hi, ai, x, y, w, n, seen, reg, masks):
    att, dfn = params[:n], params[n:2 * n]
    home, rho = params[2 * n], params[2 * n + 1]
    att = att - att[seen].mean()  # identifiability: mean attack of active teams = 0
    eta_h = att[hi] + dfn[ai] + home
    eta_a = att[ai] + dfn[hi]
    lam, mu = np.exp(eta_h), np.exp(eta_a)
    m00, m01, m10, m11 = masks
    tau = np.ones_like(lam)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    tau = np.clip(tau, 1e-10, None)
    ll = np.log(tau) + x * eta_h - lam + y * eta_a - mu  # constants dropped
    return -(w * ll).sum() + 0.5 * reg * ((att ** 2).sum() + (dfn ** 2).sum())


def fit_dc(hi, ai, x, y, w, n, reg=0.5, init=None):
    """Fit on arrays of home idx, away idx, home goals, away goals, weights."""
    seen = np.zeros(n, dtype=bool)
    seen[hi] = True
    seen[ai] = True
    masks = ((x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1))
    if init is None:
        init = np.zeros(2 * n + 2)
        init[2 * n] = 0.25
        init[2 * n + 1] = -0.05
    bounds = [(None, None)] * (2 * n + 1) + [(-0.3, 0.3)]
    res = minimize(_nll, init, args=(hi, ai, x, y, w, n, seen, reg, masks),
                   method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
    return res.x, seen


def predict_scoreline(params, n, seen, h, a, max_goals=8):
    """Return (P grid [home goals x away goals], (pH, pD, pA)).
    Teams unseen in the training window get the average strength."""
    att, dfn = params[:n].copy(), params[n:2 * n].copy()
    home, rho = params[2 * n], params[2 * n + 1]
    att = att - att[seen].mean()
    att[~seen] = 0.0                      # average attack
    dfn[~seen] = dfn[seen].mean()         # average defence
    lam = np.exp(att[h] + dfn[a] + home)
    mu = np.exp(att[a] + dfn[h])
    i = np.arange(max_goals + 1)
    ph = np.exp(i * np.log(lam) - lam - gammaln(i + 1))
    pa = np.exp(i * np.log(mu) - mu - gammaln(i + 1))
    P = np.outer(ph, pa)
    P[0, 0] *= 1 - lam * mu * rho
    P[0, 1] *= 1 + lam * rho
    P[1, 0] *= 1 + mu * rho
    P[1, 1] *= 1 - rho
    P = np.clip(P, 0, None)
    P /= P.sum()
    return P, (np.tril(P, -1).sum(), np.trace(P), np.triu(P, 1).sum())


def walk_forward_dc(df, half_life=365, min_train_seasons=3, window_days=1460,
                    reg=0.5, min_train_rows=300):
    """Monthly walk-forward: fit on past matches only, predict the next month.
    Returns an (n, 3) array of H/D/A probabilities (NaN where not predicted)."""
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = df.HomeTeam.map(idx).values
    ai = df.AwayTeam.map(idx).values
    x, y = df.FTHG.values.astype(float), df.FTAG.values.astype(float)

    seasons = sorted(df.season.unique(), key=lambda s: df.loc[df.season == s, "Date"].min())
    start = df.loc[df.season == seasons[min_train_seasons], "Date"].min()
    period = df.Date.dt.to_period("M")
    out = np.full((len(df), 3), np.nan)
    decay = np.log(2) / half_life
    init = None

    for p in sorted(period[df.Date >= start].unique()):
        te = (period == p).values
        cutoff = p.start_time
        tr = ((df.Date < cutoff) & (df.Date >= cutoff - pd.Timedelta(days=window_days))).values
        if te.sum() == 0 or tr.sum() < min_train_rows:
            continue
        age = (cutoff - df.Date[tr]).dt.days.values
        w = np.exp(-decay * age)
        init, seen = fit_dc(hi[tr], ai[tr], x[tr], y[tr], w, n, reg=reg, init=init)
        for r in np.where(te)[0]:
            _, probs = predict_scoreline(init, n, seen, hi[r], ai[r])
            out[r] = probs
    return out
