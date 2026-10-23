"""Elo v2: smarter ratings for teams entering the Championship.

Two ideas, both usable on their own:
1. Entry priors: a team relegated from the Premier League (parachute money,
   better squad) starts higher than one promoted from League One. Starting
   points are set relative to the average rating of teams that stayed in the
   league, so they don't drift with the rating scale.
2. New-team K boost: while a team has played fewer than `boost_games` matches
   in its current spell, rating updates are larger so it finds its level faster.

Entry info comes from data/entrants.csv (start_year, team, from_league).
Teams not listed are treated as "unknown".
"""
import numpy as np
import pandas as pd


def load_entrants(path="data/entrants.csv"):
    try:
        e = pd.read_csv(path)
    except FileNotFoundError:
        print("No data/entrants.csv found: every entrant is treated as 'unknown'.")
        return {}
    return {(r.team, int(r.start_year)): r.from_league for r in e.itertuples()}


def add_elo_v2(df, entrants=None, k=20, home_adv=65, regress=0.25,
               offsets=None, boost=1.0, boost_games=10, base=1500):
    """Returns (df with home_elo/away_elo/home_new/away_new, final ratings).

    offsets: starting rating relative to the mean of continuing teams, by
    origin: {"PL": ..., "L1": ..., "UNK": ...}.
    home_new / away_new: True if that team has played < boost_games matches
    in its current spell in the league (before this match).
    """
    entrants = entrants or {}
    offsets = offsets or {"PL": 0, "L1": -60, "UNK": -50}
    df = df.copy()
    start_year = df.groupby("season").Date.transform("min").dt.year
    teams_in = {s: set(g.HomeTeam) | set(g.AwayTeam) for s, g in df.groupby("season")}

    ratings, games = {}, {}
    prev_teams, cur = set(), None
    h_elo, a_elo, h_new, a_new = [], [], [], []

    for r, sy in zip(df.itertuples(), start_year):
        if r.season != cur:
            # new season: regress everyone, then (re)initialise entrants
            for t in ratings:
                ratings[t] = (1 - regress) * ratings[t] + regress * base
            continuing = [ratings[t] for t in teams_in[r.season] if t in prev_teams]
            league_mean = float(np.mean(continuing)) if continuing else base
            for t in teams_in[r.season]:
                if t not in prev_teams:
                    origin = entrants.get((t, int(sy)), "UNK")
                    ratings[t] = league_mean + offsets.get(origin, offsets["UNK"])
                    games[t] = 0
            prev_teams, cur = teams_in[r.season], r.season

        h, a = r.HomeTeam, r.AwayTeam
        rh, ra = ratings[h], ratings[a]
        gh, ga = games[h], games[a]
        h_elo.append(rh)
        a_elo.append(ra)
        h_new.append(gh < boost_games)
        a_new.append(ga < boost_games)

        expected = 1 / (1 + 10 ** (-(rh + home_adv - ra) / 400))
        score = {"H": 1, "D": 0.5, "A": 0}[r.FTR]
        mult = np.log(abs(r.FTHG - r.FTAG) + 1) + 1
        k_eff = k * (boost if min(gh, ga) < boost_games else 1.0)
        delta = k_eff * mult * (score - expected)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
        games[h] += 1
        games[a] += 1

    df["home_elo"], df["away_elo"] = h_elo, a_elo
    df["home_new"], df["away_new"] = h_new, a_new

    known = set(df.HomeTeam) | set(df.AwayTeam)
    missing = sorted({t for (t, _) in entrants} - known)
    if missing:
        print("Names in entrants.csv not found in the data (check spelling):", missing)
    return df, ratings
