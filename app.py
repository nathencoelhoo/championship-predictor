"""Streamlit app: pick a fixture, see probabilities, Elo history and form.

Run: streamlit run app.py
"""
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predictor import CLASSES, add_elo, add_form, load_data, make_X, predict_fixture
from elo_v2 import add_elo_v2, load_entrants

st.set_page_config(page_title="Championship Predictor", page_icon="⚽", layout="centered")


@st.cache_resource(show_spinner="Building ratings and training model...")
def build():
    df = load_data()
    df, ratings = add_elo(df)
    df, long = add_form(df)
    X, y = make_X(df), df.FTR.map(CLASSES)
    ok = X.notna().all(axis=1)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[ok], y[ok].astype(int))
    return df, ratings, long, model


df, ratings, long, model = build()

latest = df[df.season == df.season.iloc[-1]]
teams = sorted(set(latest.HomeTeam) | set(latest.AwayTeam))

st.title("⚽ EFL Championship Predictor")
st.caption(f"Elo + rolling form model. Data up to {df.Date.max():%d %b %Y}.")
st.caption(f"{len(df)} matches loaded | seasons: {', '.join(sorted(df.season.unique()))}")

c1, c2 = st.columns(2)
home = c1.selectbox("Home", teams, index=0)
away = c2.selectbox("Away", [t for t in teams if t != home], index=0)
date = st.date_input("Match date", value=pd.Timestamp.today().date())

probs = predict_fixture(model, ratings, long, home, away, date)

st.subheader(f"{home} vs {away}")
for label, name in [("Home win", "Home"), ("Draw", "Draw"), ("Away win", "Away")]:
    p = float(probs[name])
    st.write(f"**{label}**: {p:.1%}")
    st.progress(p)

# ---- Elo history ----
st.subheader("Elo history")
n_seasons = st.slider("Seasons to show", 1, min(6, df.season.nunique()), 3)
cutoff = df.groupby("season").Date.min().sort_values().iloc[-n_seasons]


def elo_series(team):
    h = df[df.HomeTeam == team].set_index("Date").home_elo
    a = df[df.AwayTeam == team].set_index("Date").away_elo
    s = pd.concat([h, a]).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s[s.index >= cutoff]
    s.name = team
    return s


elo = pd.concat([elo_series(home), elo_series(away)], axis=1).sort_index().ffill()
import altair as alt
plot_df = elo.rename_axis("Date").reset_index().melt("Date", var_name="Team", value_name="Elo")
st.altair_chart(
    alt.Chart(plot_df).mark_line().encode(
        x="Date:T",
        y=alt.Y("Elo:Q", scale=alt.Scale(zero=False)),
        color="Team:N",
    ),
    use_container_width=True,
)
st.caption(f"Current Elo: {home} {ratings[home]:.0f} | {away} {ratings[away]:.0f}")


# ---- Recent form ----
def form_string(team):
    last = long[long.team == team].sort_values("Date").tail(5)
    symbol = {3: "W", 1: "D", 0: "L"}
    return " ".join(f"{symbol[int(r.pts)]} {int(r.gf)}-{int(r.ga)}" for r in last.itertuples())


st.subheader("Last 5 results (oldest to newest)")
st.write(f"**{home}**: {form_string(home)}")
st.write(f"**{away}**: {form_string(away)}")

# ---- Model vs bookmaker ----
with st.expander("Compare with bookmaker odds"):
    o1, o2, o3 = st.columns(3)
    oh = o1.number_input("Home odds", 1.01, 100.0, 2.50, 0.05)
    od = o2.number_input("Draw odds", 1.01, 100.0, 3.30, 0.05)
    oa = o3.number_input("Away odds", 1.01, 100.0, 2.90, 0.05)
    inv = pd.Series({"Home": 1 / oh, "Draw": 1 / od, "Away": 1 / oa})
    book = inv / inv.sum()
    table = pd.DataFrame({
        "Model": pd.Series(probs).astype(float),
        "Bookmaker (no margin)": book,
    })
    table["Difference"] = table["Model"] - table["Bookmaker (no margin)"]
    st.dataframe(table.style.format("{:.1%}"))
    st.caption("For information only. Bookmaker odds are usually better calibrated than "
               "this model, so differences are more likely model error than value.")

st.divider()
st.caption("Portfolio project. Not betting advice.")
