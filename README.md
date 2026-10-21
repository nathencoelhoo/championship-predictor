# EFL Championship Match Predictor

Predicts Home / Draw / Away probabilities for Championship matches and
benchmarks them against bookmaker odds.

## Setup
```
pip install pandas numpy scikit-learn matplotlib
```
Download season CSVs from https://www.football-data.co.uk/englandm.php
(the "Championship" column) and save as `data/E1_1819.csv`, `data/E1_1920.csv`, etc.
Aim for 6+ seasons.

Run: `python predictor.py`

## Method
- **Elo ratings** with home advantage, goal-difference multiplier and
  between-season regression to the mean.
- **Rolling form** (points, goals for/against over the last 5 matches) and rest days,
  all shifted so a match never sees its own result.
- **Models**: Elo-only logistic regression, logistic regression + form,
  gradient boosting.
- **Evaluation**: train on past seasons, test on the latest (time-based split).
  Log loss, multiclass Brier score, accuracy, calibration plot.
  Bookmaker odds (overround removed) are the benchmark, not a feature.

## Results
_Fill in your table after running, and be honest about the gap to the bookmakers._

| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Elo only | | | |
| Logreg + form | | | |
| Gradient boosting | | | |
| Bookmaker | | | |

## Upgrades to make it stand out
1. **Walk-forward validation**: retrain every month and evaluate on the next block.
2. **xG features** from FBref/Understat (rolling xG for/against).
3. **Dixon-Coles / Poisson model** to predict scorelines, then derive H/D/A.
4. **Promoted/relegated team priors**: start new teams at a rating based on their previous league, not a flat 1450.
5. **Calibration**: wrap the boosting model in `CalibratedClassifierCV`.
6. **Betting simulation**: flat-stake value bets where model prob > implied prob, with a margin threshold. Report ROI and its uncertainty, and expect it to be near or below zero.
7. **App**: Streamlit page with a fixture selector, probability bars and an Elo history chart. Deploy on Streamlit Community Cloud.
8. **Tests and structure**: split into `data.py`, `features.py`, `models.py`, add pytest tests for leakage (e.g. a team's first match has NaN form).
