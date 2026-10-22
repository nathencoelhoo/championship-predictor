# EFL Championship Match Predictor

**Live demo:** https://championship-predictor.streamlit.app/

Predicts Home / Draw / Away probabilities for Championship matches and
benchmarks them against bookmaker odds.

## Setup
```
pip install pandas numpy scikit-learn matplotlib
```

## Data

Results and bookmaker odds come from [football-data.co.uk](https://www.football-data.co.uk/englandm.php) (Championship, 2021-22 to 2026-27, about 2,900 matches, current season up to 17 Oct 2026). Files live in `data/` as `E1_<label>.csv`; the label in the filename is only a season tag, and the model reads real match dates from inside each file. Duplicate fixtures are dropped on load.

To refresh, replace the current season's CSV in `data/` with the latest download and redeploy.

Run the evaluation and the app:
```
python predictor.py
streamlit run app.py
```

## Method
- **Elo ratings** with home advantage, goal-difference multiplier and
  between-season regression to the mean.
- **Rolling form** (points, goals for/against over the last 5 matches) and rest days,
  all shifted so a match never sees its own result.
- **Models**: Elo-only logistic regression, logistic regression + form,
  gradient boosting.
- **Evaluation**: walk-forward validation. Every month, retrain on past matches only and predict the next month, so all predictions are out-of-sample.
  Log loss and multiclass Brier score, with Bet365 odds (margin removed) as the benchmark, not a feature.
  Bootstrap confidence intervals compare each model with the bookmaker.

## Results
Walk-forward validation (retrained monthly, 1,219 matches from 2024-25 to 2026-27). Lower is better; uniform guessing scores 1.0986 log loss.

| Model | Log loss | Brier |
|---|---|---|
| Elo only | 1.0526 | 0.6350 |
| Logreg + form | 1.0560 | 0.6374 |
| Gradient boosting | 1.1169 | 0.6718 |
| Bookmaker (Bet365, margin removed) | 1.0364 | 0.6243 |

**Takeaways:** Elo captures most of the available signal and clearly beats naive guessing, but every model trails the bookmakers (95% CI for Elo minus bookmaker log loss: +0.007 to +0.025). Rolling form added nothing over Elo, and gradient boosting overfit the small dataset. Early-season error is highest because newly promoted teams start at a flat rating.

## Upgrades to make it stand out
1. **Walk-forward validation**: retrain every month and evaluate on the next block.
2. **xG features** from FBref/Understat (rolling xG for/against).
3. **Dixon-Coles / Poisson model** to predict scorelines, then derive H/D/A.
4. **Promoted/relegated team priors**: start new teams at a rating based on their previous league, not a flat 1450.
5. **Calibration**: wrap the boosting model in `CalibratedClassifierCV`.
6. **Betting simulation**: flat-stake value bets where model prob > implied prob, with a margin threshold. Report ROI and its uncertainty, and expect it to be near or below zero.
7. **App**: Streamlit page with a fixture selector, probability bars and an Elo history chart. Deploy on Streamlit Community Cloud.
8. **Tests and structure**: split into `data.py`, `features.py`, `models.py`, add pytest tests for leakage (e.g. a team's first match has NaN form).
