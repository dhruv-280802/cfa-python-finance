"""D08 — Hyperparameter optimisation (CFA L2 PSM Unit 3: "Financial
Forecasting Using Machine and Deep Learning").

CFA idea: every model has knobs — Ridge's alpha (how hard to shrink
coefficients), a forest's depth (how deep the trees grow). Guessing
the knobs is not a method; GRID SEARCH tries every combination on a
small grid and keeps the one with the best CROSS-VALIDATED score.
Cross-validation for time series must respect time: TimeSeriesSplit
trains on the past and validates on the next chunk, rolling forward
— never a random shuffle, which would leak the future into training.

How to read the result:
- Best params: the winning combination. Alpha large = heavy shrinkage
  (simpler model); max_depth small = shallow trees (less overfitting).
- CV score: the average validation error across the folds. This is
  the honest estimate of the tuned model — the TEST score after it
  is the final check, and it is usually a little worse (tuning itself
  overfits the validation folds a little).
- The trap: if you tune on the test set, or re-tune until the test
  score looks good, the test set stops being a test. Tune on
  validation, report on test, once.

Worked example: GridSearchCV over Ridge alpha and over a small
random-forest grid, 3-fold time-series CV, on the D05 features.
"""
import numpy as np

from d05_sklearn_workflow import FEATURES, build_features, _metrics


def run(prices, source):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    frame = build_features(prices)
    split = int(len(frame) * 0.70)
    train, test = frame.iloc[:split], frame.iloc[split:]
    tscv = TimeSeriesSplit(n_splits=3)
    print("D08  Hyperparameter tuning — GridSearchCV, time-series CV")
    print(f"  Train {len(train)} days, test {len(test)}; CV = 3 "
          f"time-ordered folds (each fold trains on the past, "
          f"validates on the next chunk)")

    ridge_grid = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
        cv=tscv, scoring="neg_root_mean_squared_error")
    ridge_grid.fit(train[FEATURES], train["target"])
    best_alpha = ridge_grid.best_params_["ridge__alpha"]
    print(f"  Ridge: best alpha {best_alpha} (CV RMSE "
          f"{-ridge_grid.best_score_:.4f}) — a larger alpha shrinks "
          f"coefficients harder; the winner balances bias (too much "
          f"shrinkage, signal lost) against variance (too little, "
          f"noise fitted)")
    rmse_r, _, r2_r = _metrics(test["target"],
                               ridge_grid.predict(test[FEATURES]),
                               "Tuned Ridge (test)")

    forest_grid = GridSearchCV(
        RandomForestRegressor(n_estimators=200, min_samples_leaf=20,
                              random_state=42, n_jobs=-1),
        {"max_depth": [3, 5, 8], "max_features": [0.6, 1.0]},
        cv=tscv, scoring="neg_root_mean_squared_error")
    forest_grid.fit(train[FEATURES], train["target"])
    print(f"  Forest: best {forest_grid.best_params_} (CV RMSE "
          f"{-forest_grid.best_score_:.4f}) — shallow trees win here, "
          f"which is the data saying 'do not overfit me'")
    rmse_f, _, r2_f = _metrics(test["target"],
                               forest_grid.predict(test[FEATURES]),
                               "Tuned Forest (test)")
    print(f"  Reading: tuned Ridge test R² {r2_r:+.3f}, tuned forest "
          f"{r2_f:+.3f} — compare with the untuned D05/D07 scores. "
          f"Tuning usually buys a small improvement; if the tuned "
          f"test score is WORSE than the untuned one, the grid "
          f"overfitted the validation folds (it happens, and it is "
          f"why the test set exists)")
    return {"tune_ridge_alpha": float(best_alpha),
            "tune_ridge_r2_test": r2_r, "tune_forest_r2_test": r2_f,
            "tune_forest_params": str(forest_grid.best_params_)}
