"""D07 — Random forests and XGBoost (CFA L2 PSM Unit 3: "Financial
Forecasting Using Machine and Deep Learning").

CFA idea: a decision tree asks a sequence of yes/no questions
("was yesterday's SPY return positive?", "is volatility above
2%?") and predicts the average outcome of the days that land in
the same leaf. One tree overfits — it memorises the training data.
A RANDOM FOREST fixes that with two kinds of randomness: each tree
trains on a bootstrap resample of the days (bagging), and each split
only sees a random subset of the features. Averaging hundreds of
decorrelated trees cancels out their individual mistakes. GRADIENT
BOOSTING (XGBoost) attacks the same problem from the other side:
build trees one at a time, each new tree correcting the errors of
the ones before it — powerful, but easier to overfit, so it needs
regularisation (shallow trees, small learning rate, early stopping).

How to read the result:
- Feature importances: how much each feature reduced the error
  across all trees. They sum to 1. A feature at 0.40 dominates the
  model; one at 0.02 is decoration. Importances say WHAT the model
  leans on, not WHY — a high importance can still be a spurious
  correlation.
- Out-of-sample R²/RMSE on the same split as D05/D06: the three
  models are directly comparable. If the trees beat Ridge and the
  MLP, the signal is non-linear and interaction-heavy; if they
  lose, the signal is (at best) linear and weak.

Worked example: RandomForestRegressor and XGBRegressor predicting
tomorrow's AAPL return on the D05 features. (If xgboost is not
installed, the module falls back to sklearn's HistGradientBoosting
— same boosting idea, different library — and says so.)
"""
import numpy as np

from d05_sklearn_workflow import FEATURES, build_features, _metrics


def run(prices, source):
    from sklearn.ensemble import RandomForestRegressor

    frame = build_features(prices)
    split = int(len(frame) * 0.70)
    train, test = frame.iloc[:split], frame.iloc[split:]
    print("D07  Random forest + XGBoost — same task, tree models")
    print(f"  {len(frame)} days; train {len(train)}, test {len(test)} "
          f"(time-ordered, as in D05)")

    rf = RandomForestRegressor(n_estimators=300, max_depth=5,
                               min_samples_leaf=20, random_state=42,
                               n_jobs=-1)
    rf.fit(train[FEATURES], train["target"])
    rmse_rf, _, r2_rf = _metrics(test["target"],
                                 rf.predict(test[FEATURES]), "Forest (test)")
    imp = dict(sorted(zip(FEATURES, np.round(rf.feature_importances_, 3)),
                      key=lambda kv: -kv[1]))
    print(f"  Forest feature importances (sum to 1): {imp} — the top "
          f"feature is what the trees split on most; a flat spread "
          f"means no feature dominates")

    try:
        from xgboost import XGBRegressor

        booster = XGBRegressor(n_estimators=300, max_depth=3,
                               learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, random_state=42,
                               n_jobs=-1)
        booster_name = "XGBoost"
    except Exception as exc:  # noqa: BLE001 - library optional by design
        from sklearn.ensemble import HistGradientBoostingRegressor

        booster = HistGradientBoostingRegressor(
            max_iter=300, max_depth=3, learning_rate=0.05,
            random_state=42)
        booster_name = f"HistGradientBoosting (xgboost unavailable: {exc})"
    booster.fit(train[FEATURES], train["target"])
    rmse_xgb, _, r2_xgb = _metrics(
        test["target"], booster.predict(test[FEATURES]),
        f"{booster_name[:12]:>12} (test)")
    print(f"  Reading: forest R² {r2_rf:+.3f}, boosting R² {r2_xgb:+.3f} "
          f"(D05 Ridge and D06 MLP are the comparison) — "
          f"{'the trees found structure the linear model missed' if max(r2_rf, r2_xgb) > 0.01 else 'even flexible tree ensembles cannot squeeze a tradable signal out of daily returns — the honest result'}")
    print("  Warning: boosting with a high learning rate or deep trees "
          "will fit the training data almost perfectly and fail "
          "out of sample. Shallow trees + small learning rate is the "
          "discipline, not a suggestion")
    return {"tree_rf_rmse_test": rmse_rf, "tree_rf_r2_test": r2_rf,
            "tree_xgb_rmse_test": rmse_xgb, "tree_xgb_r2_test": r2_xgb,
            "tree_top_feature": max(imp, key=imp.get)}
