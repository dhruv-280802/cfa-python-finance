"""D05 — The scikit-learn workflow (CFA L2 PSM Unit 3: "Financial
Forecasting Using Machine and Deep Learning").

CFA idea: a machine-learning model is a WORKFLOW, not a function
call — build features, split train/test, fit, then judge on data the
model has never seen. Two rules that matter more than the choice of
model:
1. NEVER shuffle time-series data into train/test. The future must
   not leak into training — train on the past, test on the future,
   or your "accuracy" is a lie.
2. ALWAYS compare against a naive baseline. If your model cannot
   beat "predict zero" (or "predict the mean"), it has learned
   nothing — no matter how fancy it is.

How to read the metrics:
- RMSE (root mean squared error): the typical size of a prediction
  error, in the same units as the target (here, daily return).
  Smaller is better; compare it to the target's own standard
  deviation — an RMSE near the std means the model predicts almost
  nothing.
- MAE (mean absolute error): the average absolute miss. Less
  sensitive to one wild day than RMSE.
- R² (out-of-sample): 1 - (model errors / variance of the target).
  1.0 = perfect, 0.0 = as good as predicting the mean, NEGATIVE =
  worse than predicting the mean. Negative out-of-sample R² is
  common in finance and means the model should not be traded.
- Train vs test gap: if train R² is far above test R², the model
  memorised the past (overfitting). A small gap with both near zero
  means the signal itself is weak — the honest finding for daily
  stock returns.

Worked example: predict tomorrow's AAPL return from today's
features (lagged returns, trailing volatility, price vs its 50-day
average) with a Ridge regression inside a scaling Pipeline.
"""
import numpy as np
import pandas as pd


def build_features(prices):
    """Features known at today's close; target is TOMORROW's return.
    Every feature is lagged by construction — nothing peeks ahead."""
    aapl = prices["AAPL"].pct_change()
    spy = prices["SPY"].pct_change()
    msft = prices["MSFT"].pct_change()
    frame = pd.DataFrame({
        "aapl_lag1": aapl.shift(1),
        "spy_lag1": spy.shift(1),
        "msft_lag1": msft.shift(1),
        "vol5": aapl.rolling(5).std().shift(1),
        "ma_ratio": (prices["AAPL"] / prices["AAPL"].rolling(50).mean() - 1).shift(1),
        "target": aapl.shift(-1),
    }).dropna()
    return frame


FEATURES = ["aapl_lag1", "spy_lag1", "msft_lag1", "vol5", "ma_ratio"]


def _metrics(y_true, y_pred, label):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    print(f"  {label}: RMSE {rmse:.4f}, MAE {mae:.4f}, R² {r2:+.3f}")
    return rmse, mae, r2


def run(prices, source):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    frame = build_features(prices)
    split = int(len(frame) * 0.70)  # time-ordered: past trains, future tests
    train, test = frame.iloc[:split], frame.iloc[split:]
    print("D05  scikit-learn workflow — predict tomorrow's AAPL return")
    print(f"  {len(frame)} usable days; train {len(train)} (past), "
          f"test {len(test)} (future) — no shuffling, no leakage")
    print(f"  Target std (test): {test['target'].std():.4f} — any RMSE "
          f"near this number means the model predicts almost nothing")

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[FEATURES], train["target"])
    coefs = dict(zip(FEATURES,
                     np.round(model[-1].coef_, 4).tolist()))
    print(f"  Ridge coefficients (on standardised features): {coefs} — "
          f"the largest |coefficient| is the feature the model leans on most")
    rmse_tr, _, r2_tr = _metrics(train["target"],
                                 model.predict(train[FEATURES]), "Train")
    rmse_te, _, r2_te = _metrics(test["target"],
                                 model.predict(test[FEATURES]), "Test ")
    # Naive baselines: predict zero, and predict the train mean.
    _metrics(test["target"], np.zeros(len(test)), "Base (predict 0)")
    _metrics(test["target"],
             np.full(len(test), train["target"].mean()), "Base (predict mean)")
    print(f"  Reading: test R² {r2_te:+.3f} — "
          f"{'the model beats the naive baselines on unseen data, but the edge is small (normal for daily returns)' if r2_te > 0.01 else 'the model does NOT beat predicting the mean out of sample — daily direction is close to noise, which is itself the finding'}")
    print(f"  Overfitting check: train R² {r2_tr:+.3f} vs test R² "
          f"{r2_te:+.3f} — "
          f"{'a big gap would mean memorisation; a small gap means the model is honest about what it knows' if abs(r2_tr - r2_te) < 0.05 else 'a large train/test gap = overfitting: the model memorised the past'}")
    return {"ml_ridge_rmse_test": rmse_te, "ml_ridge_r2_test": r2_te,
            "ml_ridge_r2_train": r2_tr, "ml_n_days": int(len(frame))}
