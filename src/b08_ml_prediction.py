"""B08 — Machine-learning prediction workflow (CFA L2 PSM: "prediction
using machine learning — regression and artificial neural networks").

CFA idea: a prediction model is a workflow, not a magic box — build
features, split train/test WITHOUT shuffling time (the future must never
leak into training), fit, then judge on data the model has never seen.
A model that cannot beat "always guess the most common outcome" is not
a model, it is a coin flip with extra steps.

Worked example: predict the SIGN of tomorrow's AAPL return from today's
features — yesterday's AAPL return, yesterday's SPY return, and the
trailing 5-day volatility. Logistic regression (statsmodels Logit).
Baseline: always predict "up" (the majority class). Report train vs
test accuracy so overfitting is visible, not hidden.
"""
import numpy as np
import pandas as pd


def build_features(prices):
    """Features (all known at today's close) and tomorrow's direction."""
    aapl = prices["AAPL"].pct_change()
    spy = prices["SPY"].pct_change()
    vol5 = aapl.rolling(5).std()
    frame = pd.DataFrame({
        "aapl_lag1": aapl.shift(1),
        "spy_lag1": spy.shift(1),
        "vol5": vol5.shift(1),
        "target_up": (aapl.shift(-1) > 0).astype(float),
    }).dropna()
    return frame


def fit_logit(X_train, y_train):
    """Logistic regression via statsmodels; falls back to a simple
    threshold rule (predict up when the SPY lag is positive) if the
    fit fails to converge."""
    try:
        import statsmodels.api as sm

        model = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=0)
        return ("logit", model)
    except Exception as exc:  # noqa: BLE001 - fall back, don't crash
        print(f"  Logit fit failed ({exc}); using threshold rule.")
        return ("threshold", None)


def predict(kind, model, X):
    if kind == "logit":
        import statsmodels.api as sm

        probs = model.predict(sm.add_constant(X, has_constant="add"))
        return (np.asarray(probs) > 0.5).astype(int)
    return (np.asarray(X["spy_lag1"]) > 0).astype(int)


def run(prices, source):
    frame = build_features(prices)
    features = ["aapl_lag1", "spy_lag1", "vol5"]
    # Time-ordered split: first 70% trains, last 30% tests. No shuffling.
    split = int(len(frame) * 0.70)
    train, test = frame.iloc[:split], frame.iloc[split:]
    kind, model = fit_logit(train[features], train["target_up"])
    pred_train = predict(kind, model, train[features])
    pred_test = predict(kind, model, test[features])
    acc_train = float((pred_train == train["target_up"].to_numpy()).mean())
    acc_test = float((pred_test == test["target_up"].to_numpy()).mean())
    baseline = float(test["target_up"].mean())  # always predict "up"
    n_up = int(test["target_up"].sum())

    print("B08  ML prediction workflow (logistic regression)")
    print(f"  Target: sign of tomorrow's AAPL return; "
          f"{len(frame)} usable days, split {len(train)}/{len(test)} "
          f"(train/test, time-ordered)")
    if kind == "logit":
        print("  Coefficients (log-odds): " + ", ".join(
            f"{n} {c:+.3f}" for n, c in
            zip(["const"] + features, model.params)))
    print(f"  Train accuracy: {acc_train:.1%}; test accuracy: "
          f"{acc_test:.1%} (out-of-sample, the only one that counts)")
    print(f"  Baseline (always predict up): {baseline:.1%} "
          f"({n_up} of {len(test)} test days were up)")
    print(f"  Verdict: the model {'beats' if acc_test > baseline else 'does NOT beat'} "
          f"the baseline — daily direction is close to unpredictable, "
          f"which is itself the honest finding")
    return {"ml_train_acc": acc_train, "ml_test_acc": acc_test,
            "ml_baseline": baseline, "ml_model": kind}
