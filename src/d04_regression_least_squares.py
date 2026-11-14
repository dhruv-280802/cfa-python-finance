"""D04 — Regression and least squares (CFA L2 PSM Unit 3: "Financial
Forecasting Using Machine and Deep Learning").

CFA idea: regression is the workhorse of finance — beta is a
regression slope, alpha is a regression intercept. The least-squares
criterion says: choose the line that minimises the SUM OF SQUARED
RESIDUALS (the vertical gaps between the line and the data). Squaring
punishes big misses more than small ones, which is why one wild day
can drag a regression around.

How to read every number this module prints (the same reading applies
to every regression in this repo):
- Coefficient: "a 1-unit rise in X is associated with a beta-unit
  rise in Y" — a slope of 1.15 means AAPL moves ~1.15% when the
  market moves 1%. Good or bad depends on the question, not the size.
- t-statistic: coefficient / standard error. |t| > ~2 means the
  estimate is more than twice its own noise — the usual "significant"
  cut-off. Small |t| = the coefficient could easily be zero.
- p-value: the probability of seeing a t this big if the true
  coefficient were zero. p < 0.05 = "significant" by convention;
  p = 0.4 = no evidence at all. A p-value is NOT the probability the
  coefficient is real — it is a statement about the data under a
  null hypothesis.
- R-squared: the share of Y's variance the model explains (0 to 1).
  0.50 = half the movement explained, half left over. High R² is not
  automatically good (a trending series can fake it); low R² is not
  automatically bad (daily stock returns are mostly noise — R² of
  0.02 can still be real if the t-stat is large).
- F-statistic / Prob(F): does the model as a whole beat "just use the
  mean"? Small Prob(F) = yes, jointly the regressors matter.
- Durbin-Watson: ~2 = residuals uncorrelated (good); near 0 = strong
  positive autocorrelation (bad — standard errors are too small);
  near 4 = negative autocorrelation.
- Jarque-Bera (on residuals): tests whether residuals look normal.
  Big JB / small p = residuals are skewed or fat-tailed, so t-tests
  are approximate, not exact.
- Shapiro-Wilk (on residuals): also tests normality; null = "the
  residuals ARE normal". p < 0.05 rejects normality. It is more
  sensitive than Jarque-Bera on small samples, but with 1,000+
  observations it rejects on tiny deviations — so read it together
  with the skew/kurtosis numbers, not alone.
- ADF (Augmented Dickey-Fuller): tests whether a SERIES is
  stationary (stable mean/variance over time). Null = "there IS a
  unit root" (non-stationary — the series wanders). p < 0.05 rejects
  the null, so the series is stationary and safe to regress.
  Prices are usually non-stationary (ADF fails to reject) while
  RETURNS are stationary — that is exactly why we regress returns,
  not prices. Regressing two wandering prices on each other gives a
  fake high R² (spurious regression).
- VIF (variance inflation factor): for each regressor, how much its
  variance is inflated because it moves with the other regressors.
  VIF = 1 means no correlation with the others; 1-5 is fine;
  5-10 is a warning; above 10 means serious multicollinearity —
  coefficients become unstable and their t-stats shrink, so drop one
  of the correlated variables or combine them.
- Breusch-Pagan / White (heteroskedasticity): null = "residual
  variance is constant". p < 0.05 rejects it — the spread of errors
  changes over time (very common in finance: calm periods, then
  crises). The coefficients stay unbiased, but the standard errors
  are wrong, so t-tests mislead. Fix: use heteroskedasticity-robust
  (White/HAC) standard errors, or model the variance (GARCH, E04).
  White's version also catches non-linear misspecification.
- RESET (Ramsey's specification test): null = "the functional form
  is right" (no missing non-linear terms). p < 0.05 rejects it —
  the model is misspecified, e.g. the true relationship is curved
  or a variable is missing. Fix: add squared terms, interactions,
  or the missing variable.
- R² vs adjusted R²: R² NEVER falls when you add a variable, even a
  useless one — so it rewards kitchen-sink models. Adjusted R²
  subtracts a penalty for each extra regressor: if adjusted R² falls
  when you add a variable, that variable earns nothing and should
  go. Judge models by adjusted R² (and by out-of-sample error),
  never by raw R² alone.

Worked example: simple regression (AAPL returns on SPY) and multiple
regression (AAPL on SPY + MSFT + TLT), estimated with statsmodels
OLS and cross-checked with scikit-learn.
"""
import numpy as np
import pandas as pd


def _ols_interpretation(model, y, X, label):
    """Print a full OLS readout with the interpretation of each stat."""
    r2 = float(model.rsquared)
    adj = float(model.rsquared_adj)
    f_p = float(model.f_pvalue)
    dw = float(__import__("statsmodels.stats.stattools", fromlist=["durbin_watson"]).durbin_watson(model.resid))
    print(f"  --- {label} ---")
    for name, coef, se, t, p in zip(
            ["const"] + list(X.columns), model.params, model.bse,
            model.tvalues, model.pvalues):
        sig = "significant" if p < 0.05 else "NOT significant"
        print(f"    {name:>8}: coef {coef:+.4f} (SE {se:.4f}, "
              f"t {t:+.1f}, p {p:.3g}) -> {sig}: a 1-unit move in "
              f"{name} is associated with a {coef:+.4f} move in the target")
    print(f"    R² {r2:.3f}: the model explains {r2:.1%} of the "
          f"target's variance — {'over half' if r2 > 0.5 else 'under half'} "
          f"explained, the rest is noise the model cannot see")
    print(f"    Adj. R² {adj:.3f} (penalises extra regressors — if it "
          f"falls when you add a variable, that variable earns nothing)")
    print(f"    R² vs adj. R² gap {r2 - adj:.4f}: a wide gap means the "
          f"model is padded with regressors that add little; a narrow "
          f"gap means every variable earns its place")
    print(f"    F-test p {f_p:.3g}: "
          f"{'the regressors jointly matter' if f_p < 0.05 else 'the model as a whole does NOT beat the mean'}")
    print(f"    Durbin-Watson {dw:.2f}: "
          f"{'residuals look uncorrelated (good)' if 1.8 < dw < 2.2 else 'residual autocorrelation present — standard errors are suspect'}")
    return r2, dw


def run(prices, source):
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import jarque_bera

    ret = prices.pct_change().dropna()
    y = ret["AAPL"]
    print("D04  Regression & least squares — AAPL daily returns")

    # ---- Simple regression: AAPL on the market -----------------------
    X1 = sm.add_constant(ret[["SPY"]])
    m1 = sm.OLS(y, X1).fit()
    r2_1, _ = _ols_interpretation(m1, y, ret[["SPY"]],
                                  "Simple: AAPL ~ SPY (this slope IS the CAPM beta)")
    beta = float(m1.params["SPY"])
    print(f"  Reading: AAPL's beta is {beta:.2f} — it moves about "
          f"{beta:.2f}x the market on a typical day, and the intercept "
          f"({float(m1.params['const']):+.4f}/day) is its alpha")

    # ---- Multiple regression: add MSFT and bonds ---------------------
    X3 = sm.add_constant(ret[["SPY", "MSFT", "TLT"]])
    m3 = sm.OLS(y, X3).fit()
    r2_3, _ = _ols_interpretation(m3, y, ret[["SPY", "MSFT", "TLT"]],
                                  "Multiple: AAPL ~ SPY + MSFT + TLT")
    print(f"  Reading: R² moved {r2_1:.3f} -> {r2_3:.3f} when MSFT and "
          f"TLT were added — "
          f"{'the extra variables earn their keep' if r2_3 - r2_1 > 0.02 else 'almost nothing new is explained; SPY already carries the signal'}")

    # ---- Residual diagnostics ----------------------------------------
    jb_stat, jb_p, skew, kurt = jarque_bera(m3.resid)
    bp_stat, bp_p, _, _ = het_breuschpagan(m3.resid, X3)
    print(f"  Residual Jarque-Bera {jb_stat:,.0f} (p {jb_p:.3g}, skew "
          f"{skew:+.2f}, kurtosis {kurt:.1f}): "
          f"{'residuals are NOT normal — fat tails/skew, so treat t-tests as approximate' if jb_p < 0.05 else 'no evidence against normal residuals'}")
    print(f"  Breusch-Pagan {bp_stat:.1f} (p {bp_p:.3g}): "
          f"{'heteroskedasticity present — residual variance changes over time, so OLS standard errors are wrong (use robust SEs)' if bp_p < 0.05 else 'no evidence of heteroskedasticity — residual variance looks constant'}")

    # ---- Extended diagnostics: the rest of the standard battery ------
    from scipy import stats as _st
    from statsmodels.stats.diagnostic import het_white, linear_reset
    from statsmodels.stats.outliers_influence import (
        variance_inflation_factor)
    from statsmodels.tsa.stattools import adfuller

    # (a) Shapiro-Wilk normality on residuals (with the big-sample caveat).
    sw_stat, sw_p = _st.shapiro(m3.resid)
    print(f"  Residual Shapiro-Wilk W {sw_stat:.4f} (p {sw_p:.3g}): "
          f"{'rejects normality — same message as Jarque-Bera, read with the skew/kurtosis above' if sw_p < 0.05 else 'no evidence against normal residuals'} "
          f"(with {len(m3.resid)} observations this test is very sensitive — a rejection on its own is not a crisis, it is a prompt to check skew/kurtosis)")

    # (b) ADF stationarity: prices wander, returns do not.
    adf_price = adfuller(prices["SPY"].dropna(), autolag="AIC")
    adf_ret = adfuller(ret["SPY"].dropna(), autolag="AIC")
    print(f"  ADF on SPY prices: stat {adf_price[0]:.2f}, p {adf_price[1]:.3g} "
          f"-> {'non-stationary (unit root NOT rejected) — prices wander, never regress prices on prices' if adf_price[1] >= 0.05 else 'stationary (unexpected for a price level)'}")
    print(f"  ADF on SPY returns: stat {adf_ret[0]:.2f}, p {adf_ret[1]:.3g} "
          f"-> {'stationary — returns have a stable mean/variance, safe to regress' if adf_ret[1] < 0.05 else 'non-stationary even in returns — investigate before trusting the regression'}")

    # (c) VIF multicollinearity for the multiple regression.
    print("  VIF (variance inflation factor) — 1 = clean, 5-10 = warning, >10 = serious:")
    vif = {}
    for i, col in enumerate(X3.columns):
        if col == "const":
            continue
        v = float(variance_inflation_factor(X3.values, i))
        vif[col] = v
        print(f"    {col:>5}: VIF {v:.2f} -> "
              f"{'clean — moves independently of the other regressors' if v < 5 else 'warning — correlated with the others, coefficients are less stable' if v < 10 else 'SERIOUS multicollinearity — drop or combine this variable'}")

    # (d) White test (heteroskedasticity + missed non-linearity).
    white_lm, white_p, white_f, white_fp = het_white(m3.resid, X3)
    print(f"  White test LM {white_lm:.1f} (p {white_p:.3g}): "
          f"{'rejects constant variance and/or the linear form — use robust standard errors and check for a curved relationship' if white_p < 0.05 else 'no evidence against constant variance or the linear form'}")

    # (e) RESET specification test (is the functional form right?).
    try:
        reset = linear_reset(m3, power=3, test_type="fitted", use_f=True)
        # statsmodels returns a ContrastResults object (older versions
        # returned a plain tuple) — read both shapes defensively.
        if hasattr(reset, "fvalue"):
            reset_f, reset_p = float(reset.fvalue), float(reset.pvalue)
        else:
            reset_f, reset_p = float(reset[0]), float(reset[1])
        print(f"  RESET test F {reset_f:.2f} (p {reset_p:.3g}): "
              f"{'REJECTS the linear specification — the true relationship is probably curved or a variable is missing; try adding squared terms' if reset_p < 0.05 else 'no evidence the linear form is wrong'}")
    except Exception as exc:  # noqa: BLE001 - RESET is a bonus, not the lesson
        reset_f, reset_p = np.nan, np.nan
        print(f"  RESET test unavailable ({exc})")

    # ---- Cross-check with scikit-learn (same least squares) ----------
    from sklearn.linear_model import LinearRegression

    sk = LinearRegression().fit(ret[["SPY", "MSFT", "TLT"]], y)
    print(f"  scikit-learn cross-check: coefficients "
          f"{np.round(sk.coef_, 4).tolist()}, intercept "
          f"{sk.intercept_:+.4f} — identical to OLS, because both solve "
          f"the same least-squares problem; statsmodels just reports "
          f"the uncertainty, sklearn just fits")
    return {"ols_beta_spy": beta, "ols_r2_simple": r2_1,
            "ols_r2_multiple": r2_3, "ols_jb_p": float(jb_p),
            "ols_bp_p": float(bp_p), "ols_shapiro_p": float(sw_p),
            "ols_adf_price_p": float(adf_price[1]),
            "ols_adf_return_p": float(adf_ret[1]),
            "ols_white_p": float(white_p), "ols_reset_p": float(reset_p),
            "ols_max_vif": float(max(vif.values()))}
