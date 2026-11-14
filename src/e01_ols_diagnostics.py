"""E01 — OLS diagnostics: the full battery (Hilpisch "Python for
Finance": risk, stylised facts, and the regressions every quant
runs before trusting a model).

CFA idea: a regression is only as good as its ASSUMPTIONS. OLS is
BLUE (best linear unbiased estimator) only if: the form is linear,
the regressors are not collinear, the errors have constant variance,
the errors are uncorrelated, and (for exact t-tests) the errors are
normal. This module runs the whole diagnostic battery on the CAPM
regression (AAPL on SPY) and, for EVERY test, prints: what it
checks, the null hypothesis in plain words, how to read the number,
whether the result is good or bad, and WHAT TO DO if it fails.

The battery (each test is also defined in D04's glossary):
- VIF: multicollinearity. VIF > 10 = serious; fix by dropping or
  combining a variable.
- Breusch-Pagan / White: heteroskedasticity. p < 0.05 = variance is
  not constant; fix with robust (HC1) standard errors or a GARCH
  model (E04). The coefficients stay unbiased — only the standard
  errors are wrong.
- Durbin-Watson + Breusch-Godfrey: autocorrelation of residuals.
  DW near 2 is clean; Breusch-Godfrey tests several lags at once
  (H0: no serial correlation up to lag 5). Fix: add lagged terms,
  or use HAC (Newey-West) standard errors.
- Jarque-Bera / Shapiro-Wilk: normality of residuals. Rejection is
  normal for financial data (fat tails); with 1,000+ observations
  the t-tests are still approximately valid, but prefer robust
  standard errors and check the skew/kurtosis.
- RESET: functional form. Rejection = the relationship is curved or
  a variable is missing; fix by adding squared terms or the missing
  variable.
- Robust vs classical standard errors: the side-by-side comparison
  shows whether any conclusion CHANGES when the assumptions are
  dropped. If a coefficient is significant classically but not
  robustly, the classical result was an artefact of the broken
  assumption — trust the robust one.
"""
import numpy as np
import pandas as pd


def _say(label, stat, p, good_text, bad_text, fix):
    """One-line verdict: the number, what it means, what to do.
    For every diagnostic here the null is 'the assumption HOLDS', so
    a small p-value (< 0.05) REJECTS it — the assumption is violated."""
    bad = p < 0.05
    print(f"  {label}: {stat} (p {p:.3g}) -> "
          f"{bad_text if bad else good_text}")
    if bad:
        print(f"      Fix: {fix}")


def run(prices, source):
    import statsmodels.api as sm
    from scipy import stats as st
    from statsmodels.stats.diagnostic import (
        acorr_breusch_godfrey, het_breuschpagan, het_white, linear_reset)
    from statsmodels.stats.outliers_influence import (
        variance_inflation_factor)
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    ret = prices.pct_change().dropna()
    y = ret["AAPL"]
    X = sm.add_constant(ret[["SPY", "MSFT", "TLT"]])
    model = sm.OLS(y, X).fit()
    print("E01  OLS diagnostics — the full battery (AAPL ~ SPY + MSFT + TLT)")
    print(f"  Sample: {len(y)} daily returns. Null hypothesis for every "
          f"test below is 'the assumption HOLDS'; a small p-value "
          f"(< 0.05) REJECTS it, i.e. the assumption is violated")

    # ---- VIF ----------------------------------------------------------
    print("  VIF (multicollinearity: 1 = clean, 5-10 = warning, >10 = serious):")
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        v = float(variance_inflation_factor(X.values, i))
        print(f"    {col:>5}: VIF {v:.2f} -> "
              f"{'clean' if v < 5 else 'warning — correlated with the others' if v < 10 else 'SERIOUS — drop or combine this variable'}")

    # ---- Heteroskedasticity -------------------------------------------
    bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, X)
    _say("Breusch-Pagan (H0: constant error variance)",
         f"LM {bp_stat:.1f}", bp_p,
         "constant variance (good)",
         "HETEROSKEDASTICITY — error variance changes over time (bad: standard errors wrong)",
         "use heteroskedasticity-robust (HC1) standard errors, or model the variance with GARCH (E04)")
    white_lm, white_p, _, _ = het_white(model.resid, X)
    _say("White test (H0: constant variance AND linear form)",
         f"LM {white_lm:.1f}", white_p,
         "constant variance, linear form OK (good)",
         "REJECTED — variance not constant and/or the form is non-linear (bad)",
         "robust standard errors; add squared terms / interactions and re-test")

    # ---- Autocorrelation ----------------------------------------------
    dw = float(durbin_watson(model.resid))
    print(f"  Durbin-Watson {dw:.2f} -> "
          f"{'residuals look uncorrelated (good)' if 1.8 < dw < 2.2 else 'POSITIVE autocorrelation (bad: standard errors too small, t-stats inflated)' if dw < 1.8 else 'negative autocorrelation (unusual — check for over-differencing)'}")
    if not 1.8 < dw < 2.2:
        print("      Fix: add lagged returns as regressors, or use HAC (Newey-West) standard errors")
    bg_stat, bg_p, _, _ = acorr_breusch_godfrey(model, nlags=5)
    _say("Breusch-Godfrey, 5 lags (H0: no serial correlation)",
         f"LM {bg_stat:.1f}", bg_p,
         "no serial correlation up to 5 lags (good)",
         "SERIAL CORRELATION up to lag 5 (bad: yesterday's error predicts today's)",
         "add lagged dependent variable / AR terms, or HAC standard errors")

    # ---- Normality ------------------------------------------------------
    jb_stat, jb_p, skew, kurt = jarque_bera(model.resid)
    _say("Jarque-Bera (H0: residuals are normal)",
         f"JB {jb_stat:,.0f} (skew {skew:+.2f}, kurtosis {kurt:.1f})",
         jb_p,
         "residuals look normal (good)",
         "residuals are NOT normal — skewed / fat-tailed (expected for stock returns)",
         "with 1,000+ observations t-tests are still approximately valid; prefer robust SEs and check the tails directly")
    sw_stat, sw_p = st.shapiro(model.resid.sample(1000, random_state=42))
    _say("Shapiro-Wilk (H0: residuals are normal)",
         f"W {sw_stat:.4f}", sw_p,
         "no evidence against normality (good)",
         "rejects normality (same message as Jarque-Bera; read with the skew/kurtosis)",
         "same as above — do not panic, do check the tails")

    # ---- Specification --------------------------------------------------
    try:
        reset_obj = linear_reset(model, power=3, test_type="fitted",
                                 use_f=True)
        # ContrastResults in current statsmodels (a tuple in older ones).
        if hasattr(reset_obj, "fvalue"):
            reset_f, reset_p = float(reset_obj.fvalue), float(reset_obj.pvalue)
        else:
            reset_f, reset_p = float(reset_obj[0]), float(reset_obj[1])
        _say("RESET (H0: the linear form is correct)",
             f"F {reset_f:.2f}", reset_p,
             "linear form looks right (good)",
             "MISSPECIFICATION — the true relationship is curved or a variable is missing (bad)",
             "add squared terms / interactions, or find the missing variable")
    except Exception as exc:  # noqa: BLE001 - RESET is a bonus
        reset_p = np.nan
        print(f"  RESET test unavailable ({exc})")

    # ---- Robust vs classical standard errors ---------------------------
    robust = model.get_robustcov_results(cov_type="HC1")
    print("  Classical vs robust (HC1) standard errors — does any "
          "conclusion change when the assumptions are dropped?")
    for name in ["SPY", "MSFT", "TLT"]:
        p_classic = float(model.pvalues[name])
        p_robust = float(robust.pvalues[list(X.columns).index(name)])
        changed = (p_classic < 0.05) != (p_robust < 0.05)
        print(f"    {name:>5}: classical p {p_classic:.3g}, robust p "
              f"{p_robust:.3g} -> "
              f"{'CONCLUSION CHANGES — the classical result was an artefact of the broken assumption; trust the robust one' if changed else 'same conclusion either way (good — the result is robust)'}")
    return {"diag_max_vif": float(max(
        variance_inflation_factor(X.values, i)
        for i, c in enumerate(X.columns) if c != "const")),
        "diag_bp_p": float(bp_p), "diag_white_p": float(white_p),
        "diag_dw": dw, "diag_bg_p": float(bg_p),
        "diag_jb_p": float(jb_p), "diag_reset_p": float(reset_p)}
