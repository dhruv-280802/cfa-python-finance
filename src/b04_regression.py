"""B04 — Correlation and regression: simple and multiple OLS
(CFA L2 Quant Methods; L2 PSM: "prediction using machine learning —
regression").

CFA idea: regression is the workhorse of "does X explain Y?". Simple
regression has one driver; multiple regression has several. Read the
output in four places: the slope (how much Y moves per unit of X),
its t-statistic/p-value (is the slope real or noise?), R-squared (how
much of Y's variation X explains), and the F-statistic (does the whole
model beat just guessing the mean?).

Worked example: (1) simple OLS — AAPL daily returns on SPY daily
returns (the slope is AAPL's beta); (2) multiple OLS — AAPL returns on
SPY + TLT returns (does adding bonds help explain Apple?). statsmodels
is used when available; otherwise plain NumPy least squares with
hand-computed standard errors.
"""
import numpy as np
import pandas as pd


def _ols(y, X, names):
    """OLS with statsmodels if present, else NumPy least squares.

    Returns a dict with coefficients, t-stats, p-values, R-squared and
    the F-statistic — the same four readings either way.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    try:
        import statsmodels.api as sm

        res = sm.OLS(y, X).fit()
        return {"coef": res.params, "t": res.tvalues, "p": res.pvalues,
                "r2": float(res.rsquared), "f": float(res.fvalue),
                "names": names}
    except ImportError:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = n - k
        s2 = float(resid @ resid / dof)
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t_stats = beta / se
        from scipy import stats as st

        p_vals = 2 * (1 - st.t.cdf(np.abs(t_stats), dof))
        ss_tot = float(((y - y.mean()) ** 2).sum())
        ss_res = float((resid ** 2).sum())
        r2 = 1 - ss_res / ss_tot
        f_stat = (r2 / (k - 1)) / ((1 - r2) / dof) if k > 1 else np.nan
        return {"coef": beta, "t": t_stats, "p": p_vals, "r2": r2,
                "f": float(f_stat), "names": names}


def run(prices, source):
    rets = prices.pct_change().dropna()
    aapl, spy, tlt = rets["AAPL"], rets["SPY"], rets["TLT"]
    corr = float(np.corrcoef(aapl, spy)[0, 1])

    # (1) Simple: AAPL = alpha + beta * SPY.
    X1 = np.column_stack([np.ones(len(spy)), spy])
    s1 = _ols(aapl, X1, ["alpha", "beta(SPY)"])
    # (2) Multiple: AAPL = alpha + b1 * SPY + b2 * TLT.
    X2 = np.column_stack([np.ones(len(spy)), spy, tlt])
    s2 = _ols(aapl, X2, ["alpha", "beta(SPY)", "beta(TLT)"])

    print("B04  Regression (OLS)")
    print(f"  Correlation(AAPL, SPY) = {corr:.3f}")
    print(f"  (1) Simple  AAPL ~ SPY: alpha {s1['coef'][0]:+.4%}/day, "
          f"beta {s1['coef'][1]:.3f} (t = {s1['t'][1]:.1f}, "
          f"p = {s1['p'][1]:.2e}), R2 = {s1['r2']:.3f}")
    print(f"      Plain English: AAPL moves about {s1['coef'][1]:.2f}x the "
          f"market each day; the market explains {s1['r2']:.0%} of "
          f"AAPL's daily moves.")
    print(f"  (2) Multiple AAPL ~ SPY + TLT: beta(SPY) {s2['coef'][1]:.3f} "
          f"(t = {s2['t'][1]:.1f}), beta(TLT) {s2['coef'][2]:+.3f} "
          f"(t = {s2['t'][2]:.1f}, p = {s2['p'][2]:.3f}), "
          f"R2 = {s2['r2']:.3f}, F = {s2['f']:.0f}")
    print(f"      Plain English: adding bonds {'helps' if s2['r2'] > s1['r2'] + 0.005 else 'barely helps'} "
          f"explain AAPL (R2 {s1['r2']:.3f} -> {s2['r2']:.3f}); "
          f"the TLT slope is "
          f"{'significant' if s2['p'][2] < 0.05 else 'not significant'}.")
    return {"corr_aapl_spy": corr, "aapl_beta": float(s1["coef"][1]),
            "aapl_alpha_daily": float(s1["coef"][0]),
            "simple_r2": float(s1["r2"]), "multiple_r2": float(s2["r2"]),
            "tlt_beta": float(s2["coef"][2]),
            "tlt_beta_p": float(s2["p"][2])}
