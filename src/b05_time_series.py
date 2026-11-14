"""B05 — Time-series basics: stationarity, autocorrelation, volatility
clustering (CFA L2 Quant Methods).

CFA idea: most finance time series are *non-stationary* in levels
(prices wander) and *stationary* in returns (returns revert). If you
regress one wandering series on another you get nonsense "significant"
results — the classic spurious regression. So the first test on any
series is always: is it stationary? The ADF test answers it.

Worked examples: (1) ADF on SPY prices (expect: non-stationary) vs SPY
returns (expect: stationary); (2) lag-1 autocorrelation of returns and
of squared returns — the second one shows volatility clustering (big
moves follow big moves); (3) an AR(1) fit on returns; (4) rolling
60-day volatility, latest value and its range.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def adf_summary(series, name):
    """Augmented Dickey-Fuller test. H0: unit root (non-stationary).
    Returns (statistic, p-value, stationary_at_5pct)."""
    stat, p, _, _, crit, _ = adfuller(series.dropna(), maxlag=20,
                                      autolag="AIC", result_object=False)
    return float(stat), float(p), bool(p < 0.05)


def run(prices, source):
    spy_price = prices["SPY"]
    spy_ret = spy_price.pct_change().dropna()
    log_ret = np.log(spy_price).diff().dropna()

    stat_p, p_p, stat_price = adf_summary(spy_price, "SPY price")
    stat_r, p_r, stat_ret = adf_summary(spy_ret, "SPY returns")
    ac1 = float(spy_ret.autocorr(lag=1))
    ac1_sq = float((spy_ret ** 2).autocorr(lag=1))
    ac5 = float(spy_ret.autocorr(lag=5))

    # AR(1) on returns: r_t = c + phi * r_{t-1} + e_t.
    frame = pd.DataFrame({"r": spy_ret, "lag": spy_ret.shift(1)}).dropna()
    ar1 = sm.OLS(frame["r"], sm.add_constant(frame["lag"])).fit()
    phi = float(ar1.params.iloc[1])
    phi_p = float(ar1.pvalues.iloc[1])

    roll_vol = spy_ret.rolling(60).std() * np.sqrt(252)
    latest_vol = float(roll_vol.iloc[-1])
    max_vol = float(roll_vol.max())
    min_vol = float(roll_vol.min())

    print("B05  Time-series basics")
    print(f"  ADF on SPY price levels: stat {stat_p:.2f}, p = {p_p:.3f} -> "
          f"{'stationary' if stat_price else 'NON-STATIONARY (as expected)'}")
    print(f"  ADF on SPY daily returns: stat {stat_r:.2f}, p = {p_r:.2e} -> "
          f"{'stationary' if stat_ret else 'non-stationary'}")
    print(f"  Return autocorrelation: lag-1 {ac1:+.3f}, lag-5 {ac5:+.3f} "
          f"(near zero = returns are hard to predict)")
    print(f"  Squared-return autocorrelation lag-1: {ac1_sq:+.3f} "
          f"(> 0 = volatility clusters: big days follow big days)")
    print(f"  AR(1) on returns: phi = {phi:+.3f} (p = {phi_p:.3f}) — "
          f"yesterday's return explains almost nothing about today's")
    print(f"  Rolling 60-day vol: latest {latest_vol:.1%}, "
          f"range over sample {min_vol:.1%} to {max_vol:.1%}")
    return {"adf_price_p": p_p, "adf_returns_p": p_r,
            "autocorr_lag1": ac1, "autocorr_sq_lag1": ac1_sq,
            "ar1_phi": phi, "ar1_phi_p": phi_p,
            "rolling_vol_latest": latest_vol,
            "rolling_vol_max": max_vol}
