"""D12 — Statistics libraries: scipy + statsmodels (tool module).

CFA idea: pandas gets the data, NumPy does the arithmetic, but the
STATISTICS live in scipy and statsmodels — distributions, tests and
regressions with full uncertainty reporting. This module is the
reference card: every test below appears somewhere in the repo, and
each one is printed with its null hypothesis in plain words, what a
high/low value means, and what to do about it. (The full glossary
of regression diagnostics lives in D04; this module adds the
distribution and comparison tests D04 does not cover.)

How to read each test (null hypothesis first, then the reading):
- Shapiro-Wilk / Jarque-Bera: H0 = "the data ARE normal". p < 0.05
  rejects normality. Returns are usually NOT normal (fat tails) —
  that is a fact about markets, not a failure of your data.
- One-sample t-test: H0 = "the true mean return is zero". p < 0.05
  means the mean is significantly different from zero; p = 0.4
  means the data cannot tell the mean from zero (the usual result
  for daily returns — the signal is tiny next to the noise).
- Two-sample t-test: H0 = "AAPL and MSFT have the SAME mean return".
  Rejection means one stock really did earn more on average.
- Pearson correlation: linear co-movement, -1 to +1; Spearman: the
  same on RANKS, so it survives outliers and non-linearity. If
  Pearson is much smaller than Spearman, a few wild days are
  distorting the linear measure.
- Mann-Whitney U: H0 = "the two return distributions are the same",
  no normality assumed. The non-parametric backup when Shapiro
  rejects normality.
- Kolmogorov-Smirnov: H0 = "the returns follow the fitted normal
  distribution". Rejection says the normal curve does not fit —
  check the tails (that is where the money is lost).
- OLS with full output: every coefficient with standard error,
  t-stat, p-value and 95% confidence interval. The interval is the
  honest statement: "beta is 1.15, and plausibly anywhere from
  1.09 to 1.21". A wide interval means the data cannot pin the
  number down — say so, do not quote the point estimate alone.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _verdict(p, what):
    return (f"p {p:.3g} -> {what}" if p < 0.05
            else f"p {p:.3g} -> no evidence for {what}")


def run(prices, source):
    import statsmodels.api as sm

    ret = prices.pct_change().dropna()
    aapl = ret["AAPL"]
    msft = ret["MSFT"]
    spy = ret["SPY"]
    print("D12  Statistics libraries — scipy + statsmodels on real returns")
    print(f"  Sample: {len(ret)} daily returns, "
          f"{ret.index[0].date()} to {ret.index[-1].date()}")

    # ---- Distributions ------------------------------------------------
    mu, sigma = float(aapl.mean()), float(aapl.std())
    ks_stat, ks_p = stats.kstest((aapl - mu) / sigma, "norm")
    jb_stat, jb_p, skew, kurt = sm.stats.stattools.jarque_bera(aapl)
    sw_stat, sw_p = stats.shapiro(aapl.sample(1000, random_state=42))
    print(f"  AAPL daily returns: mean {mu:+.4%}, std {sigma:.2%}, "
          f"skew {skew:+.2f}, excess kurtosis {kurt:.2f}")
    print(f"  Jarque-Bera {jb_stat:,.0f} ({_verdict(jb_p, 'non-normality')}) — "
          f"returns are skewed/fat-tailed, so normal-based risk numbers "
          f"(like a 2-sigma VaR) UNDERSTATE tail risk")
    print(f"  Shapiro-Wilk W {sw_stat:.4f} ({_verdict(sw_p, 'non-normality')}) — "
          f"same conclusion by a different route; with 1,000 observations "
          f"this test is sensitive, so the skew/kurtosis above are the "
          f"real evidence")
    print(f"  Kolmogorov-Smirnov vs fitted normal: D {ks_stat:.3f} "
          f"({_verdict(ks_p, 'the normal distribution does not fit')}) — "
          f"the fitted normal curve misses the tails; use the empirical "
          f"distribution (historical VaR, C03) instead")
    q = aapl.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    print(f"  Empirical quantiles: 1% {q[0.01]:+.2%}, 5% {q[0.05]:+.2%}, "
          f"median {q[0.5]:+.2%}, 95% {q[0.95]:+.2%}, 99% {q[0.99]:+.2%} — "
          f"the 1% tail ({q[0.01]:+.2%}) is the number a risk manager "
          f"actually cares about")

    # ---- Comparison tests ---------------------------------------------
    t1, p1 = stats.ttest_1samp(aapl, 0.0)
    print(f"  One-sample t-test (mean AAPL return = 0): t {t1:+.2f} "
          f"({_verdict(p1, 'the mean return differs from zero')}) — "
          f"{'the average daily return is statistically real, though tiny in money terms' if p1 < 0.05 else 'the data cannot distinguish the mean return from zero — daily drift is swamped by daily noise'}")
    t2, p2 = stats.ttest_ind(aapl, msft, equal_var=False)
    print(f"  Two-sample t-test (AAPL vs MSFT mean): t {t2:+.2f} "
          f"({_verdict(p2, 'the two stocks earned different average returns')})")
    r_p, pp = stats.pearsonr(aapl, spy)
    r_s, ps = stats.spearmanr(aapl, spy)
    print(f"  AAPL vs SPY: Pearson {r_p:+.3f} (p {pp:.3g}), Spearman "
          f"{r_s:+.3f} (p {ps:.3g}) — "
          f"{'the two agree, so the relationship is roughly linear and not driven by outliers' if abs(r_p - r_s) < 0.05 else 'Spearman differs from Pearson — a few wild days distort the linear measure; trust the rank version more'}")
    u_stat, pu = stats.mannwhitneyu(aapl, msft)
    print(f"  Mann-Whitney U (AAPL vs MSFT distributions): "
          f"({_verdict(pu, 'the two return distributions differ')}) — "
          f"the non-parametric check, valid even though returns are not normal")

    # ---- OLS with full output + confidence intervals ------------------
    X = sm.add_constant(spy)
    model = sm.OLS(aapl, X).fit()
    ci = model.conf_int()
    print("  OLS: AAPL ~ SPY (full output, condensed):")
    for name in ["const", "SPY"]:
        print(f"    {name:>5}: coef {model.params[name]:+.4f}, "
              f"SE {model.bse[name]:.4f}, t {model.tvalues[name]:+.1f}, "
              f"p {model.pvalues[name]:.3g}, 95% CI "
              f"[{ci.loc[name, 0]:+.4f}, {ci.loc[name, 1]:+.4f}] — "
              f"{'the interval excludes zero, so the effect is real' if model.pvalues[name] < 0.05 else 'the interval covers zero, so the effect may be nothing'}")
    print(f"    R² {model.rsquared:.3f} (adj. {model.rsquared_adj:.3f}), "
          f"F p {model.f_pvalue:.3g} — see D04 for the full reading of "
          f"every one of these numbers")
    return {"stats_jb_p": float(jb_p), "stats_shapiro_p": float(sw_p),
            "stats_ks_p": float(ks_p), "stats_ttest_mean_p": float(p1),
            "stats_pearson": float(r_p), "stats_spearman": float(r_s),
            "stats_beta": float(model.params["SPY"]),
            "stats_beta_ci_lo": float(ci.loc["SPY", 0]),
            "stats_beta_ci_hi": float(ci.loc["SPY", 1])}
