"""B03 — Hypothesis testing (CFA L2 Quant Methods).

CFA idea: a hypothesis test turns "is the average return really positive,
or did we just get lucky?" into a number — the p-value. H0 is the boring
claim (mean return = 0); a small p-value (< 0.05) says the data would be
very surprising if H0 were true, so we reject it. A confidence interval
says the same thing in the units of the data.

Worked example: (1) one-sample t-test — is SPY's mean daily return
above zero? (2) two-sample t-test — do AAPL and MSFT have different mean
daily returns? (3) a 95% confidence interval for SPY's mean return.
"""
import numpy as np
from scipy import stats


def run(prices, source):
    rets = prices.pct_change().dropna()
    spy, aapl, msft = rets["SPY"], rets["AAPL"], rets["MSFT"]

    # (1) One-sample t-test: H0: mean SPY return = 0.
    t1, p1 = stats.ttest_1samp(spy, 0.0)
    # (2) Two-sample t-test: H0: mean AAPL return = mean MSFT return.
    t2, p2 = stats.ttest_ind(aapl, msft, equal_var=False)
    # (3) 95% CI for the mean SPY return (annualised for readability).
    mean, se = spy.mean(), spy.std(ddof=1) / np.sqrt(len(spy))
    ci_lo, ci_hi = stats.t.interval(0.95, len(spy) - 1, loc=mean, scale=se)

    print("B03  Hypothesis testing")
    print(f"  (1) SPY mean daily return {spy.mean():+.4%}: "
          f"t = {t1:.2f}, p = {p1:.4f} -> "
          f"{'REJECT H0 (mean > 0)' if p1 < 0.05 else 'cannot reject H0'}")
    print(f"  (2) AAPL mean {aapl.mean():+.4%} vs MSFT mean {msft.mean():+.4%}: "
          f"t = {t2:.2f}, p = {p2:.4f} -> "
          f"{'REJECT H0 (means differ)' if p2 < 0.05 else 'cannot reject H0'}")
    print(f"  (3) 95% CI for SPY mean daily return: "
          f"[{ci_lo:+.4%}, {ci_hi:+.4%}] "
          f"(annualised: [{ci_lo * 252:+.1%}, {ci_hi * 252:+.1%}])")
    return {"spy_ttest_p": float(p1), "aapl_msft_ttest_p": float(p2),
            "spy_mean_ci_lo": float(ci_lo), "spy_mean_ci_hi": float(ci_hi)}
