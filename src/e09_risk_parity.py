"""E09 — Risk parity (Hilpisch "Python for Finance": portfolio
optimisation beyond mean-variance).

CFA idea: mean-variance optimisation is fragile — it loads up on
whatever asset had the best recent Sharpe and collapses when that
asset stumbles. RISK PARITY asks a different question: instead of
equal DOLLARS in each asset, put equal RISK in each asset. Each
asset's risk contribution is weight x marginal risk, and the
portfolio is risk-parity when all four contributions are equal.
The result is usually bond-heavy (bonds are calm, so they need a
big weight to contribute as much risk as equities) — which is why
real risk-parity funds use leverage to hit their return target.

How to read the result:
- Weights: compare with equal weight (25% each). If TLT's weight
  is far above 25%, that is the parity logic at work — calm assets
  need bigger weights to contribute equal risk.
- Risk contributions: in the parity portfolio they should be
  nearly identical (each ~25% of total risk). In the equal-weight
  portfolio, equities dominate the risk (often 70%+ of it) — the
  portfolio is "diversified" in dollars but concentrated in risk.
- Volatility: risk parity usually has LOWER volatility than equal
  weight (that is the point — it harvests the diversification
  benefit deliberately). If it does not, the assets are too
  correlated for parity to help.
- The catch: equal risk is not equal return. In a bull market the
  parity portfolio lags badly (too much in bonds); in a crash it
  holds up better. Risk parity is a risk choice, not a return hack.
- And the honest caveat: when correlations spike toward 1 in a
  crisis, diversification evaporates and every allocation looks
  the same — risk parity included.
"""
import numpy as np


def _risk_contributions(weights, cov):
    """Each asset's share of total portfolio volatility."""
    port_vol = float(np.sqrt(weights @ cov @ weights))
    marginal = cov @ weights / port_vol
    rc = weights * marginal
    return rc / rc.sum()


def run(prices, source):
    from scipy.optimize import minimize

    ret = prices.pct_change().dropna()
    names = list(ret.columns)
    cov = ret.cov().to_numpy() * 252
    mu = ret.mean().to_numpy() * 252
    print("E09  Risk parity — equal risk, not equal dollars")
    n = len(names)
    ew = np.ones(n) / n
    rc_ew = _risk_contributions(ew, cov)
    print("  Equal-weight risk contributions: " + ", ".join(
        f"{name} {rc:.0%}" for name, rc in zip(names, rc_ew)) +
        " — equities dominate the risk even though the dollars are equal")

    def objective(w):
        rc = _risk_contributions(w, cov)
        target = np.ones(n) / n
        return float(np.sum((rc - target) ** 2))

    res = minimize(objective, ew, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints={"type": "eq",
                                "fun": lambda w: float(np.sum(w) - 1)},
                   options={"maxiter": 500})
    w_rp = res.x
    rc_rp = _risk_contributions(w_rp, cov)
    vol_ew = float(np.sqrt(ew @ cov @ ew))
    vol_rp = float(np.sqrt(w_rp @ cov @ w_rp))
    ret_ew = float(ew @ mu)
    ret_rp = float(w_rp @ mu)
    print("  Risk-parity weights: " + ", ".join(
        f"{name} {w:.0%}" for name, w in zip(names, w_rp)))
    print("  Risk-parity risk contributions: " + ", ".join(
        f"{name} {rc:.0%}" for name, rc in zip(names, rc_rp)) +
        " — nearly equal, which is the definition of the portfolio")
    print(f"  Equal weight: return {ret_ew:.1%}/yr, vol {vol_ew:.1%}, "
          f"Sharpe {(ret_ew - 0.04) / vol_ew:.2f}")
    print(f"  Risk parity:  return {ret_rp:.1%}/yr, vol {vol_rp:.1%}, "
          f"Sharpe {(ret_rp - 0.04) / vol_rp:.2f} — "
          f"{'lower vol, as designed — the parity portfolio harvests diversification deliberately' if vol_rp < vol_ew else 'no vol benefit here — the assets are too correlated for parity to help'}")
    print("  Reading: risk parity is a RISK choice, not a return hack — "
          "expect it to lag in bull markets (too much in bonds) and "
          "hold up better in crashes. Real funds lever it up to hit "
          "their return target")
    return {"rp_weights": {name: float(w) for name, w in zip(names, w_rp)},
            "rp_vol": vol_rp, "rp_ret": ret_rp,
            "rp_max_rc_dev": float(np.max(np.abs(rc_rp - 1 / n)))}
