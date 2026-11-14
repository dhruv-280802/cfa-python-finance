"""E10 — Black-Litterman (CFA L3 portfolio management; Hilpisch
"Python for Finance": portfolio optimisation).

CFA idea: mean-variance optimisation is an "error maximiser" — it
loads up on whatever the inputs say is best, and the inputs
(expected returns) are the least reliable numbers in finance.
Black-Litterman fixes the INPUTS instead of the optimiser:
1. Start from EQUILIBRIUM: the market portfolio is (by definition)
   the average investor's holding, so the returns that make the
   market portfolio optimal — the implied equilibrium returns — are
   the neutral starting point. No views, no bets.
2. Add VIEWS: "AAPL will beat MSFT by 2%/yr" (an absolute view on
   the DIFFERENCE, with a confidence level tau). Views tilt the
   returns, and the tilt is proportional to your confidence.
3. BLEND: the posterior returns are a precision-weighted average
   of equilibrium and views. A low-confidence view barely moves
   anything; a high-confidence view dominates.
The result is a portfolio that looks like the market, tilted toward
your views — instead of the extreme corner portfolios plain
mean-variance spits out.

How to read the result:
- Equilibrium returns: the returns implied by market capitalisation
  weights. They are the "no view" answer — any tilt away from them
  must be earned by a view.
- Posterior returns: equilibrium moved toward the view. AAPL's
  posterior should be ABOVE its equilibrium (the view says it
  beats MSFT); MSFT's should be BELOW. If they barely move, tau
  (confidence) is too low; if they move violently, tau is too high.
- Weights: the maximum-Sharpe portfolio on posterior returns vs on
  equilibrium returns. The BL weights should be a modest tilt, not
  a wholesale rotation — if the optimiser still piles into one
  asset, the views are too strong or the covariance is too
  concentrated.
- Market caps here use yfinance info (SPY/TLT market cap is ETF
  AUM — a proxy for the market portfolio, labelled as such).
"""
import numpy as np
import pandas as pd


def _max_sharpe(mu, cov, rf=0.04):
    from scipy.optimize import minimize

    n = len(mu)

    def neg_sharpe(w):
        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        return -(port_ret - rf) / port_vol if port_vol > 0 else 0.0

    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints={"type": "eq",
                                "fun": lambda w: float(np.sum(w) - 1)},
                   options={"maxiter": 500})
    return res.x


def run(prices, source):
    import yfinance as yf

    ret = prices.pct_change().dropna()
    names = list(ret.columns)
    cov = ret.cov().to_numpy() * 252
    print("E10  Black-Litterman — equilibrium + one view")
    # Market-cap weights (SPY/TLT caps are ETF AUM — a proxy, labelled).
    # The info endpoint is flaky under load, so one retry before falling back.
    import time

    caps = {}
    for name in names:
        info = {}
        for attempt in range(2):
            try:
                info = yf.Ticker(name).info
                break
            except Exception:  # noqa: BLE001 - retry once, then fall back
                if attempt == 0:
                    time.sleep(2)
        caps[name] = float(info.get("marketCap") or np.nan)
    if any(np.isnan(v) for v in caps.values()):
        caps = {name: 1.0 for name in names}  # fall back to equal weight
        print("  (Market caps unavailable — using equal-weight market portfolio as the prior)")
    total_cap = sum(caps.values())
    w_mkt = np.array([caps[n] / total_cap for n in names])
    print("  Market weights: " + ", ".join(
        f"{n} {w:.0%}" for n, w in zip(names, w_mkt)))

    delta = 2.5  # risk-aversion coefficient (assumption, standard value)
    pi = delta * cov @ w_mkt  # implied equilibrium returns
    print("  Equilibrium returns (implied by the market portfolio): "
          + ", ".join(f"{n} {r:.1%}" for n, r in zip(names, pi)))

    # One view: AAPL beats MSFT by 2%/yr. P picks the difference.
    tau = 0.05  # confidence in the view (small = modest tilt)
    i_aapl, i_msft = names.index("AAPL"), names.index("MSFT")
    P = np.zeros((1, len(names)))
    P[0, i_aapl] = 1.0
    P[0, i_msft] = -1.0
    Q = np.array([0.02])
    omega = np.array([[tau * float(P @ cov @ P.T)]])
    # Posterior: BL formula.
    middle = np.linalg.inv(np.linalg.inv(tau * cov) + P.T @ np.linalg.inv(omega) @ P)
    post_mu = middle @ (np.linalg.inv(tau * cov) @ pi
                        + P.T @ np.linalg.inv(omega) @ Q)
    print("  View: AAPL beats MSFT by 2%/yr (confidence tau 0.05)")
    print("  Posterior returns: " + ", ".join(
        f"{n} {r:.1%}" for n, r in zip(names, post_mu)))
    print(f"  Reading: AAPL's posterior "
          f"({post_mu[i_aapl]:.1%}) is "
          f"{'above its equilibrium — the view tilted it up, as it should' if post_mu[i_aapl] > pi[i_aapl] else 'NOT above equilibrium — check the view arithmetic'}; "
          f"MSFT's ({post_mu[i_msft]:.1%}) is "
          f"{'below its equilibrium — the view tilted it down' if post_mu[i_msft] < pi[i_msft] else 'NOT below equilibrium — check the view arithmetic'}")
    w_eq = _max_sharpe(pi, cov)
    w_bl = _max_sharpe(post_mu, cov)
    print("  Max-Sharpe weights on equilibrium returns: " + ", ".join(
        f"{n} {w:.0%}" for n, w in zip(names, w_eq)))
    print("  Max-Sharpe weights on posterior returns:   " + ", ".join(
        f"{n} {w:.0%}" for n, w in zip(names, w_bl)))
    tilt = float(np.max(np.abs(w_bl - w_eq)))
    print(f"  Largest weight change {tilt:.0%} — "
          f"{'a modest tilt: the view moved the portfolio without hijacking it (this is the point of Black-Litterman)' if tilt < 0.25 else 'a large tilt: the view dominates — lower tau (confidence) if you want the market to have more say'}")
    return {"bl_pi_aapl": float(pi[i_aapl]),
            "bl_post_aapl": float(post_mu[i_aapl]),
            "bl_post_msft": float(post_mu[i_msft]),
            "bl_max_tilt": tilt,
            "bl_weights": {n: float(w) for n, w in zip(names, w_bl)}}
