"""B06 — Monte Carlo simulation (CFA L1 PSM and L2 PSM: "run Monte Carlo
simulations").

CFA idea: instead of one forecast, simulate thousands of possible
futures and look at the distribution. A Monte Carlo answers questions
a single point estimate cannot: "how bad could the bad case be?" and
"what is the chance I lose money over the next year?".

Worked example: a $100,000 equal-weight portfolio of SPY/AAPL/MSFT,
simulated 10,000 times over 1 year with geometric Brownian motion.
Drift and volatility come from the real 5-year history (not assumed);
each path is one possible year. Report the median outcome, the 5th and
95th percentiles, and the probability of a loss.
"""
import numpy as np
import pandas as pd


def simulate_gbm_paths(start_prices, mu_annual, sigma_annual, corr,
                       n_paths=10_000, n_days=252, seed=42):
    """Simulate correlated GBM price paths.

    start_prices: array of current prices; mu_annual / sigma_annual:
    per-asset annual drift and vol; corr: correlation matrix. Returns
    an array of shape (n_paths, n_days + 1, n_assets) of prices.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(start_prices)
    dt = 1 / 252
    chol = np.linalg.cholesky(corr)
    shocks = rng.standard_normal((n_paths, n_days, n_assets))
    correlated = shocks @ chol.T
    drift = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    diffusion = sigma_annual * np.sqrt(dt)
    increments = drift + diffusion * correlated
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1, n_assets)), log_paths], axis=1)
    return start_prices * np.exp(log_paths)


def run(prices, source):
    tickers = ["SPY", "AAPL", "MSFT"]
    rets = prices[tickers].pct_change().dropna()
    mu = rets.mean().to_numpy() * 252
    sigma = rets.std().to_numpy() * np.sqrt(252)
    corr = rets.corr().to_numpy()
    start = prices[tickers].iloc[-1].to_numpy()
    paths = simulate_gbm_paths(start, mu, sigma, corr)

    # Equal-weight portfolio: $100,000 split three ways.
    weights = np.repeat(1 / 3, 3)
    shares = 100_000 * weights / start
    terminal = paths[:, -1, :] @ shares
    median = float(np.median(terminal))
    p5 = float(np.percentile(terminal, 5))
    p95 = float(np.percentile(terminal, 95))
    p_loss = float((terminal < 100_000).mean())
    p_gain20 = float((terminal > 120_000).mean())
    expected = float(terminal.mean())

    # Same simulation for SPY alone, for comparison.
    spy_terminal = paths[:, -1, 0] / start[0] * 100_000
    spy_p5 = float(np.percentile(spy_terminal, 5))
    spy_p_loss = float((spy_terminal < 100_000).mean())

    print("B06  Monte Carlo simulation (10,000 paths, 1 year)")
    print(f"  Equal-weight SPY/AAPL/MSFT, $100,000:")
    print(f"    Expected terminal value: ${expected:,.0f}")
    print(f"    Median: ${median:,.0f}; 5th pct ${p5:,.0f}; "
          f"95th pct ${p95:,.0f}")
    print(f"    P(loss over 1 year): {p_loss:.1%}; "
          f"P(gain > 20%): {p_gain20:.1%}")
    print(f"  SPY alone: 5th pct ${spy_p5:,.0f}, P(loss) {spy_p_loss:.1%} "
          f"— the diversified portfolio's bad case is less bad")
    return {"mc_expected": expected, "mc_median": median,
            "mc_p5": p5, "mc_p95": p95, "mc_p_loss": p_loss,
            "mc_p_gain20": p_gain20, "spy_mc_p5": spy_p5,
            "spy_mc_p_loss": spy_p_loss}
