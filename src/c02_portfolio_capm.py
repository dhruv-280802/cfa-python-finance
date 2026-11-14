"""C02 — Portfolio theory and CAPM (CFA L1/L2 Portfolio Management).

CFA idea: the efficient frontier is the set of portfolios no rational
investor would refuse — for each level of risk, the highest return
available. CAPM then prices a single asset by its beta: expected return
= risk-free rate + beta * market risk premium. Alpha is what is left
after that — skill, or luck, or a mismeasured beta.

Worked examples: (1) 5,000 random portfolios of SPY/AAPL/MSFT/TLT —
the frontier's minimum-volatility and maximum-Sharpe portfolios;
(2) CAPM for MSFT vs SPY: beta, alpha (annualised), R-squared;
(3) the Security Market Line check: is MSFT's realised return above
or below what CAPM says it should have earned?
"""
import numpy as np
import pandas as pd


def random_portfolios(mu, sigma, corr, n=5_000, seed=42):
    """Random long-only portfolios (Dirichlet weights): returns the
    weights, annualised returns, vols and Sharpe ratios."""
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    weights = rng.dirichlet(np.ones(n_assets), size=n)
    cov = np.outer(sigma, sigma) * corr
    port_mu = weights @ mu
    port_vol = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    return weights, port_mu, port_vol


def capm(rets_asset, rets_market, risk_free_annual=0.04):
    """CAPM regression: r_asset - rf = alpha + beta * (r_market - rf).
    Returns (alpha annualised, beta, R-squared)."""
    rf = risk_free_annual / 252
    y = rets_asset - rf
    x = rets_market - rf
    X = np.column_stack([np.ones(len(x)), x])
    beta_vec, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily, beta = beta_vec[0], beta_vec[1]
    fitted = X @ beta_vec
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    return float(alpha_daily * 252), float(beta), float(r2)


def run(prices, source):
    rets = prices.pct_change().dropna()
    mu = rets.mean().to_numpy() * 252
    sigma = rets.std().to_numpy() * np.sqrt(252)
    corr = rets.corr().to_numpy()
    tickers = list(rets.columns)
    weights, port_mu, port_vol = random_portfolios(mu, sigma, corr)
    sharpe = (port_mu - 0.04) / port_vol
    i_minvol = int(np.argmin(port_vol))
    i_maxsharpe = int(np.argmax(sharpe))
    eq_w = np.repeat(1 / len(mu), len(mu))
    eq_vol = float(np.sqrt(eq_w @ (np.outer(sigma, sigma) * corr) @ eq_w))
    eq_mu = float(eq_w @ mu)

    alpha, beta, r2 = capm(rets["MSFT"], rets["SPY"])
    msft_realised = float(rets["MSFT"].mean() * 252)
    market_premium = float(rets["SPY"].mean() * 252 - 0.04)
    capm_expected = 0.04 + beta * market_premium

    print("C02  Portfolio theory & CAPM (5,000 random portfolios)")
    print(f"  Minimum-volatility portfolio: return {port_mu[i_minvol]:+.2%}, "
          f"vol {port_vol[i_minvol]:.2%}, weights "
          f"({', '.join(f'{t} {w:.0%}' for t, w in zip(tickers, weights[i_minvol]))})")
    print(f"  Maximum-Sharpe portfolio: return {port_mu[i_maxsharpe]:+.2%}, "
          f"vol {port_vol[i_maxsharpe]:.2%}, Sharpe {sharpe[i_maxsharpe]:.2f}, weights "
          f"({', '.join(f'{t} {w:.0%}' for t, w in zip(tickers, weights[i_maxsharpe]))})")
    print(f"  Equal-weight portfolio: return {eq_mu:+.2%}, vol {eq_vol:.2%}")
    print(f"  CAPM for MSFT vs SPY: beta {beta:.2f}, alpha {alpha:+.2%}/yr, "
          f"R2 {r2:.3f}")
    print(f"  MSFT realised return {msft_realised:+.2%} vs CAPM-implied "
          f"{capm_expected:+.2%} -> alpha {alpha:+.2%} means MSFT "
          f"{'beat' if alpha > 0 else 'lagged'} its risk-adjusted benchmark")
    frontier = pd.DataFrame({
        "return": port_mu, "vol": port_vol, "sharpe": sharpe})
    return {"min_vol": float(port_vol[i_minvol]),
            "min_vol_return": float(port_mu[i_minvol]),
            "max_sharpe": float(sharpe[i_maxsharpe]),
            "max_sharpe_return": float(port_mu[i_maxsharpe]),
            "eq_weight_vol": eq_vol, "eq_weight_return": eq_mu,
            "msft_beta": beta, "msft_alpha": alpha, "msft_r2": r2,
            "frontier": frontier}
