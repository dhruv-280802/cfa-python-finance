"""A11 — Classes and objects: modelling an asset and a portfolio
(CFA L1 PSM: organising code so it can grow).

CFA idea: a class bundles data with the operations on that data. An
Asset knows its prices and can report its own return and volatility; a
Portfolio holds Assets with weights and can report its own value and
risk. Once the logic lives in classes, run.py stays short and every
later module can reuse them instead of copying formulas.

Worked example: build an Asset for each ticker from the real price
table, combine them into an equal-weight Portfolio, and ask the
portfolio for its value path, return, volatility and Sharpe.
"""
import numpy as np
import pandas as pd


class Asset:
    """One tradeable asset: a ticker plus its daily closing prices."""

    def __init__(self, ticker, prices):
        self.ticker = ticker
        self.prices = pd.Series(np.asarray(prices, dtype=float),
                                index=prices.index)
        if (self.prices <= 0).any():
            raise ValueError(f"Asset {ticker}: prices must be positive")

    def __repr__(self):
        return (f"Asset({self.ticker}: {len(self.prices)} closes, "
                f"${self.prices.iloc[0]:.2f} -> "
                f"${self.prices.iloc[-1]:.2f})")

    def returns(self):
        """Daily simple returns."""
        return self.prices.pct_change().dropna()

    def total_return(self):
        return float(self.prices.iloc[-1] / self.prices.iloc[0] - 1)

    def annualised_vol(self):
        return float(self.returns().std() * np.sqrt(252))

    def annualised_mean(self):
        return float(self.returns().mean() * 252)


class Portfolio:
    """A collection of Assets with fixed weights (buy-and-hold)."""

    def __init__(self, holdings):
        """holdings: dict of {Asset: weight}; weights must sum to 1."""
        total = sum(holdings.values())
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Portfolio weights sum to {total}, not 1")
        self.holdings = dict(holdings)

    def __repr__(self):
        parts = ", ".join(f"{a.ticker} {w:.0%}"
                          for a, w in self.holdings.items())
        return f"Portfolio({parts})"

    def value_path(self, initial=100_000.0):
        """Portfolio value over time, $1 = `initial` at the start."""
        rets = pd.concat(
            [a.returns() * w for a, w in self.holdings.items()],
            axis=1).sum(axis=1)
        return initial * (1 + rets).cumprod()

    def total_return(self):
        path = self.value_path()
        return float(path.iloc[-1] / path.iloc[0] - 1)

    def annualised_vol(self):
        rets = pd.concat(
            [a.returns() * w for a, w in self.holdings.items()],
            axis=1).sum(axis=1)
        return float(rets.std() * np.sqrt(252))

    def sharpe(self, risk_free=0.04):
        rets = pd.concat(
            [a.returns() * w for a, w in self.holdings.items()],
            axis=1).sum(axis=1)
        excess = rets - risk_free / 252
        return float(excess.mean() / excess.std() * np.sqrt(252))


def run(prices, source):
    assets = {t: Asset(t, prices[t]) for t in prices.columns}
    print("A11  Classes & objects")
    for a in assets.values():
        print(f"  {a} | total return {a.total_return():+.1%}, "
              f"vol {a.annualised_vol():.1%}")
    n = len(assets)
    portfolio = Portfolio({a: 1 / n for a in assets.values()})
    path = portfolio.value_path()
    print(f"  {portfolio}")
    print(f"  $100,000 equal-weight -> ${path.iloc[-1]:,.2f} "
          f"({portfolio.total_return():+.1%} total)")
    print(f"  Portfolio vol {portfolio.annualised_vol():.1%}, "
          f"Sharpe (Rf 4%) {portfolio.sharpe():.2f}")
    return {"portfolio_final_value": float(path.iloc[-1]),
            "portfolio_total_return": portfolio.total_return(),
            "portfolio_vol": portfolio.annualised_vol(),
            "portfolio_sharpe": portfolio.sharpe()}
