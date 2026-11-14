"""E07 — Pairs trading and statistical arbitrage (Hilpisch "Python
for Finance": algorithmic trading; the mean-reversion-pairs repo in
this bundle is the full version of this idea).

CFA idea: two stocks in the same business (Morgan Stanley and
Goldman Sachs — both big investment banks) should move together
over the long run. COINTEGRATION makes that precise: each price
wanders (non-stationary), but a linear combination of them — the
SPREAD — is stationary (mean-reverting). Trade the spread, not the
stocks: when it stretches far above its mean, short the spread
(short the outperformer, long the underperformer); when it snaps
back, collect the difference.

How to read the result:
- Engle-Granger test: H0 = "no cointegration" (the spread wanders
  forever). p < 0.05 rejects it — the spread IS mean-reverting and
  the pair is tradable. p = 0.4 = the two stocks just happen to
  move together sometimes; there is no trade here.
- Hedge ratio: how many shares of GS per share of MS keep the
  spread neutral (from regressing GS on MS). A ratio of 0.18 means
  100 shares of MS against 18 of GS. The ratio is an ESTIMATE —
  re-estimate it regularly, because it drifts.
- Half-life: how fast the spread reverts (from an AR(1) regression
  on the spread). 10-30 days is tradeable; 200+ days means the
  "mean reversion" is too slow to monetise; under 2 days means it
  reverts before you can trade it.
- Current z-score: (spread - mean) / std. |z| > 2 = stretched
  (the textbook entry signal); |z| < 0.5 = fairly priced (no
  trade). A z-score is a DESCRIPTION, not a recommendation — the
  spread can stay stretched (a real regime change breaks the
  relationship, and the "mean reversion" never comes).
- Correlation is NOT cointegration: two stocks can be 95%
  correlated and still drift apart forever. The test is on the
  spread's stationarity, not on the correlation.
"""
import os

import numpy as np
import pandas as pd

from common import DATA_DIR

CACHE = os.path.join(DATA_DIR, "ms_gs_prices.csv")


def _load_pair():
    """MS/GS daily closes, cached separately from the main price file."""
    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        if len(df) > 500 and {"MS", "GS"}.issubset(df.columns):
            return df
    import yfinance as yf

    raw = yf.download(["MS", "GS"], period="5y", interval="1d",
                      auto_adjust=True, progress=False)
    closes = raw["Close"]
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    closes = closes.dropna()
    closes.to_csv(CACHE)
    return closes


def run(prices, source):
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint

    print("E07  Pairs trading / statistical arbitrage — MS vs GS")
    pair = _load_pair()
    ms, gs = pair["MS"], pair["GS"]
    print(f"  {len(pair)} trading days "
          f"({pair.index[0].date()} to {pair.index[-1].date()})")
    corr = float(np.corrcoef(ms.pct_change().dropna(),
                             gs.pct_change().dropna())[0, 1])
    print(f"  Daily-return correlation {corr:.2f} — high, but "
          f"correlation is NOT cointegration (see below)")

    # Engle-Granger cointegration test on the price LEVELS.
    score, p_value, crit = coint(ms, gs)
    print(f"  Engle-Granger test stat {score:.2f}, p {p_value:.3g} "
          f"(5% critical value {crit[1]:.2f}) -> "
          f"{'COINTEGRATED: the spread is stationary and mean-reverting — the pair is tradable' if p_value < 0.05 else 'NOT cointegrated: the spread wanders — there is no reliable mean reversion to trade'}")
    X = sm.add_constant(ms)
    hedge = sm.OLS(gs, X).fit()
    ratio = float(hedge.params["MS"])
    spread = gs - ratio * ms
    z = (spread - spread.mean()) / spread.std()
    print(f"  Hedge ratio {ratio:.4f} (GS = {float(hedge.params['const']):+.2f} "
          f"+ {ratio:.4f} x MS): hold {ratio:.2f} GS shares per 1 MS "
          f"share for a neutral spread; R² {float(hedge.rsquared):.3f}")
    # Half-life from AR(1) on the spread: spread_t = a + rho*spread_{t-1}.
    lag = spread.shift(1).dropna()
    aligned = pd.DataFrame({"s": spread.loc[lag.index], "lag": lag})
    rho = float(sm.OLS(aligned["s"],
                       sm.add_constant(aligned["lag"])).fit().params["lag"])
    half_life = float(-np.log(2) / np.log(rho)) if 0 < rho < 1 else float("inf")
    print(f"  Spread AR(1) coefficient {rho:.3f} -> half-life "
          f"{half_life:.1f} days — "
          f"{'tradeable mean reversion (10-30 days is the sweet spot)' if 5 < half_life < 60 else 'too slow to monetise — the spread takes months to come back' if half_life >= 60 else 'too fast to trade, or the spread is not really mean-reverting'}")
    z_now = float(z.iloc[-1])
    print(f"  Current spread z-score {z_now:+.2f} -> "
          f"{'STRETCHED: the textbook mean-reversion entry zone (|z| > 2)' if abs(z_now) > 2 else 'fairly priced: no mean-reversion signal right now' if abs(z_now) < 0.5 else 'mildly stretched: watch, do not chase'}")
    print("  Warning: cointegration is a historical property, not a "
          "guarantee. A merger, a scandal or a business-model change "
          "breaks the relationship, and the spread never comes back — "
          "that is the risk that kills pairs traders")
    return {"pairs_coint_p": float(p_value),
            "pairs_hedge_ratio": ratio,
            "pairs_half_life_days": half_life,
            "pairs_z_now": z_now,
            "pairs_corr": corr}
