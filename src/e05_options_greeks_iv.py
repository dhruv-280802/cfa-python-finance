"""E05 — Options Greeks and implied volatility (Hilpisch
"Derivatives Analytics with Python": market-based valuation and
model calibration).

CFA idea: an option's price is not a guess — it is a function of
five inputs (spot, strike, time, rate, volatility) plus the market's
own view. The GREEKS are the sensitivities: how the price moves
when ONE input moves and the rest stay still. Traders do not trade
options, they trade the Greeks.
- Delta: price change for a $1 move in the underlying. A call delta
  of 0.54 means the option gains ~$0.54 when the stock rises $1.
  Delta is also (roughly) the risk-neutral probability of finishing
  in the money. |Delta| near 1 = deep in the money (behaves like
  the stock); near 0 = far out of the money (a lottery ticket).
- Gamma: how fast delta itself changes. High gamma = the hedge
  must be rebalanced constantly (gamma risk is largest at the
  money, near expiry). A long-gamma position profits from big
  moves in either direction.
- Theta: time decay per day. Options are wasting assets — theta is
  negative for a long call, and it accelerates into expiry. Buying
  an option means paying theta every day you hold it.
- Vega: price change for a 1-point (1%) rise in implied volatility.
  Long options love volatility; short options hate it. Vega is
  largest at the money.
- Rho: price change for a 1-point rise in rates. Small for
  short-dated options; matters for LEAPS.
- IMPLIED VOLATILITY (IV): the volatility that makes the
  Black-Scholes price EQUAL the market price — the market's forecast
  of future volatility, solved by Newton-Raphson iteration. Plot IV
  against strike and you get the VOLATILITY SMILE (or smirk): IV is
  usually higher for out-of-the-money puts, because the market
  charges more for crash insurance than the lognormal model assumes.

Worked example: real SPY options chain (nearest expiry) — compute
all five Greeks for the at-the-money call, then solve IV for every
strike near the money and print the smile.
"""
import math

import numpy as np
import pandas as pd


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_price(S, K, T, r, sigma, kind="call"):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def greeks(S, K, T, r, sigma, kind="call"):
    """All five Greeks for a European option (Black-Scholes)."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    sqrt_t = math.sqrt(T)
    if kind == "call":
        delta = _norm_cdf(d1)
        theta = (-(S * _norm_pdf(d1) * sigma) / (2 * sqrt_t)
                 - r * K * math.exp(-r * T) * _norm_cdf(d2))
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2)
    else:
        delta = _norm_cdf(d1) - 1
        theta = (-(S * _norm_pdf(d1) * sigma) / (2 * sqrt_t)
                 + r * K * math.exp(-r * T) * _norm_cdf(-d2))
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2)
    gamma = _norm_pdf(d1) / (S * sigma * sqrt_t)
    vega = S * _norm_pdf(d1) * sqrt_t  # per 1.00 (100%) vol move
    return {"delta": delta, "gamma": gamma, "theta": theta / 365.0,
            "vega": vega / 100.0, "rho": rho / 100.0}


def implied_vol(market_price, S, K, T, r, kind="call"):
    """Newton-Raphson: the sigma that reproduces the market price."""
    sigma = 0.20
    for _ in range(100):
        price = bs_price(S, K, T, r, sigma, kind)
        vega = S * _norm_pdf(
            (math.log(S / K) + (r + 0.5 * sigma ** 2) * T)
            / (sigma * math.sqrt(T))) * math.sqrt(T)
        diff = price - market_price
        if abs(diff) < 1e-6:
            return sigma
        if vega < 1e-8:
            return np.nan
        sigma -= diff / vega
        sigma = min(max(sigma, 0.01), 3.0)
    return np.nan


def run(prices, source):
    import datetime

    import yfinance as yf

    print("E05  Options Greeks + implied volatility — real SPY chain")
    # Spot: fetch a FRESH close. The project's cached prices can end
    # on an intraday print (the cache was written mid-session), and a
    # $2 error in the spot moves a 3-day ATM option's IV by several
    # points — so the spot for options work always comes from a fresh
    # fetch, never the cache.
    hist5 = yf.Ticker("SPY").history(period="5d", auto_adjust=True)
    spot = float(hist5["Close"].iloc[-1])
    spot_date = hist5.index[-1].date()
    ticker = yf.Ticker("SPY")
    expiry = ticker.options[0]
    chain = ticker.option_chain(expiry)
    # Time to expiry: calendar days from TODAY (not from the last
    # price date) — the chain is priced as of today.
    T = max((np.datetime64(expiry) - np.datetime64(
        datetime.date.today())).astype("timedelta64[D]").astype(int)
        / 365.0, 1 / 365.0)
    r = 0.04  # assumption, stated (matches the 4% risk-free used elsewhere)
    print(f"  SPY spot ${spot:.2f} (close {spot_date}); nearest expiry "
          f"{expiry} ({T * 365:.0f} days); risk-free {r:.0%} (assumption)")
    calls = chain.calls
    atm = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]].iloc[0]
    K = float(atm["strike"])
    mid = (float(atm["bid"]) + float(atm["ask"])) / 2
    iv_atm = implied_vol(mid, spot, K, T, r, "call")
    iv_yahoo = float(atm["impliedVolatility"]) \
        if pd.notna(atm["impliedVolatility"]) else np.nan
    g = greeks(spot, K, T, r, iv_atm, "call")
    print(f"  ATM call (strike ${K:.0f}, mid ${mid:.2f}, IV {iv_atm:.1%}):")
    print(f"    Cross-check: Yahoo's own IV for this contract is "
          f"{iv_yahoo:.1%} — our solver agrees within "
          f"{abs(iv_atm - iv_yahoo) * 100:.1f} points, so the "
          f"Newton-Raphson inversion is working (a big gap would mean "
          f"a wrong spot, a wrong T, or a stale quote)")
    print(f"    Delta {g['delta']:+.2f} — the option gains ~${abs(g['delta']):.2f} "
          f"if SPY rises $1; also the rough probability of finishing "
          f"in the money ({g['delta']:.0%})")
    print(f"    Gamma {g['gamma']:.4f} — delta changes by {g['gamma']:.4f} "
          f"per $1 move in SPY; highest at the money, so the hedge "
          f"needs the most rebalancing here")
    print(f"    Theta {g['theta']:+.3f}/day — the option LOSES "
          f"${abs(g['theta']):.2f} of value every day just from time "
          f"passing (time decay accelerates into expiry)")
    print(f"    Vega {g['vega']:+.3f} per 1pt vol — a 1-point rise in "
          f"implied vol adds ${g['vega']:.2f}; long options LOVE "
          f"volatility, short options hate it")
    print(f"    Rho {g['rho']:+.3f} per 1pt rates — small for a "
          f"{T * 365:.0f}-day option; it only matters for LEAPS")
    # IV smile: strikes within +/-5% of spot, calls with a real quote.
    rows = []
    for _, row in calls.iterrows():
        if abs(row["strike"] / spot - 1) > 0.05 or row["bid"] <= 0:
            continue
        mid_p = (float(row["bid"]) + float(row["ask"])) / 2
        iv = implied_vol(mid_p, spot, float(row["strike"]), T, r, "call")
        if iv == iv:  # not NaN
            rows.append((float(row["strike"]), iv))
    rows.sort()
    print("  IV smile (strike -> implied vol):")
    for strike, iv in rows:
        moneyness = "ATM" if abs(strike / spot - 1) < 0.01 else \
            ("OTM" if strike > spot else "ITM")
        print(f"    ${strike:.0f} ({moneyness}): IV {iv:.1%}")
    ivs = [iv for _, iv in rows]
    print(f"  Reading: IV ranges {min(ivs):.1%} to {max(ivs):.1%} — "
          f"{'a visible smile: the market charges more for far strikes than Black-Scholes assumes, i.e. it expects fatter tails than the lognormal model' if max(ivs) - min(ivs) > 0.03 else 'a fairly flat smile: the market roughly agrees with the lognormal model at this expiry'}; "
          f"IV {iv_atm:.1%} vs SPY realised vol ~{float(prices['SPY'].pct_change().tail(30).std() * np.sqrt(252)):.1%} — "
          f"{'IV above realised vol: options are expensive vs history (typical — sellers demand a premium)' if iv_atm > float(prices['SPY'].pct_change().tail(30).std() * np.sqrt(252)) else 'IV below recent realised vol: options are cheap vs history'}")
    return {"iv_atm": float(iv_atm), "greek_delta": float(g["delta"]),
            "greek_gamma": float(g["gamma"]),
            "greek_theta_day": float(g["theta"]),
            "greek_vega": float(g["vega"]),
            "iv_smile_range": float(max(ivs) - min(ivs))}
