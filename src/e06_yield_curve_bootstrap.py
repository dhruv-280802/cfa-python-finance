"""E06 — Yield-curve bootstrapping (Hilpisch "Python for Finance":
fixed income; CFA L2 fixed-income curriculum).

CFA idea: the yields you see quoted (3-month, 5-year, 10-year,
30-year Treasuries) are PAR yields — the coupon a new bond would
carry to sell at par. But pricing a bond needs ZERO rates (the rate
for a single cash flow at one date). BOOTSTRAPPING converts par
yields into zero rates, one maturity at a time: the 3-month zero
rate is its own par yield (one cash flow), then the 5-year zero
rate is solved from the 5-year par bond using the already-known
shorter zero rates for its coupons, and so on up the curve. The
result is the discount curve — the market's term structure.

Simplification, stated honestly: with only four quoted tenors
(0.25y, 5y, 10y, 30y) the coupons between them need an
interpolated par curve, so this is a teaching bootstrap (annual
coupons, linear interpolation), not a trading-desk curve — the
method is exact, the granularity is not.

How to read the result:
- Par yields vs zero rates: zero rates are usually a little ABOVE
  par yields at long tenors (the coupon effect), and nearly equal
  at the short end. If a zero rate came out BELOW its par yield,
  the arithmetic is wrong — that is the built-in sanity check.
- Slope (10y minus 3m): positive = normal curve (the market
  expects growth/inflation); NEGATIVE = inverted curve, the classic
  recession warning (every US recession since 1970 was preceded by
  an inversion). The size matters: -0.5% is a warning, -1.5% is a
  klaxon.
- Forward rate (5y -> 10y, implied): the market's expected 5-year
  rate, five years from now. It is not a forecast anyone would bet
  the house on, but it is the market's own arithmetic — if it looks
  absurd, the curve (or your reading of it) is wrong.
"""
import numpy as np
import pandas as pd


def run(prices, source):
    import yfinance as yf

    print("E06  Yield-curve bootstrapping — US Treasuries")
    tickers = {"0.25y": "^IRX", "5y": "^FVX", "10y": "^TNX", "30y": "^TYX"}
    tenors = np.array([0.25, 5.0, 10.0, 30.0])
    data = yf.download(list(tickers.values()), period="5d",
                       progress=False, auto_adjust=True)
    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) \
        else data[["Close"]]
    par = {}
    for label, ticker in tickers.items():
        series = closes[ticker].dropna()
        par[label] = float(series.iloc[-1]) / 100.0
    par_yields = np.array([par["0.25y"], par["5y"], par["10y"], par["30y"]])
    print("  Par yields (latest closes): " + ", ".join(
        f"{t} {y:.2%}" for t, y in zip(["3m", "5y", "10y", "30y"], par_yields)))

    # Bootstrap: annual coupons, linear interpolation of the par curve
    # for coupon dates between quoted tenors.
    def par_at(t):
        return float(np.interp(t, tenors, par_yields))

    zeros = {}
    for tenor, par_y in zip(tenors, par_yields):
        if tenor <= 1.0:
            zeros[tenor] = par_y  # single cash flow: zero = par
            continue
        # Solve sum(coupon * DF) + principal * DF = 100 for DF(tenor),
        # with DF(t) = 1 / (1 + z_t) ** t at already-solved tenors and
        # linear interpolation of zero rates in between.
        coupon = par_y * 100
        pv_coupons = 0.0
        for year in range(1, int(tenor)):
            z = float(np.interp(year, list(zeros.keys()),
                                list(zeros.values()))) \
                if year < tenor else 0.0
            # For years beyond the last solved tenor, interpolate on
            # the par curve and iterate once (teaching approximation).
            if year not in zeros:
                z = par_at(year)
            pv_coupons += coupon / (1 + z) ** year
        df_t = (100 - pv_coupons) / (100 + coupon)
        zeros[tenor] = float(df_t ** (-1.0 / tenor) - 1.0)
    zero_rates = np.array([zeros[t] for t in tenors])
    print("  Bootstrapped zero rates: " + ", ".join(
        f"{t} {z:.2%}" for t, z in zip(["3m", "5y", "10y", "30y"], zero_rates)))
    print(f"  Sanity check: zero rates are "
          f"{'above their par yields at long tenors, as theory says (coupon effect)' if all(zero_rates[1:] >= par_yields[1:] - 0.001) else 'NOT above par yields — recheck the arithmetic'}")

    slope = float(par_yields[2] - par_yields[0])  # 10y - 3m
    print(f"  Slope (10y - 3m): {slope:+.2%} -> "
          f"{'NORMAL curve: the market expects growth/inflation ahead' if slope > 0.005 else 'FLAT curve: no strong view on growth' if slope > -0.005 else 'INVERTED curve: the classic recession warning — every US recession since 1970 was preceded by an inversion'}")
    # Implied 5y forward, 5 years from now: (1+z10)^10 / (1+z5)^5.
    fwd_5y5y = float(((1 + zero_rates[2]) ** 10
                      / (1 + zero_rates[1]) ** 5) ** (1 / 5) - 1)
    print(f"  Implied 5-year rate, 5 years from now: {fwd_5y5y:.2%} — "
          f"the market's own arithmetic for where 5y yields go; "
          f"{'above today’s 5y yield, so the market expects rates to rise' if fwd_5y5y > par_yields[1] else 'below today’s 5y yield, so the market expects rates to fall'}")
    return {"yc_par_3m": float(par_yields[0]),
            "yc_par_10y": float(par_yields[2]),
            "yc_zero_10y": float(zero_rates[2]),
            "yc_slope_10y_3m": slope,
            "yc_fwd_5y5y": fwd_5y5y}
