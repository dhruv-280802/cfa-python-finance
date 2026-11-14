"""B07 — Building a financial model end to end: a DCF for Apple
(CFA L2 PSM: "choosing valuation methodology, target prices" and the
Analyst Skills PSM's valuation work).

CFA idea: a discounted-cash-flow model values a company as the present
value of its future free cash flows plus a terminal value. Three inputs
drive almost everything: the starting cash flow, the growth path, and
the discount rate (WACC). Change any one and the answer moves — which
is why a model is never one number, it is a table of sensitivities.

Worked example: AAPL, valued from its REAL latest fiscal-year free cash
flow (Yahoo Finance cash-flow statement) and REAL share price and shares
outstanding. Growth, WACC and terminal growth are stated assumptions —
labelled as such, not presented as facts. Output: intrinsic value per
share vs the market price, and a WACC x growth sensitivity table.
"""
import numpy as np
import pandas as pd


def fetch_aapl_fundamentals():
    """Latest fiscal-year free cash flow, current price, shares
    outstanding for AAPL from Yahoo Finance. Returns (fcf, price,
    shares, fcf_source). Falls back to stated assumptions if the
    fetch fails — and says so."""
    try:
        import yfinance as yf

        t = yf.Ticker("AAPL")
        cf = t.cashflow
        fcf = float(cf.loc["Free Cash Flow"].iloc[0])  # latest FY, $.
        info = t.info
        price = float(info["currentPrice"])
        shares = float(info["sharesOutstanding"])
        return fcf, price, shares, "REAL (Yahoo Finance cash-flow statement)"
    except Exception as exc:  # noqa: BLE001 - any fetch failure -> fallback
        print(f"  Fundamentals fetch failed ({exc}); using stated "
              "assumption for base FCF.")
        return 100e9, 250.0, 15.0e9, "ASSUMED (fetch failed)"


def dcf_value(fcf0, growth_5y, growth_terminal, wacc, years=5):
    """Value a firm: project FCF growing at growth_5y for `years` years,
    discount at wacc, add a Gordon-growth terminal value, return the
    enterprise value in the same units as fcf0."""
    pv = 0.0
    fcf = fcf0
    for year in range(1, years + 1):
        fcf *= (1 + growth_5y)
        pv += fcf / (1 + wacc) ** year
    terminal = fcf * (1 + growth_terminal) / (wacc - growth_terminal)
    pv += terminal / (1 + wacc) ** years
    return pv


def run(prices, source):
    fcf0, price, shares, fcf_source = fetch_aapl_fundamentals()
    growth_5y, growth_t, wacc = 0.06, 0.025, 0.09  # stated assumptions
    ev = dcf_value(fcf0, growth_5y, growth_t, wacc)
    intrinsic = ev / shares
    upside = intrinsic / price - 1

    # Sensitivity: WACC 8-10% x 5y growth 4-8%.
    waccs = [0.08, 0.09, 0.10]
    growths = [0.04, 0.06, 0.08]
    table = pd.DataFrame(
        [[dcf_value(fcf0, g, growth_t, w) / shares
          for w in waccs] for g in growths],
        index=[f"g={g:.0%}" for g in growths],
        columns=[f"WACC={w:.0%}" for w in waccs])

    print("B07  Financial model — AAPL DCF")
    print(f"  Base FCF (latest fiscal year): ${fcf0 / 1e9:.1f}bn "
          f"[{fcf_source}]")
    print(f"  Assumptions (stated, not facts): 5y growth {growth_5y:.0%}, "
          f"terminal growth {growth_t:.1%}, WACC {wacc:.0%}")
    print(f"  Enterprise value: ${ev / 1e12:.2f}tn; shares: "
          f"{shares / 1e9:.2f}bn")
    print(f"  Intrinsic value: ${intrinsic:.2f} vs market ${price:.2f} "
          f"-> {upside:+.1%} implied upside")
    print("  Sensitivity — intrinsic value per share:")
    print(table.to_string(float_format=lambda v: f"${v:.2f}"))
    return {"aapl_fcf_bn": float(fcf0 / 1e9),
            "aapl_intrinsic": float(intrinsic),
            "aapl_market_price": float(price),
            "aapl_implied_upside": float(upside),
            "dcf_table": table}
