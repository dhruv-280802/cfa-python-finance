"""D02 — Financial statements to features (CFA L2 PSM Unit 2:
"Data Collection, Wrangling and Feature Engineering in Machine
Learning").

CFA idea: a financial statement is a feature store. Revenue growth,
margins, return on equity and free cash flow are the raw materials
every quantamental model eats — but Yahoo's tables arrive wide
(line items as rows, fiscal years as columns), with inconsistent
row names, so the work is (1) fetch, (2) transpose, (3) engineer.

Worked example: AAPL + MSFT annual statements (last 5 fiscal years),
engineered into one tidy features table — revenue growth, gross /
operating / net margin, ROE, free cash flow and FCF margin, plus a
lagged growth feature and a "margin expanding" flag — then merged
with year-end closing prices from the project's price data. The
table is saved to results/d02_financial_features.csv.
"""
import numpy as np
import pandas as pd

from common import RESULTS_DIR  # noqa: F401  (kept for symmetry with run.py)

# Yahoo row names drift between tickers/years; try each candidate in
# order and take the first that exists (case-insensitive, substring).
ROW_CANDIDATES = {
    "revenue": ["Total Revenue", "Revenue"],
    "gross_profit": ["Gross Profit"],
    "op_income": ["Operating Income", "Total Operating Income"],
    "net_income": ["Net Income", "Net Income From Continuing Operation"],
    "total_assets": ["Total Assets"],
    "equity": ["Stockholders Equity", "Total Stockholder Equity",
               "Total Equity"],
    "op_cashflow": ["Operating Cash Flow", "Total Cash From Operations",
                    "Cash Flow From Operations"],
    "capex": ["Capital Expenditure", "Capital Expenditures"],
}


def _find_row(frame, candidates):
    """First row whose index matches any candidate (case-insensitive)."""
    lower = {str(i).lower(): i for i in frame.index}
    for cand in candidates:
        key = cand.lower()
        if key in lower:
            return frame.loc[lower[key]]
        for idx_lower, idx in lower.items():
            if key in idx_lower:
                return frame.loc[idx]
    return None


def _statements(ticker):
    """Annual income statement / balance sheet / cash flow as Series
    per line item, indexed by fiscal year. Returns None if Yahoo
    refuses the ticker (the caller decides what to do)."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    fin, bal, cfs = t.financials, t.balance_sheet, t.cashflow
    if fin is None or fin.empty:
        return None
    years = [int(c.year) for c in fin.columns]

    def series(frame, key):
        if frame is None or frame.empty:
            return pd.Series(np.nan, index=years, dtype=float)
        row = _find_row(frame, ROW_CANDIDATES[key])
        if row is None:
            return pd.Series(np.nan, index=years, dtype=float)
        vals = pd.to_numeric(row, errors="coerce")
        vals.index = [int(c.year) for c in row.index]
        return vals.reindex(years).astype(float)

    out = pd.DataFrame({
        "revenue": series(fin, "revenue"),
        "gross_profit": series(fin, "gross_profit"),
        "op_income": series(fin, "op_income"),
        "net_income": series(fin, "net_income"),
        "total_assets": series(bal, "total_assets"),
        "equity": series(bal, "equity"),
        "op_cashflow": series(cfs, "op_cashflow"),
        "capex": series(cfs, "capex"),
    }, index=years)
    out.index.name = "fiscal_year"
    # Yahoo returns columns newest-first: sort ascending so growth,
    # lags and "latest year" all run in calendar order. Drop years
    # where revenue itself is missing (a genuine Yahoo data gap —
    # AAPL FY2022 has no revenue row); keeping them would poison
    # every growth and margin calculation with NaNs.
    out = out.sort_index()
    n_dropped = int(out["revenue"].isna().sum())
    if n_dropped:
        print(f"  Note: {n_dropped} fiscal year(s) dropped — revenue "
              f"missing in Yahoo's table (a data gap, not a zero)")
    return out[out["revenue"].notna()]


def engineer(raw):
    """Turn raw statement lines into model-ready features."""
    f = raw.copy()
    f["revenue_growth"] = f["revenue"].pct_change(fill_method=None)
    f["revenue_growth_lag1"] = f["revenue_growth"].shift(1)
    f["gross_margin"] = f["gross_profit"] / f["revenue"]
    f["op_margin"] = f["op_income"] / f["revenue"]
    f["net_margin"] = f["net_income"] / f["revenue"]
    f["roe"] = f["net_income"] / f["equity"]
    # Yahoo reports capex as a negative outflow; FCF = op CF + capex.
    f["fcf"] = f["op_cashflow"] + f["capex"]
    f["fcf_margin"] = f["fcf"] / f["revenue"]
    f["margin_expanding"] = (f["op_margin"].diff() > 0).astype(float)
    return f


def run(prices, source):
    print("D02  Financial statements -> features (AAPL + MSFT, annual)")
    tables = []
    for ticker in ["AAPL", "MSFT"]:
        raw = _statements(ticker)
        if raw is None:
            print(f"  {ticker}: statements unavailable from Yahoo today")
            continue
        feats = engineer(raw)
        feats["ticker"] = ticker
        # Merge with the market: closing price at each fiscal year-end
        # (fiscal year ends ~Sep for both; take the last close in Q4).
        closes = prices[ticker]
        year_end = {}
        for yr in feats.index:
            q4 = closes[(closes.index.year == yr) & (closes.index.month >= 10)]
            year_end[yr] = float(q4.iloc[-1]) if len(q4) else np.nan
        feats["year_end_price"] = pd.Series(year_end)
        feats["market_cap_proxy"] = feats["year_end_price"] * np.nan  # see note
        tables.append(feats.reset_index())
        latest = feats.iloc[-1]
        print(f"  {ticker} FY{feats.index[-1]}: revenue "
              f"${latest['revenue'] / 1e9:.1f}bn "
              f"(growth {latest['revenue_growth']:+.1%}), gross margin "
              f"{latest['gross_margin']:.1%}, ROE {latest['roe']:.1%}, "
              f"FCF ${latest['fcf'] / 1e9:.1f}bn, year-end price "
              f"${latest['year_end_price']:.2f}")
    if not tables:
        print("  No statements fetched — features table not built")
        return {"features_rows": 0}
    features = pd.concat(tables, ignore_index=True)
    print(f"  Features table: {features.shape[0]} rows x "
          f"{features.shape[1]} cols; columns: "
          f"{', '.join(features.columns)}")
    print("  How to read these features: revenue growth is the top "
          "line — a company growing revenue 5%/yr is a different "
          "animal from one shrinking 5%/yr; gross margin is pricing "
          "power (what is left after making the product); ROE is how "
          "hard equity works for shareholders (20%+ is strong, under "
          "10% is weak); FCF margin is the cash the business actually "
          "keeps — the number a DCF (B07) discounts. A feature with "
          "no variation across years carries no signal for a model")
    print("  Note: market_cap_proxy is left NaN on purpose — share "
          "count is not in the statements tables, so a real market cap "
          "needs Ticker.info (see D01); never invent a share count")
    return {"features_rows": int(features.shape[0]),
            "features_table": features}
