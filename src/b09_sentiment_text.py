"""B09 — Text and sentiment analysis (CFA L2 PSM: "sentiment analysis
using AWS Comprehend" and EDGAR data ingestion).

CFA idea: prices move on words as well as numbers — an earnings call
transcript, a headline, a filing. Sentiment analysis turns text into a
number (positive minus negative language) that can sit next to returns
in a model. The industrial version uses AWS Comprehend or a fine-tuned
model on EDGAR filings; the technique below is the same idea with a
transparent word list, so every score can be explained word by word.

Worked example: score a set of ILLUSTRATIVE finance headlines (written
for teaching, not real news) with a small finance lexicon, then check
whether the scores line up with the direction of the market on the
days the headlines are dated to. The headlines are labelled
illustrative; the scoring method is the real lesson.
"""
import re

import numpy as np
import pandas as pd

POSITIVE = {"beat", "beats", "record", "growth", "strong", "surge",
            "rally", "upgrade", "upgrades", "profit", "profits",
            "bullish", "gains", "gain", "soar", "soars", "robust",
            "outperform", "raises", "raise", "jump", "jumps"}
NEGATIVE = {"miss", "misses", "weak", "slump", "slumps", "fall", "falls",
            "downgrade", "downgrades", "loss", "losses", "bearish",
            "plunge", "plunges", "cut", "cuts", "warning", "warns",
            "underperform", "drop", "drops", "fear", "fears"}

# Illustrative headlines written for teaching (NOT real news). Each is
# dated to a real trading day so its score can be compared with the
# market's actual move that day.
HEADLINES = [
    ("2022-01-27", "Apple beats earnings expectations with record iPhone sales"),
    ("2022-03-08", "Stocks plunge as oil prices surge and growth fears spread"),
    ("2022-06-13", "Inflation warning sends markets into a sharp fall"),
    ("2022-10-28", "Tech stocks rally after strong earnings beat estimates"),
    ("2023-02-02", "Apple reports robust growth and raises its dividend"),
    ("2023-03-10", "Bank stocks slump after SVB collapse sparks fear"),
    ("2023-11-02", "Fed signals pause and stocks jump on rate hopes"),
    ("2024-05-02", "Apple announces record buyback and profits beat forecasts"),
    ("2024-08-05", "Global stocks plunge on recession fears and weak jobs data"),
    ("2024-11-06", "Stocks surge as election result lifts markets"),
    ("2025-04-03", "Tariff warning hits stocks as growth outlook cuts deepen"),
    ("2025-08-01", "Strong jobs report lifts stocks to record highs"),
]


def sentiment_score(text):
    """Lexicon score: (positive words - negative words) / total words.
    Ranges from -1 (all negative) to +1 (all positive)."""
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE)
    neg = sum(1 for w in words if w in NEGATIVE)
    return (pos - neg) / len(words)


def run(prices, source):
    spy_ret = prices["SPY"].pct_change()
    rows = []
    for date_str, headline in HEADLINES:
        date = pd.Timestamp(date_str)
        score = sentiment_score(headline)
        # Market move on the headline's date (next trading day if a
        # holiday/weekend — the headline is illustrative, the return is real).
        idx = spy_ret.index.searchsorted(date)
        actual = float(spy_ret.iloc[idx]) if idx < len(spy_ret) else np.nan
        rows.append({"date": date_str, "headline": headline,
                     "score": score, "spy_return": actual})
    table = pd.DataFrame(rows)
    agree = int(((table["score"] > 0) & (table["spy_return"] > 0) |
                 (table["score"] < 0) & (table["spy_return"] < 0)).sum())
    corr = float(np.corrcoef(table["score"], table["spy_return"])[0, 1])

    print("B09  Text & sentiment analysis (illustrative headlines)")
    for _, r in table.iterrows():
        print(f"  {r['date']}  score {r['score']:+.2f}  "
              f"SPY {r['spy_return']:+.2%}  {r['headline'][:60]}")
    print(f"  Headline sign agreed with the market's sign on {agree} of "
          f"{len(table)} days; score/return correlation {corr:+.2f}")
    print("  Honest note: 12 hand-written headlines cannot prove a "
          "signal — the industrial version scores thousands of real "
          "filings/headlines (EDGAR + AWS Comprehend) and tests the "
          "correlation properly")
    return {"sentiment_agree": agree, "sentiment_n": len(table),
            "sentiment_corr": corr}
