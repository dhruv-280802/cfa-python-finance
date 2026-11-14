"""D09 — Text cleaning, tokenisation and word clouds (CFA L2 PSM
Unit 4: "Natural Language Processing (NLP) in Python").

CFA idea: a model cannot read words — it reads NUMBERS. The pipeline
that turns text into numbers has four steps, and every one of them
is a choice that changes the result:
1. CLEANING: lowercase everything ("Apple" and "apple" are the same
   word), strip punctuation and numbers, drop stopwords ("the", "and",
   "of") — words so common they carry no signal. Over-clean and you
   delete meaning ("not good" without "not" is "good"); under-clean
   and the vocabulary explodes.
2. TOKENISATION: split text into tokens (here, words). A token is
   the unit the model sees.
3. ENCODING: map each distinct word to an integer id. The vocabulary
   is the dictionary; rare words get lumped into UNK (unknown) so
   the model never sees a word it has no vector for.
4. PADDING: neural networks need fixed-length inputs, so short
   sequences are padded with 0s (PAD) to the length of the longest
   one. The model must learn to ignore the padding — that is why
   PAD gets its own id and (in D10) a zero vector.

The corpus below is 40 short ILLUSTRATIVE finance sentences written
for teaching (20 positive, 20 negative) — the same sentences D10
and D11 train on, so the three modules are directly comparable.
Real work would use thousands of real headlines; the pipeline is
identical, only the corpus grows.

Worked example: clean, tokenise, encode and pad the corpus, then
draw a word cloud — the bigger the word, the more often it appears.
"""
import os
import re

import numpy as np

from common import RESULTS_DIR

# Illustrative teaching sentences (NOT real news), 20 positive / 20 negative.
SENTENCES = [
    ("Apple beats earnings expectations with record iPhone sales and strong guidance", 1),
    ("Stocks rally as inflation cools and the Fed signals rate cuts ahead", 1),
    ("Tech shares surge after robust cloud growth lifts profits", 1),
    ("Bank earnings beat forecasts as lending margins expand", 1),
    ("Oil prices climb on strong demand and tight supply", 1),
    ("Company raises dividend after record free cash flow", 1),
    ("Shares jump as new product launch exceeds sales targets", 1),
    ("Economy adds jobs and consumer spending stays resilient", 1),
    ("Chipmaker posts record revenue driven by AI demand", 1),
    ("Merger deal lifts shares as synergies promise higher margins", 1),
    ("Retail sales beat estimates and outlook improves", 1),
    ("Central bank pauses hikes boosting market sentiment", 1),
    ("Energy stocks gain as crude rallies to new highs", 1),
    ("Analyst upgrades stock citing strong order backlog", 1),
    ("Company buys back shares after profit surge", 1),
    ("Housing starts rise and mortgage rates ease", 1),
    ("Export growth accelerates lifting industrial stocks", 1),
    ("Goldman profits beat on strong trading revenue", 1),
    ("Airline bookings surge as travel demand recovers", 1),
    ("Software firm raises outlook after subscription growth", 1),
    ("Stocks plunge as recession fears spread and growth slows", 0),
    ("Company misses earnings and cuts full year guidance", 0),
    ("Bank shares slump after loan losses mount", 0),
    ("Inflation warning sends markets into a sharp fall", 0),
    ("Tech stocks drop as chip demand weakens", 0),
    ("Oil prices tumble on oversupply concerns", 0),
    ("Retailer warns of weak holiday sales and falling margins", 0),
    ("Central bank hikes rates more than expected rattling markets", 0),
    ("Company suspends dividend after cash flow deteriorates", 0),
    ("Shares fall as regulator opens investigation into accounting", 0),
    ("Unemployment rises and consumer confidence drops", 0),
    ("Supply chain disruptions hit production and profits", 0),
    ("Currency crisis deepens as capital flees emerging markets", 0),
    ("Analyst downgrades stock citing shrinking market share", 0),
    ("Housing market cools as mortgage rates climb", 0),
    ("Airline cancels flights as fuel costs soar", 0),
    ("Company recalls product after safety concerns emerge", 0),
    ("Bond yields spike as debt fears grow", 0),
    ("Manufacturing contracts for a third straight month", 0),
    ("Shares plunge after fraud allegations surface", 0),
]

STOPWORDS = {
    "a", "an", "the", "and", "or", "as", "at", "by", "for", "in", "of",
    "on", "to", "with", "after", "as", "is", "are", "was", "were",
    "it", "its", "this", "that", "from", "has", "have", "had", "be",
    "been", "new", "more", "than", "up", "down",
}


def clean_text(text):
    """Lowercase, keep letters only, drop stopwords. Returns tokens."""
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def build_vocab(texts, max_vocab=120):
    """Most-frequent words first; id 0 = PAD, 1 = UNK (unknown word)."""
    from collections import Counter

    counts = Counter()
    for text in texts:
        counts.update(clean_text(text))
    words = [w for w, _ in counts.most_common(max_vocab)]
    vocab = {w: i + 2 for i, w in enumerate(words)}
    return vocab, counts


def encode(texts, vocab, max_len):
    """Tokens -> integer ids, padded with 0 (PAD) to max_len."""
    encoded = []
    for text in texts:
        ids = [vocab.get(w, 1) for w in clean_text(text)]
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        encoded.append(ids)
    return np.array(encoded, dtype=int)


def run(prices, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    texts = [s for s, _ in SENTENCES]
    labels = np.array([y for _, y in SENTENCES])
    print("D09  Text cleaning, tokenisation & word cloud")
    print(f"  Corpus: {len(texts)} illustrative sentences "
          f"({int(labels.sum())} positive, {int((1 - labels).sum())} negative)")

    example = texts[0]
    print(f"  Example: '{example}'")
    print(f"    cleaned tokens: {clean_text(example)} — stopwords and "
          f"punctuation removed; 'beats' survives because it carries signal")
    vocab, counts = build_vocab(texts)
    max_len = max(len(clean_text(t)) for t in texts)
    encoded = encode(texts, vocab, max_len)
    print(f"  Vocabulary: {len(vocab)} distinct words "
          f"(top 5: {counts.most_common(5)}) — the vocabulary is "
          f"the model's whole world; any word outside it becomes UNK")
    print(f"  Padded length: {max_len} tokens; encoded example: "
          f"{encoded[0].tolist()} (0 = PAD, 1 = UNK)")
    print(f"  Reading: padding with 0 is a convention the model must "
          f"learn — in D10 the PAD embedding is a zero vector, so "
          f"padding adds literally nothing to the network's memory")

    # Word cloud: size = frequency. Saved to results/ (PNG, local only —
    # the GitHub push path takes text files, so charts stay on disk).
    cleaned_text = " ".join(" ".join(clean_text(t)) for t in texts)
    wc = WordCloud(width=800, height=400, background_color="white",
                   random_state=42).generate(cleaned_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most frequent words in the 40-sentence corpus")
    path = os.path.join(RESULTS_DIR, "d09_wordcloud.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    top_words = [w for w, _ in counts.most_common(8)]
    print(f"  Word cloud saved -> {path}; most frequent words: "
          f"{top_words} — frequency is NOT importance: 'stocks' is "
          f"common in both classes, so it cannot separate them; the "
          f"discriminating words are the rare emotional ones "
          f"('plunge', 'surge', 'record')")
    return {"nlp_vocab_size": int(len(vocab)),
            "nlp_max_len": int(max_len),
            "nlp_top_word": top_words[0]}
