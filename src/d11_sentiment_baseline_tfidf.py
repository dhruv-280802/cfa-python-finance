"""D11 — Sentiment with an off-the-shelf model (CFA L2 PSM Unit 4:
"Natural Language Processing (NLP) in Python").

CFA idea: you do not always build the model — often you USE one.
The practical, off-the-shelf sentiment pipeline is TF-IDF +
logistic regression: TF-IDF turns each sentence into a vector of
word weights (a word scores high when it appears often in THIS
sentence but rarely in the corpus — "plunge" is informative,
"stocks" is not), and logistic regression learns which weighted
words push the sentence toward positive or negative. It is fast,
transparent (every coefficient is a word you can read), and on
small text sets it usually BEATS a neural network — the honest
baseline every fancy model must clear.

How to read the result:
- Accuracy: share of test sentences classified correctly. 50% is a
  coin flip (the baseline for a balanced set). Above 70% on 40
  sentences is respectable; below 60% means the signal is weak or
  the vocabulary overlaps too much.
- Precision / recall per class: precision = "of the sentences I
  called positive, how many really were"; recall = "of the truly
  positive sentences, how many did I catch". A model can be 80%
  accurate and still useless if it never catches the negative class.
- Top coefficients: the words the model leans on most. If the top
  positive words are "record", "surge", "beats" and the top negative
  ones are "plunge", "misses", "fears", the model learned the
  finance vocabulary — if the top words are stopwords, the cleaning
  (D09) failed.

Where a pre-trained transformer slots in: the industrial version
replaces TF-IDF + logistic regression with a fine-tuned FinBERT
(a BERT model pre-trained on financial text, via Hugging Face):
the interface is the same — text in, sentiment out — but the model
arrives already knowing finance language, so it needs far less
labelled data. It is NOT included here because it downloads ~400MB
of weights and wants a GPU for training; the TF-IDF baseline is
the correct first step, and the swap is a one-line change in
`sklearn.pipeline` terms (replace vectorizer + classifier with a
transformers pipeline).
"""
import numpy as np

from d09_text_cleaning_tokenization import SENTENCES


def run(prices, source):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline

    texts = [s for s, _ in SENTENCES]
    labels = np.array([y for _, y in SENTENCES])
    print("D11  Sentiment — TF-IDF + logistic regression (off-the-shelf)")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels)
    model = make_pipeline(
        TfidfVectorizer(max_features=200, stop_words="english"),
        LogisticRegression(max_iter=1000, random_state=42))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = float((pred == y_test).mean())
    print(f"  Train {len(X_train)} sentences, test {len(X_test)} "
          f"(stratified, seed 42); test accuracy {acc:.1%} "
          f"({int((pred == y_test).sum())}/{len(y_test)}) — "
          f"{'well above the 50% coin flip: the vocabulary carries real signal' if acc >= 0.7 else 'near the 50% coin flip: on 40 sentences the classes overlap heavily, or the model needs more data'}")
    report = classification_report(y_test, pred, output_dict=True,
                                   zero_division=0)
    for cls, name in (("0", "negative"), ("1", "positive")):
        m = report[cls]
        print(f"  {name:>8}: precision {m['precision']:.2f} (of the "
              f"sentences called {name}, this share really were), "
              f"recall {m['recall']:.2f} (of the truly {name} sentences, "
              f"this share were caught), F1 {m['f1-score']:.2f}")
    vectorizer = model[0]
    coefs = model[1].coef_[0]
    words = np.array(vectorizer.get_feature_names_out())
    top_pos = words[np.argsort(coefs)[-5:]][::-1]
    top_neg = words[np.argsort(coefs)[:5]]
    print(f"  Most positive words: {list(top_pos)}; most negative: "
          f"{list(top_neg)} — if these look like finance emotion "
          f"words, the model learned the vocabulary; if they look "
          f"random, it fitted noise")
    print("  Compare with D10 (custom LSTM): the simpler model is the "
          "bar the LSTM must clear. In production the next step is "
          "FinBERT — same text-in/sentiment-out interface, "
          "pre-trained on financial text, ~400MB download, GPU helps")
    return {"tfidf_acc_test": acc,
            "tfidf_top_positive": str(list(top_pos)),
            "tfidf_top_negative": str(list(top_neg))}
