"""D10 — A custom-built LSTM for sentiment (CFA L2 PSM Unit 4:
"Natural Language Processing (NLP) in Python").

CFA idea: a plain feedforward network reads a sentence all at once
and forgets the ORDER of the words — "not good" and "good not" look
identical. A RECURRENT network reads one word at a time and carries
a MEMORY forward. An LSTM (Long Short-Term Memory) is a recurrent
network with three GATES that decide what the memory keeps:
- FORGET gate: what to throw away from the memory ("the" adds nothing).
- INPUT gate: what new information to write in ("plunge" matters).
- OUTPUT gate: what to read out for the prediction.
Because the gates are learned, the network can remember a word from
the start of a sentence ("not") when it reaches the end ("good") —
that is the whole point of the architecture, and the reason the
official CFA module uses an LSTM for sentiment analysis.

This module builds the LSTM FROM SCRATCH in NumPy — forward pass,
backpropagation through time, and Adam optimisation — no deep
learning framework. The embedding layer is fixed random vectors
(seed 42); only the LSTM weights and the final classifier are
trained. The corpus is D09's 40 illustrative sentences, so the
result is directly comparable with D11's TF-IDF baseline.

How to read the result:
- Training loss (binary cross-entropy): 0.69 is coin-flip level
  (ln 2), 0.30 is decent, below 0.10 is memorisation on a toy set.
  Watch it FALL across epochs — if it flatlines near 0.69, the
  network learned nothing (check the learning rate and the gates).
- Train vs test accuracy: on 40 sentences the test set is 8
  sentences, so one sentence = 12.5 points. Treat the test number
  as a sanity check, not a claim — the ARCHITECTURE is the lesson,
  and D11 shows what a simpler model does on the same data.
"""
import numpy as np

from d09_text_cleaning_tokenization import (
    SENTENCES, build_vocab, clean_text, encode)

EMB_DIM = 8
HIDDEN = 8
MAX_LEN = 12
EPOCHS = 120
LR = 0.01


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LSTM:
    """One-layer LSTM + sigmoid classifier, NumPy from scratch."""

    def __init__(self, vocab_size, rng):
        n_in = EMB_DIM + HIDDEN  # [previous hidden; current embedding]
        scale = 1.0 / np.sqrt(n_in)
        self.params = {}
        for gate in ("f", "i", "c", "o"):
            self.params["W" + gate] = rng.normal(0, scale, (HIDDEN, n_in))
            # Forget-gate bias starts at 1.0: remember by default.
            self.params["b" + gate] = np.ones(HIDDEN) if gate == "f" \
                else np.zeros(HIDDEN)
        self.params["Wy"] = rng.normal(0, 1.0 / np.sqrt(HIDDEN), (1, HIDDEN))
        self.params["by"] = np.zeros(1)
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.step = 0

    def forward(self, emb_seq):
        """emb_seq: (T, EMB_DIM). Returns (prob, caches)."""
        T = len(emb_seq)
        h = np.zeros(HIDDEN)
        c = np.zeros(HIDDEN)
        caches = []
        for t in range(T):
            x = np.concatenate([h, emb_seq[t]])
            f = _sigmoid(self.params["Wf"] @ x + self.params["bf"])
            i = _sigmoid(self.params["Wi"] @ x + self.params["bi"])
            g = np.tanh(self.params["Wc"] @ x + self.params["bc"])
            o = _sigmoid(self.params["Wo"] @ x + self.params["bo"])
            c_prev = c
            c = f * c_prev + i * g
            h = o * np.tanh(c)
            caches.append((x, f, i, g, o, c_prev, c, h))
        logit = float(self.params["Wy"] @ h + self.params["by"])
        return _sigmoid(logit), caches, h

    def backward(self, emb_seq, caches, h_last, y, prob):
        """Gradients of the BCE loss w.r.t. every parameter."""
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dz = prob - y  # d loss / d logit (sigmoid + BCE simplify nicely)
        grads["Wy"] = dz * h_last.reshape(1, -1)
        grads["by"] = np.array([dz])
        dh = (self.params["Wy"].T * dz).reshape(-1)
        dc_next = np.zeros(HIDDEN)
        d_emb = np.zeros_like(emb_seq)
        for t in reversed(range(len(emb_seq))):
            x, f, i, g, o, c_prev, c, h = caches[t]
            tanh_c = np.tanh(c)
            do = dh * tanh_c
            d_o = do * o * (1 - o)
            dc = dh * o * (1 - tanh_c ** 2) + dc_next
            df = dc * c_prev
            d_f = df * f * (1 - f)
            di = dc * g
            d_i = di * i * (1 - i)
            dg = dc * i
            d_g = dg * (1 - g ** 2)
            for gate, d_pre in (("f", d_f), ("i", d_i),
                                ("c", d_g), ("o", d_o)):
                grads["W" + gate] += np.outer(d_pre, x)
                grads["b" + gate] += d_pre
            dx = (self.params["Wf"].T @ d_f + self.params["Wi"].T @ d_i
                  + self.params["Wc"].T @ d_g + self.params["Wo"].T @ d_o)
            dh = dx[:HIDDEN]
            d_emb[t] = dx[HIDDEN:]
            dc_next = dc * f
        return grads

    def adam_step(self, grads):
        self.step += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.params:
            g = np.clip(grads[k], -5, 5)  # clip: exploding gradients
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1 ** self.step)
            v_hat = self.v[k] / (1 - b2 ** self.step)
            self.params[k] -= LR * m_hat / (np.sqrt(v_hat) + eps)


def run(prices, source):
    rng = np.random.default_rng(42)
    texts = [s for s, _ in SENTENCES]
    labels = np.array([y for _, y in SENTENCES], dtype=float)
    vocab, _ = build_vocab(texts)
    encoded = encode(texts, vocab, MAX_LEN)
    # Fixed random embeddings (seeded): PAD row 0 stays zero, so padding
    # contributes nothing to the memory — exactly the D09 promise.
    emb_table = rng.normal(0, 0.5, (len(vocab) + 2, EMB_DIM))
    emb_table[0] = 0.0

    order = rng.permutation(len(texts))
    train_idx, test_idx = order[:32], order[32:]
    model = LSTM(len(vocab) + 2, rng)
    print("D10  Custom-built LSTM (NumPy, from scratch) — sentiment")
    print(f"  Architecture: embedding dim {EMB_DIM} -> LSTM hidden "
          f"{HIDDEN} -> sigmoid output; {len(vocab) + 2} word ids, "
          f"sequences padded to {MAX_LEN}; Adam lr {LR}, {EPOCHS} epochs")
    losses = []
    for epoch in range(1, EPOCHS + 1):
        total_loss, grads_sum = 0.0, None
        for idx in train_idx:
            emb_seq = emb_table[encoded[idx]]
            prob, caches, h = model.forward(emb_seq)
            y = labels[idx]
            loss = -(y * np.log(prob + 1e-9)
                     + (1 - y) * np.log(1 - prob + 1e-9))
            total_loss += loss
            grads = model.backward(emb_seq, caches, h, y, prob)
            grads_sum = grads if grads_sum is None else {
                k: grads_sum[k] + grads[k] for k in grads}
        avg_grads = {k: v / len(train_idx) for k, v in grads_sum.items()}
        model.adam_step(avg_grads)
        losses.append(total_loss / len(train_idx))
        if epoch in (1, 30, 60, 90, EPOCHS):
            print(f"    epoch {epoch:>3}: train loss {losses[-1]:.3f} "
                  f"({'coin-flip level — nothing learned yet' if losses[-1] > 0.6 else 'falling — the gates are learning what matters' if losses[-1] > 0.2 else 'low — the network fits the toy set; check the test score before believing it'})")

    def accuracy(idxs):
        correct = 0
        for idx in idxs:
            prob, _, _ = model.forward(emb_table[encoded[idx]])
            correct += int((prob > 0.5) == bool(labels[idx]))
        return correct / len(idxs)

    acc_train, acc_test = accuracy(train_idx), accuracy(test_idx)
    print(f"  Train accuracy {acc_train:.1%} ({int(acc_train * len(train_idx))}/{len(train_idx)}), "
          f"test accuracy {acc_test:.1%} ({int(acc_test * len(test_idx))}/{len(test_idx)}) — "
          f"with 8 test sentences, one sentence is 12.5 points, so "
          f"read this as a sanity check, not a claim")
    print("  Reading: the LSTM is the CFA module's architecture, built "
          "from first principles — the lesson is the GATES (forget / "
          "input / output), not the accuracy on 40 sentences. D11 runs "
          "the same task with a far simpler model; compare the two")
    return {"lstm_loss_final": float(losses[-1]),
            "lstm_acc_train": float(acc_train),
            "lstm_acc_test": float(acc_test)}
