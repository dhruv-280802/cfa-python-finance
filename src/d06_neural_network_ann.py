"""D06 — Feedforward neural networks (CFA L2 PSM Unit 3: "Financial
Forecasting Using Machine and Deep Learning").

CFA idea: a neural network is just regression with extra steps. One
neuron takes a weighted sum of its inputs, adds a bias, and passes
the result through a NON-LINEAR activation (here ReLU: max(0, x)) —
without the non-linearity, ten layers would collapse into one linear
regression. Stack neurons into layers, layers into a network, and the
network can bend itself around patterns a straight line cannot. The
price: the network can also bend itself around NOISE (overfitting),
its weights start random (so results vary run to run unless you fix
the seed), and you cannot read a coefficient off it the way you can
with OLS — it is a black box with a good track record.

Theory in one paragraph: training = backpropagation. The network
makes a prediction, the loss (squared error) measures how wrong it
was, and the chain rule pushes the error backwards through every
layer, nudging each weight a little in the direction that reduces
the loss. Repeat thousands of times (epochs). Inputs MUST be scaled
first — an unscaled price (~$300) next to a return (~0.01) makes the
optimiser chase the big number and ignore the small one.

How to read the result: the same R²/RMSE as D05, on the same
features and the same train/test split — so the ANN and the Ridge
model are directly comparable. If the ANN wins by a hair, that is
the honest size of the non-linear edge in daily returns.

Worked example: sklearn's MLPRegressor (2 hidden layers: 16, then 8
neurons) predicting tomorrow's AAPL return, vs the D05 Ridge model.
"""
import numpy as np

from d05_sklearn_workflow import FEATURES, build_features, _metrics


def run(prices, source):
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    frame = build_features(prices)
    split = int(len(frame) * 0.70)
    train, test = frame.iloc[:split], frame.iloc[split:]
    print("D06  Feedforward neural network (MLP) — same task as D05")
    print(f"  Architecture: 5 inputs -> 16 ReLU neurons -> 8 ReLU "
          f"neurons -> 1 output; trained by backpropagation (Adam), "
          f"max 2000 iterations, seed fixed at 42")

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(train[FEATURES], train["target"])
    _, _, r2_ridge = _metrics(test["target"],
                              ridge.predict(test[FEATURES]),
                              "Ridge (test) ")
    # The target (daily returns, ~0.01 in size) is scaled x100 before
    # fitting: MLPRegressor optimises the raw squared error, and with
    # unscaled targets the loss is ~0.0002 — so small that the
    # solver's tolerance stops training after a few dozen iterations
    # having learned nothing. Scaling the target to percent makes the
    # optimisation well-conditioned; predictions are divided back.
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=2000,
                     learning_rate_init=0.005, random_state=42))
    mlp.fit(train[FEATURES], train["target"] * 100)
    print(f"  MLP converged in {mlp[-1].n_iter_} iterations; final "
          f"training loss {mlp[-1].loss_:.4f} (mean squared error on "
          f"returns x100 — lower is better, but a very low training "
          f"loss with a poor test score is the definition of overfitting)")
    rmse_te, _, r2_te = _metrics(test["target"],
                                 mlp.predict(test[FEATURES]) / 100,
                                 "MLP (test)  ")
    print(f"  Reading: MLP test R² {r2_te:+.3f} vs Ridge {r2_ridge:+.3f} — "
          f"{'the network found a small non-linear edge the straight line missed' if r2_te > r2_ridge + 0.005 else 'no real edge over a straight line — the extra flexibility mostly fitted noise, which is the usual outcome on daily returns'}")
    print("  Lesson: a neural network is not automatically better than "
          "regression. On noisy, low-signal data the simpler model "
          "often wins — and you can explain every coefficient of the "
          "simple one in an interview")
    return {"ann_rmse_test": rmse_te, "ann_r2_test": r2_te,
            "ann_r2_ridge": r2_ridge, "ann_iterations": int(mlp[-1].n_iter_)}
