"""The five experiments from the assignment. Each returns plain Python data."""
import time

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from .data import ALL_FEATURES, LABEL
from .model import MixedRegression

ITERS = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100]
KNEE_N_ITER = 5  # chosen from Part 1
COMPONENTS = [1, 2, 4, 8, 16, 32]
DEPTHS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30]


def median_time(fn, repeats=50):
    """Median wall time of fn() over repeated calls, after one warm-up call."""
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def evaluate_mixture(d, features, n_components, n_iter, time_pred=False):
    """Fit on train, route test points with the classifier, return (MAE in ppb, time, model)."""
    Xtr, ytr = d["train_std"][features], d["train_std"][LABEL]
    Xte = d["test_std"][features]
    model = MixedRegression(n_components=n_components, n_iter=n_iter)
    model.fit(Xtr, ytr, d["train"][LABEL])
    model.fit_classifier(Xtr, ytr)
    pt = None
    if time_pred:
        pt = median_time(lambda: model.predict(Xte))
    pred, _ = model.predict(Xte)
    mae = mean_absolute_error(d["test"][LABEL], pred * d["y_sd"] + d["y_mean"])
    return float(mae), pt, model


def part1_iterations(d):
    return [evaluate_mixture(d, ALL_FEATURES, 4, ni)[0] for ni in ITERS]


def part2_components(d):
    maes, models = [], {}
    for nc in COMPONENTS:
        mae, _, m = evaluate_mixture(d, ALL_FEATURES, nc, KNEE_N_ITER)
        maes.append(mae)
        models[nc] = m
    return maes, models


def part3_assignments(d, models, ks=(2, 4, 8)):
    """True O3 values of the training points assigned to each component."""
    X, y = d["train_std"][ALL_FEATURES], d["train_std"][LABEL]
    y_orig = d["train"][LABEL].values
    out = {}
    for k in ks:
        best = models[k].assign(X, y)
        out[k] = [y_orig[best == c] for c in range(k)]
    return out


def part4_ablation(d):
    """Test MAE per number of components, dropping one feature at a time."""
    out = {}
    for excluded in ALL_FEATURES:
        feats = [f for f in ALL_FEATURES if f != excluded]
        out[excluded] = [evaluate_mixture(d, feats, nc, KNEE_N_ITER)[0] for nc in COMPONENTS]
    return out


def part5_tree_vs_mixture(d):
    """Tune a decision tree at depth 10, sweep depth, and time both models."""
    Xtr, ytr = d["train_std"][ALL_FEATURES], d["train_std"][LABEL]
    Xte, yte = d["test_std"][ALL_FEATURES], d["test"][LABEL]

    def tree(**kw):
        return DecisionTreeRegressor(random_state=0, **kw).fit(Xtr, ytr)

    best = {"mae": np.inf}
    for criterion in ["squared_error", "friedman_mse", "absolute_error"]:
        for splitter in ["best", "random"]:
            for min_split in [2, 10, 50]:
                t = tree(criterion=criterion, splitter=splitter,
                         max_depth=10, min_samples_split=min_split)
                mae = mean_absolute_error(yte, t.predict(Xte) * d["y_sd"] + d["y_mean"])
                if mae < best["mae"]:
                    best = {"mae": float(mae), "criterion": criterion,
                            "splitter": splitter, "min_split": min_split}

    trees = []
    for depth in DEPTHS:
        t = tree(criterion=best["criterion"], splitter=best["splitter"],
                 max_depth=depth, min_samples_split=best["min_split"])
        pt = median_time(lambda: t.predict(Xte))
        pred = t.predict(Xte) * d["y_sd"] + d["y_mean"]
        trees.append({"max_depth": depth, "mae": float(mean_absolute_error(yte, pred)),
                      "pred_time": pt})

    mixtures = []
    for nc in COMPONENTS:
        mae, pt, _ = evaluate_mixture(d, ALL_FEATURES, nc, KNEE_N_ITER, time_pred=True)
        mixtures.append({"n_components": nc, "mae": mae, "pred_time": pt})
    return best, trees, mixtures
