"""Choosing a regularization strength by heritability on held-out groups.

H2-opt's noise level (select_noise_level) is the one whose first trait has the highest
heritability on held-out groups, averaged over validation folds of the groups. PCH penalizes
ridge * v * |w|^2 (v the mean variance of the measurements); its ridge is chosen the same way
(PCH.tune, through cross_validate). Also the assignment of groups to folds.
"""

import numpy as np
import torch

from .heritability import _as_environment, anova_heritability
from .train import synthetic_traits, train

# Noise levels of H2-opt: the standard deviation of the normal input noise as a fraction of
# noise_scale(X), the typical spread of a measurement.
NOISE_LEVELS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3)

# ridge 1 is the penalty of input noise with the standard deviation of a typical measurement.
RIDGES = tuple(float(r) for r in np.logspace(-4, 2, 13))


def per_group(groups, draw):
    """One value per group, given to every individual of the group.

    draw(c) returns the values of the c groups, in the sorted order of their labels."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    return np.asarray(draw(inverse.max() + 1))[inverse]


def group_folds(groups, n_folds=5, seed=0):
    """Fold (0..n_folds-1) of each individual; all individuals of a group share a fold, and the
    groups are spread evenly over the folds in a random order."""
    return per_group(groups, lambda c: np.random.RandomState(seed).permutation(c) % n_folds)


def cross_validate(X, groups, environment, score_fold, n_folds=5, seed=0):
    """Scores of a set of settings, per fold of the groups.

    score_fold(X_fit, groups_fit, environment_fit, X_val, groups_val, environment_val) returns
    one score per setting. Returns (scores averaged over the folds, (n_folds, n_settings) scores).
    """
    X = np.asarray(X)
    groups = np.asarray(groups)
    environment = _as_environment(environment, len(groups))
    fold = group_folds(groups, n_folds, seed)
    fold_scores = np.array([
        score_fold(X[fold != f], groups[fold != f], environment[fold != f],
                   X[fold == f], groups[fold == f], environment[fold == f])
        for f in range(n_folds)])
    return fold_scores.mean(axis=0), fold_scores


def noise_scale(X):
    """Root of the mean variance of the measurements (columns of the (n, m) array X)."""
    return float(np.sqrt(np.mean(np.var(np.asarray(X, dtype=float), axis=0))))


def select_noise_level(make_model, X, groups, environment, levels=NOISE_LEVELS, n_splits=2,
                       n_folds=5, seed=0, **train_options):
    """Noise level of H2-opt with the highest held-out heritability of the first trait.

    The groups are assigned to n_folds folds (group_folds); for each of the first n_splits folds,
    a fresh model (make_model(), a TraitModels) is trained on the other folds with normal noise of
    standard deviation level * noise_scale(X), for each level, and its first trait is scored by
    its ANOVA heritability on the held-out fold. train_options go to train (e.g. n_iter, clip,
    device). Returns (the level with the highest mean score, (n_splits, n_levels) scores).
    """
    X = np.asarray(X)
    groups = np.asarray(groups)
    environment = _as_environment(environment, len(groups))
    fold = group_folds(groups, n_folds, seed)
    scale = noise_scale(X)
    scores = np.zeros((n_splits, len(levels)))
    for split in range(n_splits):
        is_val = fold == split
        for a, level in enumerate(levels):
            torch.manual_seed(seed)
            model = make_model()
            train(model, X, groups, environment, is_val.astype(int), n_traits=1,
                  noise_level=level * scale, noise='normal', verbose=False, **train_options)
            Y = synthetic_traits(model, X, ~is_val, [0])
            scores[split, a] = anova_heritability(Y[is_val], groups[is_val],
                                                  environment[is_val])[0]
    return levels[int(np.argmax(scores.mean(axis=0)))], scores
