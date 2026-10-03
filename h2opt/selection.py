"""Choosing a regularization strength by cross-validated heritability.

PCH and LinearH2opt both penalize ridge * v * |w|^2 (v the mean variance of the measurements),
so they share one grid and one rule: the ridge whose traits have the highest mean heritability on
held-out groups, averaged over folds of the groups. Also the assignment of groups to folds.
"""

import numpy as np

from .heritability import _as_environment

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
