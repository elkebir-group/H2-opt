"""Folds of groups: assigning groups of individuals to folds, and cross-validation over them."""

import numpy as np


def per_group(groups, draw):
    """One value per group, given to every individual of the group.

    draw(c) returns the values of the c groups, in the sorted order of their labels."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    return np.asarray(draw(inverse.max() + 1))[inverse]


def group_folds(groups, n_folds=5, seed=0):
    """Fold (0..n_folds-1) of each individual; all individuals of a group share a fold, and the
    groups are spread evenly over the folds in a random order."""
    return per_group(groups, lambda c: np.random.RandomState(seed).permutation(c) % n_folds)


def cross_validate(units, score_fold, n_folds=5, seed=0):
    """Scores of a set of settings, per fold of the units (labels of the individuals, e.g. their
    groups; group_folds keeps the individuals of a unit together).

    score_fold(fit, val), with boolean masks of the training and held-out individuals, returns one
    score per setting. Returns (scores averaged over the folds, (n_folds, n_settings) scores).
    """
    fold = group_folds(np.asarray(units), n_folds, seed)
    fold_scores = np.array([score_fold(fold != f, fold == f) for f in range(n_folds)])
    return fold_scores.mean(axis=0), fold_scores
