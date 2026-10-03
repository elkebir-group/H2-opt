"""Baseline synthetic traits to compare H2-opt against."""

import numpy as np
import torch

from .heritability import anova_heritability
from .train import decorrelate


def _remove_projections(X, u):
    """Remove from each column of X its projection on each column of u (u is not orthogonalized)."""
    for a in range(X.shape[1]):
        for b in range(u.shape[1]):
            X[:, a] = X[:, a] - (np.dot(X[:, a], u[:, b]) / np.dot(u[:, b], u[:, b])) * u[:, b]
    return X


def max_heritability_features(X, n_traits, groups, environment, train_test):
    """Greedily select the n_traits most heritable measurements (e.g. wavelengths) on the training set.

    After each pick, the chosen measurement is projected out of all others before the next pick.
    Returns the selected original measurements, decorrelated, as an (n, n_traits) array.
    """
    groups = np.asarray(groups)
    environment = np.asarray(environment) if environment is not None else None
    is_train = np.asarray(train_test) == 0

    X_train = np.copy(X[is_train])
    X_train = X_train - np.mean(X_train, axis=0).reshape((1, -1))
    chosen = []
    for _ in range(n_traits):
        heritability = anova_heritability(torch.tensor(X_train).float(), groups[is_train],
                                          environment[is_train] if environment is not None else None).numpy()
        heritability[np.isnan(heritability)] = 0
        heritability[np.array(chosen, dtype=int)] = 0
        best = np.argmax(heritability)
        chosen.append(best)

        X_train = _remove_projections(X_train, np.copy(X_train[:, best:best + 1]))
        # columns that became constant (including the chosen one) are replaced by a constant 1
        scale = np.sum(np.abs(X_train), axis=0)
        X_train[:, np.isnan(scale)] = 1
        X_train[:, scale < 1e-10] = 1
        X_train[:, best] = 1

    Y = np.copy(X[:, np.array(chosen)])
    return decorrelate(torch.tensor(Y).float()).numpy()
