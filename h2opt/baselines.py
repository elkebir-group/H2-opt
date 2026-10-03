"""Baseline synthetic traits to compare H2-opt against.

Each baseline is a linear map fitted on training individuals: fit(X, groups, environment) then transform(X).
Traits are centered by the training mean and made uncorrelated on the training individuals (Gram-Schmidt in
order), like H2-opt's traits. All computations happen in the span of the centered training data, so the number
of measurements may far exceed the number of individuals (e.g. image pixels).
"""

import numpy as np
import torch

from .heritability import anova_heritability


def _group_center(Z, labels):
    """Subtract from each row the mean of its group."""
    _, inverse = np.unique(labels, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, Z.shape[1]))
    np.add.at(sums, inverse, Z)
    return Z - (sums / np.bincount(inverse)[:, None])[inverse]


def _top_generalized(A, B, k, tol=1e-10):
    """Top-k solutions of A w = l B w for symmetric A and positive semidefinite B.

    Directions with (numerically) zero B are excluded, i.e. B is inverted on its range.
    """
    lam, U = np.linalg.eigh((B + B.T) / 2)
    keep = lam > lam.max() * tol
    whiten = U[:, keep] / np.sqrt(lam[keep])
    _, V = np.linalg.eigh(whiten.T @ ((A + A.T) / 2) @ whiten)
    if k > V.shape[1]:
        raise ValueError(f'at most {V.shape[1]} traits can be extracted, {k} requested')
    return whiten @ V[:, ::-1][:, :k]


class LinearBaseline:
    """Shared fit/transform; subclasses define _directions(Z, groups, environment) in training coordinates."""

    def __init__(self, n_traits):
        self.n_traits = n_traits

    def fit(self, X, groups, environment=None):
        X = np.asarray(X, dtype=float)
        groups = np.asarray(groups)
        if environment is not None:
            environment = np.asarray(environment)
            if environment.ndim == 1:
                environment = environment.reshape((-1, 1))

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # orthonormal coordinates of the training data: Xc = Z V'
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        keep = s > s.max() * 1e-10
        V = Vt[keep].T
        Z = Xc @ V

        W = V @ self._directions(Z, groups, environment, n_features=X.shape[1])

        # Gram-Schmidt on the training traits: Y = Q R, keep the scale of each trait's own component
        _, R = np.linalg.qr(Xc @ W)
        self.components_ = W @ np.linalg.inv(R) @ np.diag(np.diag(R))
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_

    def fit_transform(self, X, groups, environment=None):
        return self.fit(X, groups, environment).transform(X)


class PCA(LinearBaseline):
    """Principal components of the measurements."""

    def _directions(self, Z, groups, environment, n_features):
        return np.eye(Z.shape[1])[:, :self.n_traits]


class GeneticPCA(LinearBaseline):
    """Principal components of the genetic covariance G = (MS_between - MS_within) / r0 of the groups.

    One-way ANOVA estimate ignoring environment; r0 is the effective group size for unbalanced groups.
    """

    def _directions(self, Z, groups, environment, n_features):
        n = Z.shape[0]
        _, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
        c = len(sizes)
        if c < 2 or n <= c:
            raise ValueError('genetic PCA needs at least two groups and some group with two or more individuals')
        r0 = (n - np.sum(sizes ** 2) / n) / (c - 1)

        within = _group_center(Z, groups)
        between = Z - within
        G = (between.T @ between / (c - 1) - within.T @ within / (n - c)) / r0
        _, V = np.linalg.eigh((G + G.T) / 2)
        return V[:, ::-1][:, :self.n_traits]


class LDA(LinearBaseline):
    """Linear discriminant analysis of the groups: maximizes between-group over within-group variance.

    Ignores environment and is unregularized; the within-group scatter is inverted on its range.
    """

    def _directions(self, Z, groups, environment, n_features):
        within = _group_center(Z, groups)
        between = Z - within
        return _top_generalized(between.T @ between, within.T @ within, self.n_traits)


class PCH(LinearBaseline):
    """Principal components of heritability with ridge regularization (Wang et al. 2007).

    Maximizes w'A w / w'(B + ridge * tr(B) / p * I) w, where w'A w / w'B w is exactly the ANOVA heritability
    of the trait X w (anova_heritability: environment means removed, groups with one individual dropped)
    and p is the number of measurements. Successive traits maximize the same ratio subject to being
    uncorrelated with the earlier ones. ridge = 0 gives unregularized PCH.
    """

    def __init__(self, n_traits, ridge=0.0):
        super().__init__(n_traits)
        self.ridge = ridge

    def _directions(self, Z, groups, environment, n_features):
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        keep = counts[inverse] >= 2
        Zc = Z[keep] - Z[keep].mean(axis=0)
        Ze = Zc
        if environment is not None:
            for a in range(environment.shape[1]):
                Ze = _group_center(Ze, environment[keep, a])
        R = _group_center(Ze, groups[keep])
        _, inverse_kept, counts_kept = np.unique(groups[keep], return_inverse=True, return_counts=True)
        scale = (counts_kept / (counts_kept - 1.0))[inverse_kept]

        A = Ze.T @ Ze - R.T @ (R * scale[:, None])
        B = Zc.T @ Zc
        B = B + self.ridge * np.trace(B) / n_features * np.eye(B.shape[0])
        return _top_generalized(A, B, self.n_traits)


class MaxHeritabilityFeatures:
    """Greedily select the n_traits most heritable measurements (e.g. wavelengths) on the training individuals.

    After each pick, the chosen measurement is projected out of all others before the next pick.
    transform returns the selected measurements, centered and made uncorrelated on the training individuals.
    """

    def __init__(self, n_traits):
        self.n_traits = n_traits

    def fit(self, X, groups, environment=None):
        X = np.asarray(X, dtype=float)
        groups = np.asarray(groups)
        X_fit = X - X.mean(axis=0)
        chosen = []
        for _ in range(self.n_traits):
            heritability = anova_heritability(torch.tensor(X_fit).float(), groups, environment).numpy()
            heritability[np.isnan(heritability)] = 0
            heritability[np.array(chosen, dtype=int)] = 0
            best = int(np.argmax(heritability))
            chosen.append(best)

            u = np.copy(X_fit[:, best])
            X_fit = X_fit - np.outer(u, (u @ X_fit) / (u @ u))
            # columns that became constant (including the chosen one) are replaced by a constant 1
            scale = np.sum(np.abs(X_fit), axis=0)
            X_fit[:, np.isnan(scale)] = 1
            X_fit[:, scale < 1e-10] = 1
            X_fit[:, best] = 1

        self.features_ = np.array(chosen)
        self.mean_ = X[:, self.features_].mean(axis=0)
        _, R = np.linalg.qr(X[:, self.features_] - self.mean_)
        self.components_ = np.linalg.inv(R) @ np.diag(np.diag(R))
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float)[:, self.features_] - self.mean_) @ self.components_

    def fit_transform(self, X, groups, environment=None):
        return self.fit(X, groups, environment).transform(X)
