"""Baseline synthetic traits to compare H2-opt against.

Each baseline is a linear map fitted on training individuals: fit(X, groups, environment) then
transform(X). Traits are centered by the training mean and made uncorrelated on the training
individuals by Decorrelation, like H2-opt's traits. All computations happen in the span of
the centered training data, so the number of measurements may far exceed the number of individuals
(e.g. image pixels).
"""

import numpy as np

from .decorrelation import Decorrelation
from .heritability import Henderson3, _as_environment, anova_heritability, heritability
from .selection import RIDGES, cross_validate


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
    """Shared fit/transform; subclasses define _directions(Z, ...) in training coordinates."""

    def __init__(self, n_traits):
        self.n_traits = n_traits

    def fit(self, X, groups, environment=None):
        X, Xc, V, Z = self._coordinates(X)
        groups = np.asarray(groups)
        environment = _as_environment(environment, len(groups))
        return self._finish(Xc, V @ self._directions(Z, groups, environment,
                                                     n_features=X.shape[1]))

    def _coordinates(self, X):
        """X as floats, centered by its mean (set as mean_), and orthonormal coordinates Z of the
        centered data: Xc = Z V'."""
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        keep = s > s.max() * 1e-10
        V = Vt[keep].T
        return X, Xc, V, Xc @ V

    def _finish(self, Xc, W):
        """Set the map of the traits Xc W, made uncorrelated on the training individuals."""
        # Xc @ W has mean 0, so the map of the decorrelated traits is linear in Xc
        self.components_ = W @ Decorrelation().fit(Xc @ W).components_
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
    """Principal components of the genetic covariance G = (MS_between - MS_within) / r0.

    One-way ANOVA estimate ignoring environment; r0 is the effective group size for unbalanced
    groups.
    """

    def _directions(self, Z, groups, environment, n_features):
        n = Z.shape[0]
        _, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
        c = len(sizes)
        if c < 2 or n <= c:
            raise ValueError('genetic PCA needs at least two groups and some group with two or '
                             'more individuals')
        r0 = (n - np.sum(sizes ** 2) / n) / (c - 1)

        within = _group_center(Z, groups)
        between = Z - within
        G = (between.T @ between / (c - 1) - within.T @ within / (n - c)) / r0
        _, V = np.linalg.eigh((G + G.T) / 2)
        return V[:, ::-1][:, :self.n_traits]


class LDA(LinearBaseline):
    """Linear discriminant analysis: maximizes between-group over within-group variance.

    Ignores environment and is unregularized; the within-group scatter is inverted on its range.
    """

    def _directions(self, Z, groups, environment, n_features):
        within = _group_center(Z, groups)
        between = Z - within
        return _top_generalized(between.T @ between, within.T @ within, self.n_traits)


class PCH(LinearBaseline):
    """Principal components of heritability with ridge regularization (Wang et al. 2007).

    Maximizes w'A w / w'(B + ridge * tr(B) / p * I) w, where w'A w / w'B w is exactly the
    heritability of the trait X w by the estimator (ESTIMATORS in h2opt.heritability) and p is the
    number of measurements: 'anova', anova_heritability (environment means removed, groups with
    one individual dropped), or 'henderson3', Henderson3(groups, environment, subgroups). Successive
    traits maximize the same ratio subject to being uncorrelated with the earlier ones. ridge = 0
    gives unregularized PCH.
    """

    def __init__(self, n_traits, ridge=0.0, estimator='anova'):
        super().__init__(n_traits)
        self.ridge = ridge
        self.estimator = estimator

    def fit(self, X, groups, environment=None, subgroups=None):
        fitted = self.fit_ridges(X, groups, environment, subgroups, self.n_traits, [self.ridge],
                                 self.estimator)[0]
        self.mean_, self.components_ = fitted.mean_, fitted.components_
        return self

    def fit_transform(self, X, groups, environment=None, subgroups=None):
        return self.fit(X, groups, environment, subgroups).transform(X)

    @classmethod
    def fit_ridges(cls, X, groups, environment, subgroups, n_traits, ridges, estimator='anova'):
        """PCH fitted at each of the ridges; the decomposition of X and the heritability forms
        are computed once. Returns a list of fitted PCH."""
        start = cls(n_traits, estimator=estimator)
        X, Xc, V, Z = start._coordinates(X)
        groups = np.asarray(groups)
        environment = _as_environment(environment, len(groups))
        if estimator == 'anova':
            if subgroups is not None:
                raise ValueError("subgroups need estimator='henderson3'")
            A, B = cls.heritability_forms(Z, groups, environment)
        else:
            A, B = Henderson3(groups, environment, subgroups).forms()
            A, B = Z.T @ A @ Z, Z.T @ B @ Z
        fitted = []
        for ridge in ridges:
            pch = cls(n_traits, ridge, estimator)
            pch.mean_ = start.mean_
            regularized = B + ridge * np.trace(B) / X.shape[1] * np.eye(B.shape[0])
            fitted.append(pch._finish(Xc, V @ _top_generalized(A, regularized, n_traits)))
        return fitted

    @classmethod
    def tune(cls, X, groups, environment, n_traits, ridges=RIDGES, n_folds=5, seed=0,
             subgroups=None, estimator='anova'):
        """Choose the ridge by cross-validated heritability.

        For each fold of the groups (h2opt.selection.cross_validate) and each ridge, PCH is fitted
        on the other folds and scored by the mean heritability (by the same estimator) of its
        traits on the held-out fold. X is one (n, p) array, or a list of arrays of the same
        individuals (e.g. one per date); then one ridge serves all of them, PCH is fitted to each,
        and the score is the mean over them.
        Returns (ridge with the best mean score, scores (len(ridges),), fold_scores).
        """
        sets = X if isinstance(X, list | tuple) else [X]
        groups = np.asarray(groups)
        environment = _as_environment(environment, len(groups))

        def part(labels, rows):
            return None if labels is None else np.asarray(labels)[rows]

        def score_fold(fit, val):
            scores = []
            for measurements in sets:
                measurements = np.asarray(measurements)
                fitted = cls.fit_ridges(measurements[fit], groups[fit], environment[fit],
                                        part(subgroups, fit), n_traits, ridges, estimator)
                traits = np.concatenate([pch.transform(measurements[val]) for pch in fitted],
                                        axis=1)
                scores.append(heritability(traits, groups[val], environment[val],
                                           part(subgroups, val), estimator))
            return np.mean(scores, axis=0).reshape((len(ridges), n_traits)).mean(axis=1)

        scores, fold_scores = cross_validate(groups, score_fold, n_folds, seed)
        return ridges[int(np.argmax(scores))], scores, fold_scores

    @staticmethod
    def heritability_forms(Z, groups, environment=None):
        """Matrices A and B with w'A w / w'B w = anova_heritability(Z w) for every w."""
        environment = _as_environment(environment, len(groups))
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        keep = counts[inverse] >= 2
        Zc = Z[keep] - Z[keep].mean(axis=0)
        Ze = Zc
        for a in range(environment.shape[1]):
            Ze = _group_center(Ze, environment[keep, a])
        R = _group_center(Ze, groups[keep])
        _, inverse_kept, counts_kept = np.unique(groups[keep], return_inverse=True,
                                                 return_counts=True)
        scale = (counts_kept / (counts_kept - 1.0))[inverse_kept]
        return Ze.T @ Ze - R.T @ (R * scale[:, None]), Zc.T @ Zc


class MaxHeritabilityFeatures:
    """Greedily select the n_traits most heritable measurements (features) on training individuals.

    After each pick, the chosen measurement is projected out of all others before the next pick.
    transform returns the selected measurements, centered and made uncorrelated on the training
    individuals.
    """

    def __init__(self, n_traits):
        self.n_traits = n_traits

    def fit(self, X, groups, environment=None):
        X = np.asarray(X, dtype=float)
        groups = np.asarray(groups)
        X_fit = X - X.mean(axis=0)
        chosen = []
        for _ in range(self.n_traits):
            heritability = anova_heritability(X_fit, groups, environment)
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
        decorrelation = Decorrelation().fit(X[:, self.features_])
        self.mean_, self.components_ = decorrelation.mean_, decorrelation.components_
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float)[:, self.features_] - self.mean_) @ self.components_

    def fit_transform(self, X, groups, environment=None):
        return self.fit(X, groups, environment).transform(X)
