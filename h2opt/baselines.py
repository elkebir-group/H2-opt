"""Baseline synthetic traits to compare H2-opt against.

Each baseline is a linear map fitted on training individuals: fit(X, groups, environment) then
transform(X). Traits are centered by the training mean and made uncorrelated on the training
individuals by Decorrelation, like H2-opt's traits. All computations happen in the span of
the centered training data, so the number of measurements may far exceed the number of individuals
(e.g. image pixels).
"""

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import eigsh

from .decorrelation import Decorrelation
from .heritability import _as_environment, anova_heritability, heritability_design
from .selection import NOISE_LEVELS, best_level, cross_validate_levels

# Ridges of PCH: the squares of H2-opt's noise levels (1e-6 to 1e4). PCH penalizes
# ridge * v * |w|^2 with v the mean variance of the measurements, the penalty that normal input
# noise of standard deviation sqrt(ridge * v) adds to a linear trait, so ridge s^2 matches H2-opt
# at noise level s.
RIDGES = tuple(s**2 for s in NOISE_LEVELS)


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


def _top_symmetric(matrix, k):
    """Eigenvectors of the k largest eigenvalues of the symmetric matrix, largest first."""
    n = matrix.shape[0]
    if k > n:
        raise ValueError(f'at most {n} traits can be extracted, {k} requested')
    if n > 10 * k + 100:
        # a few Lanczos iterations instead of the full decomposition
        values, V = eigsh(matrix, k=k, which='LA', tol=0, v0=np.ones(n))
    else:
        values, V = scipy.linalg.eigh(matrix, subset_by_index=[n - k, n - 1])
    return V[:, np.argsort(values)[::-1]]


def _top_ridges(A, B, k, shifts):
    """Top-k solutions of A w = l (B + s I) w for each shift s > 0, for symmetric A and positive
    semidefinite B, scaled as by _top_generalized (w'(B + s I) w = 1). B + s I has the
    eigenvectors of B, so one decomposition of B serves all shifts."""
    values, U = np.linalg.eigh((B + B.T) / 2)
    A = U.T @ ((A + A.T) / 2) @ U
    directions = []
    for s in shifts:
        d = 1 / np.sqrt(np.maximum(values, 0) + s)
        directions.append(U @ (d[:, None] * _top_symmetric(d[:, None] * A * d, k)))
    return directions


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
        A, B = heritability_design(estimator, groups, environment, subgroups).forms()
        A, B = Z.T @ A @ Z, Z.T @ B @ Z
        shifts = [ridge * np.trace(B) / X.shape[1] for ridge in ridges]
        positive = [s for s in shifts if s > 0]
        top = dict(zip(positive, _top_ridges(A, B, n_traits, positive), strict=True))
        fitted = []
        for ridge, shift in zip(ridges, shifts, strict=True):
            pch = cls(n_traits, ridge, estimator)
            pch.mean_ = start.mean_
            W = top[shift] if shift > 0 else _top_generalized(A, B, n_traits)
            fitted.append(pch._finish(Xc, V @ W))
        return fitted

    def _coordinates(self, X):
        """As LinearBaseline._coordinates, from the eigendecomposition of the smaller of the
        Gram matrix Xc Xc' and the scatter matrix Xc'Xc rather than the SVD of Xc (faster when the
        measurements far outnumber the individuals, or the reverse); directions with singular value
        at most 1e-6 of the largest are dropped."""
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        if Xc.shape[0] <= Xc.shape[1]:
            values, U = np.linalg.eigh(Xc @ Xc.T)
            keep = values > values.max() * 1e-12
            V = (Xc.T @ U[:, keep]) / np.sqrt(values[keep])
        else:
            values, V = np.linalg.eigh(Xc.T @ Xc)
            V = V[:, values > values.max() * 1e-12]
        return X, Xc, V, Xc @ V

    @classmethod
    def tune(cls, X, groups, environment, ridges=RIDGES, n_folds=5, seed=0, subgroups=None,
             estimator='anova', split_units=None):
        """Choose the ridge by the held-out heritability of the first trait: the procedure and the
        rule of H2-opt's noise level (h2opt.selection.cross_validate_levels on n_folds splits,
        best_level).

        For each fold of the groups (or of the split_units, labels of the individuals) and each
        ridge, the first PCH trait is fitted on the other folds and scored by its heritability
        (by the same estimator) on the held-out fold. X is one (n, p) array, or a list of arrays
        of the same individuals (e.g. one per date); then one ridge serves all of them, PCH is
        fitted to each, and the score is the mean over them.
        Returns (the ridge with the best mean score, the scores (len(ridges),) averaged over the
        folds, and the (n_folds, len(ridges)) fold scores).
        """
        sets = [np.asarray(x) for x in (X if isinstance(X, list | tuple) else [X])]
        groups = np.asarray(groups)
        environment = _as_environment(environment, len(groups))

        def part(labels, rows):
            return None if labels is None else np.asarray(labels)[rows]

        def fit(fit_rows, subset, ridge):
            # copies with the same training rows share one decomposition (fit_ridges)
            out = np.zeros((len(sets), len(ridge), len(groups)))
            same = {}
            for c, rows in enumerate(fit_rows):
                same.setdefault(rows.tobytes(), []).append(c)
            for copies in same.values():
                rows = fit_rows[copies[0]]
                for k, measurements in enumerate(sets):
                    fitted = cls.fit_ridges(measurements[rows], groups[rows], environment[rows],
                                            part(subgroups, rows), 1, list(ridge[copies]),
                                            estimator)
                    for c, pch in zip(copies, fitted, strict=True):
                        out[k, c] = pch.transform(measurements)[:, 0]
            return out

        scores, _ = cross_validate_levels(fit, groups, environment, ridges, None, n_folds, n_folds,
                                          seed, subgroups, estimator, split_units)
        fold_scores = scores[0].mean(axis=0)
        return best_level(ridges, fold_scores), fold_scores.mean(axis=0), fold_scores


class MaxHeritabilityFeatures:
    """Greedily select the n_traits most heritable measurements (features) on training individuals.

    After each pick, the chosen measurement is projected out of all others before the next pick.
    A measurement that is constant, or whose residual falls below 1e-8 of its centered norm (it
    lies in the span of the picks), is never picked. transform returns the selected measurements,
    centered and made uncorrelated on the training individuals.
    """

    def __init__(self, n_traits):
        self.n_traits = n_traits

    def fit(self, X, groups, environment=None):
        X = np.asarray(X, dtype=float)
        groups = np.asarray(groups)
        X_fit = X - X.mean(axis=0)
        norms = np.linalg.norm(X_fit, axis=0)
        alive = norms > 0
        chosen = []
        for _ in range(self.n_traits):
            heritability = np.zeros(X.shape[1])
            heritability[alive] = anova_heritability(X_fit[:, alive], groups, environment)
            heritability[np.isnan(heritability)] = 0
            heritability[~alive] = -np.inf
            best = int(np.argmax(heritability))
            chosen.append(best)

            u = np.copy(X_fit[:, best])
            X_fit = X_fit - np.outer(u, (u @ X_fit) / (u @ u))
            alive &= np.linalg.norm(X_fit, axis=0) > 1e-8 * norms
            alive[best] = False

        self.features_ = np.array(chosen)
        decorrelation = Decorrelation().fit(X[:, self.features_])
        self.mean_, self.components_ = decorrelation.mean_, decorrelation.components_
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float)[:, self.features_] - self.mean_) @ self.components_

    def fit_transform(self, X, groups, environment=None):
        return self.fit(X, groups, environment).transform(X)
