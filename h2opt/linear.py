"""Linear H2-opt trained to convergence with L-BFGS."""

import numpy as np
import torch

from .decorrelation import Decorrelation, orthonormal_basis, residualize
from .heritability import AnovaDesign, anova_heritability
from .selection import RIDGES, cross_validate


class _Spectrum:
    """Eigendecomposition of the training covariance, shared by fits with different ridges.

    X V (the measurements in the eigenbasis) is precomputed so that a trait X w with w = V z costs
    one n x m product.
    """

    def __init__(self, X):
        n = X.shape[0]
        Xc = X - X.mean(0)
        lam, self.V = torch.linalg.eigh(Xc.T @ Xc / n)
        self.lam = lam.clamp(min=0)
        self.feature_variance = lam.sum() / len(lam)
        self.XV = X @ self.V

    def scale(self, ridge):
        """Diagonal of the preconditioner (Cov(X) + ridge * v * I)^(-1/2) in the eigenbasis."""
        return 1 / torch.sqrt(self.lam + max(ridge, 1e-12) * self.feature_variance)


def _as_float64(X):
    return torch.as_tensor(np.asarray(X), dtype=torch.float64)


class LinearH2opt:
    """Linear synthetic traits y = X w that maximize ANOVA heritability, trained one at a time.

    Each trait maximizes gen(y) / (tot(y) + ridge * v * |w|^2) on the training individuals, where
    gen and tot are the genetic and total variance of anova_heritability and v is the mean variance
    of the measurements; with ridge -> 0 this is exactly the ANOVA heritability. In the objective,
    trait k is its residual after least-squares regression on traits 1..k-1 on the training
    individuals; the output traits are made uncorrelated on the training individuals by
    Decorrelation.
    Optimization runs in float64 with L-BFGS (strong Wolfe line search), over w = P u with P =
    (Cov(X) + ridge * v * I)^(-1/2): the same model and objective, better conditioned. Unlike train,
    this converges; n_iter (L-BFGS steps of up to 20 evaluations) only has to be large enough.

    The penalty equals the expected effect of independent input noise with standard deviation
    sigma = sqrt(ridge * v) on the objective (the noise adds sigma^2 |w|^2 to the total variance
    and nothing to the genetic variance), the data augmentation of train. Tune ridge on validation
    individuals (see tune).

    fit(X, groups, environment, validation=None): validation is an optional (X, groups, environment)
    tuple; the heritability of each trait on it after every step is stored in validation_curves_
    (n_traits, n_iter). feature_variance_ is v, so noise_sd_ = sqrt(ridge * v). weights_ is the
    (m, n_traits) matrix of the w's. transform(X) returns the (n, n_traits) traits.
    """

    def __init__(self, n_traits=1, ridge=1e-4, n_iter=200, seed=0):
        self.n_traits = n_traits
        self.ridge = ridge
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, groups, environment=None, validation=None, _spectrum=None):
        X = _as_float64(X)
        m = X.shape[1]
        spectrum = _spectrum if _spectrum is not None else _Spectrum(X)
        design = AnovaDesign(groups, environment)
        if validation is not None:
            val_XV = _as_float64(validation[0]) @ spectrum.V
            val_design = AnovaDesign(validation[1], validation[2])

        d = spectrum.scale(self.ridge)
        penalty_scale = self.ridge * spectrum.feature_variance
        self.feature_variance_ = float(spectrum.feature_variance)
        self.noise_sd_ = float(penalty_scale) ** 0.5
        self.validation_curves_ = np.zeros((self.n_traits, self.n_iter))
        # d * z of the fitted traits, one per column
        dz_fitted = torch.zeros((m, 0), dtype=torch.float64)
        for k in range(self.n_traits):
            basis = orthonormal_basis(spectrum.XV @ dz_fitted)
            # w = P u with u ~ N(0, I); in the eigenbasis w = V (d * z) with z = V' u
            torch.manual_seed(self.seed + k)
            z = (spectrum.V.T @ torch.randn(m, dtype=torch.float64)).requires_grad_(True)
            optimizer = torch.optim.LBFGS([z], lr=1, max_iter=20, history_size=100,
                                          tolerance_grad=0, tolerance_change=0,
                                          line_search_fn='strong_wolfe')

            # closure runs only inside this iteration's optimizer.step, so binding the loop
            # variables late is correct.
            def closure():
                optimizer.zero_grad()  # noqa: B023
                dz = d * z  # noqa: B023
                y = residualize((spectrum.XV @ dz)[:, None], basis)  # noqa: B023
                genetic, total = design.heritability(y, return_variance=True)
                loss = -(genetic / (total + penalty_scale * (dz @ dz))).sum()
                loss.backward()
                return loss

            for step in range(self.n_iter):
                optimizer.step(closure)
                if validation is not None:
                    with torch.no_grad():
                        dz = torch.cat([dz_fitted, (d * z)[:, None]], 1)
                        traits = Decorrelation().fit(spectrum.XV @ dz).transform(val_XV @ dz)
                        self.validation_curves_[k, step] = val_design.heritability(
                            torch.tensor(traits[:, -1:])).item()

            dz_fitted = torch.cat([dz_fitted, (d * z).detach()[:, None]], 1)
        self.weights_ = (spectrum.V @ dz_fitted).numpy()
        self.decorrelation_ = Decorrelation().fit(spectrum.XV @ dz_fitted)
        return self

    def transform(self, X):
        return self.decorrelation_.transform(np.asarray(X, dtype=np.float64) @ self.weights_)

    def fit_transform(self, X, groups, environment=None, validation=None):
        return self.fit(X, groups, environment, validation).transform(X)

    @classmethod
    def tune(cls, X, groups, environment, n_traits=1, ridges=RIDGES, n_folds=5, n_iter=200,
             seed=0):
        """Choose one ridge for all traits by cross-validated heritability at convergence.

        For each fold of the groups (h2opt.selection.cross_validate) and each ridge, n_traits
        traits are fitted for n_iter steps on the other folds and scored by their mean
        heritability on the held-out fold. Returns (ridge with the best mean score, scores
        (len(ridges),), fold_scores (n_folds, len(ridges))).
        """
        def score_fold(X_fit, groups_fit, environment_fit, X_val, groups_val, environment_val):
            spectrum = _Spectrum(_as_float64(X_fit))
            scores = []
            for ridge in ridges:
                model = cls(n_traits, ridge=ridge, n_iter=n_iter, seed=seed)
                model.fit(X_fit, groups_fit, environment_fit, _spectrum=spectrum)
                scores.append(anova_heritability(model.transform(X_val), groups_val,
                                                 environment_val).mean())
            return scores

        scores, fold_scores = cross_validate(X, groups, environment, score_fold, n_folds, seed)
        return ridges[int(np.argmax(scores))], scores, fold_scores
