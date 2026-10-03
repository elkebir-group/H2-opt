"""Linear H2-opt trained to convergence with L-BFGS."""

import numpy as np
import torch

from .heritability import AnovaDesign


def _residualize(y, previous, rows):
    """Least-squares residual of y on the earlier traits, coefficients from the given rows.

    Returns the residual, the mean of the earlier traits on the rows and the coefficients.
    """
    if previous.shape[1] == 0:
        return y, None, None
    mean = previous[rows].mean(0)
    centered = previous - mean
    coef = torch.linalg.lstsq(centered[rows], y[rows] - y[rows].mean())[0]
    return y - centered @ coef, mean, coef


def _apply(X, w, mean, coef, previous):
    """Trait X w residualized on the earlier traits with coefficients fitted on the training individuals."""
    y = (X @ w)[:, None]
    if mean is not None:
        y = y - (previous - mean) @ coef
    return y


class LinearH2opt:
    """Linear synthetic traits y = X w maximizing ANOVA heritability, trained one at a time with L-BFGS.

    Each trait maximizes gen(y) / (tot(y) + ridge * v * |w|^2) on the training individuals, where gen and tot are
    the genetic and total variance of anova_heritability and v is the mean variance of the measurements; with
    ridge -> 0 this is exactly the ANOVA heritability. Trait k is made uncorrelated with traits 1..k-1 on the
    training individuals (least-squares residual). Optimization runs in float64 with L-BFGS (strong Wolfe line
    search), over w = P u with P = (Cov(X) + ridge * v * I)^(-1/2): the same model and objective, better
    conditioned. Unlike train, this converges, so n_iter (L-BFGS steps of up to 20 evaluations) acts as a
    stopping point to tune, together with ridge, on validation individuals.

    fit(X, groups, environment, validation=None): validation is an optional (X, groups, environment) tuple;
    the heritability of each trait on it after every step is stored in validation_curves_ (n_traits, n_iter).
    transform(X) returns the (n, n_traits) traits.
    """

    def __init__(self, n_traits=1, ridge=1e-4, n_iter=100, seed=0):
        self.n_traits = n_traits
        self.ridge = ridge
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, groups, environment=None, validation=None):
        X = torch.tensor(np.asarray(X), dtype=torch.float64)
        n, m = X.shape
        rows = np.arange(n)
        design = AnovaDesign(groups, environment)
        if validation is not None:
            X_val = torch.tensor(np.asarray(validation[0]), dtype=torch.float64)
            val_design = AnovaDesign(validation[1], validation[2])

        Xc = X - X.mean(0)
        lam, V = torch.linalg.eigh(Xc.T @ Xc / n)
        self.feature_variance_ = lam.sum() / len(lam)
        P = V @ torch.diag(1 / torch.sqrt(lam.clamp(min=0) + max(self.ridge, 1e-12) * self.feature_variance_)) @ V.T

        self.weights_, self.means_, self.coefs_ = [], [], []
        self.validation_curves_ = np.zeros((self.n_traits, self.n_iter))
        traits = torch.zeros((n, 0), dtype=torch.float64)
        for k in range(self.n_traits):
            torch.manual_seed(self.seed + k)
            u = torch.randn(m, dtype=torch.float64).requires_grad_(True)
            optimizer = torch.optim.LBFGS([u], lr=1, max_iter=20, history_size=100, tolerance_grad=0,
                                          tolerance_change=0, line_search_fn='strong_wolfe')

            def closure():
                optimizer.zero_grad()
                w = P @ u
                y, _, _ = _residualize((X @ w)[:, None], traits, rows)
                genetic, total = design.heritability(y, return_variance=True)
                loss = -(genetic / (total + self.ridge * self.feature_variance_ * (w @ w))).sum()
                loss.backward()
                return loss

            if validation is not None:
                previous_val = torch.tensor(self.transform(X_val))
            for step in range(self.n_iter):
                optimizer.step(closure)
                if validation is not None:
                    with torch.no_grad():
                        w = P @ u
                        _, mean, coef = _residualize((X @ w)[:, None], traits, rows)
                        y = _apply(X_val, w, mean, coef, previous_val)
                        self.validation_curves_[k, step] = val_design.heritability(y).item()

            with torch.no_grad():
                w = (P @ u).detach()
                y, mean, coef = _residualize((X @ w)[:, None], traits, rows)
                self.weights_.append(w)
                self.means_.append(mean)
                self.coefs_.append(coef)
                traits = torch.cat([traits, y], 1)
        return self

    def transform(self, X, n_traits=None):
        X = torch.tensor(np.asarray(X), dtype=torch.float64)
        traits = torch.zeros((X.shape[0], 0), dtype=torch.float64)
        for w, mean, coef in list(zip(self.weights_, self.means_, self.coefs_))[:n_traits]:
            traits = torch.cat([traits, _apply(X, w, mean, coef, traits)], 1)
        return traits.numpy()

    def fit_transform(self, X, groups, environment=None, validation=None):
        return self.fit(X, groups, environment, validation).transform(X)
