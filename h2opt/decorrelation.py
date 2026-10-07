"""Uncorrelated synthetic traits: each trait minus its least-squares fit on the traits before it."""

import numpy as np
import torch


class Decorrelation:
    """Make traits uncorrelated on the individuals they are fitted on, in order.

    fit(Y) takes the (n, k) traits of the fitting individuals (e.g. the training individuals).
    transform(Y) centers traits of any individuals by the fitting mean and replaces trait j by its
    residual after least-squares regression on traits 1..j-1, with the coefficients of the fitting
    individuals. On the fitting individuals the result has uncorrelated columns, and each trait
    keeps the scale of its own component. components_ is the (k, k) upper triangular matrix of the
    map: transform(Y) = (Y - mean_) @ components_.
    """

    def fit(self, Y):
        Y = np.asarray(Y, dtype=np.float64)
        self.mean_ = Y.mean(axis=0)
        # Y - mean = Q R; column j of Q R_jj is the residual of trait j on the traits before it
        _, R = np.linalg.qr(Y - self.mean_)
        self.components_ = np.linalg.inv(R) @ np.diag(np.diag(R))
        return self

    def transform(self, Y):
        return (np.asarray(Y, dtype=np.float64) - self.mean_) @ self.components_

    def fit_transform(self, Y):
        return self.fit(Y).transform(Y)


def orthonormal_basis(B):
    """Orthonormal basis of the space spanned by the centered columns of the (n, k) tensor B.

    k may be 0."""
    if B.shape[1] == 0:
        return B
    return torch.linalg.qr(B - B.mean(axis=0))[0]


