"""Nearest-neighbor field adjustment of traits (Papadakis 1937).

In a field trial, the mean trait of the neighbors of an individual (other plots near it, of
other groups) estimates its local field effect. The adjustment removes the part of the trait that
this neighbor mean predicts: y - b (x - mean x), with x the neighbor mean and b the least-squares
slope of y on x. b is computed from the trait, not chosen, so a trait without a field effect
(b near 0) is left almost unchanged. The adjustment is differentiable in y, so H2-opt can
maximize the heritability of the adjusted trait (h2opt.train, argument neighbors; there with
b = 1, see papadakis_torch).
"""

import numpy as np


def papadakis(Y, neighbors, rows):
    """Adjusted traits: Y (n,) or (n, k), neighbors (n, n) nonnegative weights (neighbors[i, j] > 0
    if j is a neighbor of i), rows (n,) boolean, the individuals that b is fitted on. The neighbor
    mean of i is the weighted mean of Y over its neighbors (all individuals, NaN values left out).
    Returns the adjusted Y, NaN where Y is NaN."""
    Y = np.asarray(Y, dtype=float)
    one = Y.ndim == 1
    Y = Y[:, None] if one else Y
    have = np.isfinite(Y)
    weights = np.asarray(neighbors, dtype=float)
    x = (weights @ np.where(have, Y, 0)) / (weights @ have)
    fit = np.asarray(rows, dtype=bool)[:, None] & have & np.isfinite(x)
    count = fit.sum(0)
    xc = np.where(fit, x - np.where(fit, x, 0).sum(0) / count, 0)
    yc = np.where(fit, Y - np.where(fit, Y, 0).sum(0) / count, 0)
    b = (xc * yc).sum(0) / (xc ** 2).sum(0)
    adjusted = Y - b * (x - np.where(fit, x, 0).sum(0) / count)
    return adjusted[:, 0] if one else adjusted


def papadakis_torch(Y, neighbors, mask):
    """The adjustment of each column of Y (n, k) for training, with the slope b fixed at 1: Y minus
    its centered neighbor mean. mask (n, k): 1 for the individuals of copy k (its training
    individuals); the neighbor mean of copy k uses only its individuals, so other individuals
    never enter its loss. neighbors: (n, n) tensor of nonnegative weights. Y is centered on the
    rows of each copy. Returns the adjusted Y, zero outside each copy's rows.

    b is fixed because a b fitted from the trait makes the heritability of the adjusted trait a
    function with local optima, which gradient ascent does not escape. With b = 1 the adjustment
    is linear, so the heritability stays a ratio of quadratic forms of the trait. Scores use the
    fitted b (papadakis)."""
    x = (neighbors @ (Y * mask)) / (neighbors @ mask).clamp(min=1e-12)
    xc = (x - (x * mask).sum(0) / mask.sum(0)) * mask
    return (Y - xc) * mask
