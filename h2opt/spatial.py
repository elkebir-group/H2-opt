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
    An individual without neighbors (none with a finite value) is left unadjusted and is not in
    the fit of b. Returns the adjusted Y, NaN where Y is NaN."""
    Y = np.asarray(Y, dtype=float)
    one = Y.ndim == 1
    Y = Y[:, None] if one else Y
    have = np.isfinite(Y)
    weights = np.asarray(neighbors, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        x = (weights @ np.where(have, Y, 0)) / (weights @ have)
    fit = np.asarray(rows, dtype=bool)[:, None] & have & np.isfinite(x)
    count = fit.sum(0)
    xc = np.where(fit, x - np.where(fit, x, 0).sum(0) / count, 0)
    yc = np.where(fit, Y - np.where(fit, Y, 0).sum(0) / count, 0)
    b = (xc * yc).sum(0) / (xc ** 2).sum(0)
    shift = b * (x - np.where(fit, x, 0).sum(0) / count)
    adjusted = Y - np.where(np.isfinite(x), shift, 0)
    return adjusted[:, 0] if one else adjusted


def papadakis_torch(Y, neighbors, mask):
    """The adjustment of each column of Y (n, k) for training, with the slope b fixed at 1: Y minus
    its centered neighbor mean. mask (n, k): 1 for the individuals of copy k (its training
    individuals); the neighbor mean of copy k uses only its individuals, so other individuals
    never enter its loss. neighbors: (n, n) tensor of nonnegative weights. An individual without
    neighbors among the copy's individuals is left unadjusted, as in papadakis. Y is centered on
    the rows of each copy. Returns the adjusted Y, zero outside each copy's rows.

    b is fixed because a b fitted from the trait makes the heritability of the adjusted trait a
    function with local optima, which gradient ascent does not escape. With b = 1 the adjustment
    is linear, so the heritability stays a ratio of quadratic forms of the trait. Scores use the
    fitted b (papadakis)."""
    weight = neighbors @ mask
    has = mask * (weight > 0)
    x = (neighbors @ (Y * mask)) / weight.clamp(min=1e-12)
    xc = (x - (x * has).sum(0) / has.sum(0).clamp(min=1)) * has
    return (Y - xc) * mask
