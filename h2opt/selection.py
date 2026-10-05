"""Choosing a regularization strength by heritability on held-out groups.

H2-opt's noise level (select_noise_level), one for all traits: the level whose traits have the
highest mean held-out heritability over validation folds of the groups. PCH penalizes
ridge * v * |w|^2 (v the mean variance of the measurements); its ridge (PCH.tune, through
cross_validate) is chosen by the same rule, from the matching grid. Also the assignment of groups
to folds.
"""

import copy

import numpy as np
import torch

from .decorrelation import Decorrelation
from .heritability import _as_environment, anova_heritability
from .train import train_batch

# Ridges of PCH; ridge 1 is the penalty of input noise with the standard deviation of a typical
# measurement.
RIDGES = tuple(float(r) for r in np.logspace(-6, 2, 17))

# Noise levels of H2-opt: the standard deviation of the normal input noise as a fraction of
# noise_scale(X), the typical spread of a measurement. Level sqrt(r) has the penalty of ridge r,
# so the two grids match.
NOISE_LEVELS = tuple(float(np.sqrt(r)) for r in RIDGES)


def per_group(groups, draw):
    """One value per group, given to every individual of the group.

    draw(c) returns the values of the c groups, in the sorted order of their labels."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    return np.asarray(draw(inverse.max() + 1))[inverse]


def group_folds(groups, n_folds=5, seed=0):
    """Fold (0..n_folds-1) of each individual; all individuals of a group share a fold, and the
    groups are spread evenly over the folds in a random order."""
    return per_group(groups, lambda c: np.random.RandomState(seed).permutation(c) % n_folds)


def cross_validate(X, groups, environment, score_fold, n_folds=5, seed=0):
    """Scores of a set of settings, per fold of the groups.

    score_fold(X_fit, groups_fit, environment_fit, X_val, groups_val, environment_val) returns
    one score per setting. Returns (scores averaged over the folds, (n_folds, n_settings) scores).
    """
    X = np.asarray(X)
    groups = np.asarray(groups)
    environment = _as_environment(environment, len(groups))
    fold = group_folds(groups, n_folds, seed)
    fold_scores = np.array([
        score_fold(X[fold != f], groups[fold != f], environment[fold != f],
                   X[fold == f], groups[fold == f], environment[fold == f])
        for f in range(n_folds)])
    return fold_scores.mean(axis=0), fold_scores


def noise_scale(X):
    """Root of the mean variance of the measurements (columns of the (n, m) array X)."""
    return float(np.sqrt(np.mean(np.var(np.asarray(X, dtype=float), axis=0))))


def select_noise_level(make_model, X, groups, environment, n_traits, subsets=None,
                       levels=NOISE_LEVELS, n_splits=5, n_folds=5, seed=0, **train_options):
    """Noise level of H2-opt, one for all traits, by the held-out heritability of the traits.

    One selection runs on each subset of the individuals (subsets: (k, n) boolean, e.g. the
    training individuals of k outer folds; default: one selection on all). Within a subset, the
    groups are assigned to n_folds folds (group_folds); each of the first n_splits folds (default:
    all) is the held-out fold of one validation split. In each split and for each level, traits
    0..n_traits-1 are trained in order on the other folds, as in train: a fresh model per trait
    (make_model(), a module mapping (n, m) to (n, 1)), normal noise of standard deviation
    level * noise_scale(X) (X of the subset), in the loss the trait's residual on the earlier
    traits of the same split and level. The score of a trait is the ANOVA heritability on the
    held-out fold of the trait made uncorrelated with the earlier traits on the training folds
    (Decorrelation, as in synthetic_traits). The chosen level has the best score averaged over the
    traits and splits, as for the PCH ridge (PCH.tune). All subsets, splits and levels of one
    trait are trained together by train_batch; train_options go to it (e.g. n_iter, device).
    Returns (the chosen level of each subset (k,), (k, n_traits, n_splits, n_levels) scores, and
    the trained traits before decorrelation, (k, n_traits, n_splits, n_levels, n) float32, on all
    n individuals). The traits of a level do not depend on the other levels, so the traits and
    scores give the choice of any other rule over the levels without training again.
    """
    X = np.asarray(X)
    groups = np.asarray(groups)
    environment = _as_environment(environment, len(groups))
    subsets = np.ones((1, len(groups)), dtype=bool) if subsets is None else np.asarray(subsets)
    n_subsets = len(subsets)
    # copies: (subset, split, level), with the training and held-out individuals of the split
    copies, fit, held_out, sd = [], [], [], []
    for i, subset in enumerate(subsets):
        fold = np.full(len(groups), -1)
        fold[subset] = group_folds(groups[subset], n_folds, seed)
        scale = noise_scale(X[subset])
        for split in range(n_splits):
            for a, level in enumerate(levels):
                copies.append((i, split, a))
                fit.append(subset & (fold != split))
                held_out.append(fold == split)
                sd.append(level * scale)
    x = torch.as_tensor(X, dtype=torch.float32)
    earlier = [np.zeros((len(groups), 0)) for _ in copies]
    scores = np.zeros((n_subsets, n_traits, n_splits, len(levels)))
    traits = np.zeros((n_subsets, n_traits, n_splits, len(levels), len(groups)), dtype=np.float32)
    for t in range(n_traits):
        torch.manual_seed(seed + t)
        start = make_model()
        models = [copy.deepcopy(start) for _ in copies]
        train_batch(models, X, groups, environment, fit, sd, earlier, noise='normal',
                    **train_options)
        for c, (model, (i, split, a)) in enumerate(zip(models, copies, strict=True)):
            model.cpu()
            with torch.no_grad():
                output = model(x)[:, 0].double().numpy()
            traits[i, t, split, a] = output
            earlier[c] = np.column_stack([earlier[c], output])
            trait = Decorrelation().fit(earlier[c][fit[c]]).transform(earlier[c])[:, -1:]
            scores[i, t, split, a] = anova_heritability(trait[held_out[c]], groups[held_out[c]],
                                                        environment[held_out[c]])[0]
    chosen = np.array([levels[int(np.argmax(scores[i].mean(axis=(0, 1))))]
                       for i in range(n_subsets)])
    return chosen, scores, traits
