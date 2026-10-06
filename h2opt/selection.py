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
from .heritability import _as_environment, heritability
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


def cross_validate(units, score_fold, n_folds=5, seed=0):
    """Scores of a set of settings, per fold of the units (labels of the individuals, e.g. their
    groups; group_folds keeps the individuals of a unit together).

    score_fold(fit, val), with boolean masks of the training and held-out individuals, returns one
    score per setting. Returns (scores averaged over the folds, (n_folds, n_settings) scores).
    """
    fold = group_folds(np.asarray(units), n_folds, seed)
    fold_scores = np.array([score_fold(fold != f, fold == f) for f in range(n_folds)])
    return fold_scores.mean(axis=0), fold_scores


def noise_scale(X):
    """Root of the mean variance of the measurements (columns of the (n, m) array X)."""
    return float(np.sqrt(np.mean(np.var(np.asarray(X, dtype=float), axis=0))))


def select_noise_level(make_model, X, groups, environment, n_traits, subsets=None,
                       levels=NOISE_LEVELS, n_splits=5, n_folds=5, seed=0, subgroups=None,
                       estimator='anova', split_units=None, **train_options):
    """Noise level of H2-opt, one for all traits, by the held-out heritability of the traits.

    One selection runs on each subset of the individuals (subsets: (k, n) boolean, e.g. the training
    individuals of k outer folds; default: one selection on all). Within a subset, the groups (or
    the split_units, labels of the individuals; e.g. np.arange(n) to split individuals) are assigned
    to n_folds folds (group_folds); each of the first n_splits folds (default: all) is the held-out
    fold of one validation split. In each split and for each level, traits 0..n_traits-1 are trained
    in order on the other folds, as in train: a fresh model per trait (make_model(), a module
    mapping the measurements of n individuals to (n, 1)), normal noise of standard deviation level *
    noise_scale(X) (X of the subset), in the loss the trait's residual on the earlier traits of the
    same split and level. The score of a trait is its heritability (estimator and subgroups as in
    train) on the held-out fold, after it is made uncorrelated with the earlier traits on the
    training folds (Decorrelation, as in synthetic_traits). The chosen level has the best score
    averaged over the traits and splits, as for the PCH ridge (PCH.tune). X is one array of
    measurements (first axis: individuals), or a list of arrays of the same individuals (e.g. one
    per date): then every set is trained and scored, and one level serves all sets by their mean
    score, as in PCH.tune. All subsets, splits and levels of one trait (and set) are trained
    together by train_batch; train_options go to it (e.g. n_iter, device, max_copies).
    Returns (the chosen level of each subset (k,), (k, n_traits, n_splits, n_levels) scores, and
    the trained traits before decorrelation, (k, n_traits, n_splits, n_levels, n) float32, on all
    n individuals); with a list of sets, scores and traits have a set axis after the first. The
    traits of a level do not depend on the other levels, so the traits and scores give the choice
    of any other rule over the levels without training again.
    """
    sets = X if isinstance(X, list | tuple) else [X]
    groups = np.asarray(groups)
    environment = _as_environment(environment, len(groups))
    subgroups = None if subgroups is None else np.asarray(subgroups)
    units = groups if split_units is None else np.asarray(split_units)
    subsets = np.ones((1, len(groups)), dtype=bool) if subsets is None else np.asarray(subsets)
    n_subsets = len(subsets)
    # copies: (subset, split, level), with the training and held-out individuals of the split
    copies, fit, held_out = [], [], []
    for i, subset in enumerate(subsets):
        fold = np.full(len(groups), -1)
        fold[subset] = group_folds(units[subset], n_folds, seed)
        for split in range(n_splits):
            for a in range(len(levels)):
                copies.append((i, split, a))
                fit.append(subset & (fold != split))
                held_out.append(fold == split)
    shape = (n_subsets, len(sets), n_traits, n_splits, len(levels))
    scores = np.zeros(shape)
    traits = np.zeros(shape + (len(groups),), dtype=np.float32)
    for k, measurements in enumerate(sets):
        measurements = np.asarray(measurements)
        x = torch.as_tensor(measurements, dtype=torch.float32)
        scale = [noise_scale(measurements[subset].reshape((subset.sum(), -1)))
                 for subset in subsets]
        sd = [levels[a] * scale[i] for i, _, a in copies]
        earlier = [np.zeros((len(groups), 0)) for _ in copies]
        for t in range(n_traits):
            torch.manual_seed(seed + t)
            start = make_model()
            models = [copy.deepcopy(start) for _ in copies]
            train_batch(models, measurements, groups, environment, fit, sd, earlier,
                        noise='normal', subgroups=subgroups, estimator=estimator,
                        **train_options)
            for c, (model, (i, split, a)) in enumerate(zip(models, copies, strict=True)):
                model.cpu()
                with torch.no_grad():
                    output = model(x)[:, 0].double().numpy()
                traits[i, k, t, split, a] = output
                earlier[c] = np.column_stack([earlier[c], output])
                trait = Decorrelation().fit(earlier[c][fit[c]]).transform(earlier[c])[:, -1:]
                rows = held_out[c]
                scores[i, k, t, split, a] = heritability(
                    trait[rows], groups[rows], environment[rows],
                    None if subgroups is None else subgroups[rows], estimator)[0]
    chosen = np.array([levels[int(np.argmax(scores[i].mean(axis=(0, 1, 2))))]
                       for i in range(n_subsets)])
    if not isinstance(X, list | tuple):
        scores, traits = scores[:, 0], traits[:, 0]
    return chosen, scores, traits
