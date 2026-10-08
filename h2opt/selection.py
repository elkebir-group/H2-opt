"""Choosing a regularization strength by heritability on held-out groups.

H2-opt's noise level (select_noise_level), one for all traits: the level whose traits have the
highest mean held-out heritability over validation folds of the groups (h2opt.folds). The same
for the convolutional branch of LinearConvModel (score_linear_conv_levels), with the noise of the
linear branch given.
"""

import copy

import numpy as np
import torch

from .decorrelation import Decorrelation
from .folds import group_folds
from .heritability import _as_environment, heritability
from .train import synthetic_traits, train_batch, train_linear_conv_models

# Noise levels of H2-opt: the standard deviation of the normal input noise as a fraction of
# noise_scale(X), the typical spread of a measurement; 0.001 to 10 in quarter-decade steps.
NOISE_LEVELS = tuple(float(s) for s in np.logspace(-3, 1, 17))


def noise_scale(X):
    """Root of the mean variance of the measurements (columns of the (n, m) array X)."""
    return float(np.sqrt(np.mean(np.var(np.asarray(X, dtype=float), axis=0))))


def best_level(levels, scores):
    """The level with the best score averaged over all axes but the last (scores: (..., levels)),
    the rule of select_noise_level. With the scores of levels trained in separate runs, it gives
    the choice of one run over all of them."""
    scores = np.asarray(scores)
    return levels[int(np.argmax(scores.reshape((-1, scores.shape[-1])).mean(axis=0)))]


def select_noise_level(make_model, X, groups, environment, n_traits, subsets=None,
                       levels=NOISE_LEVELS, n_splits=5, n_folds=5, seed=0, subgroups=None,
                       estimator='anova', split_units=None, **train_options):
    """Noise of H2-opt, one level for all traits, by the held-out heritability of the traits.

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
    averaged over the traits and splits (best_level). X is one array of
    measurements (first axis: individuals), or a list of arrays of the same individuals (e.g. one
    per date): then every set is trained and scored, and one level serves all sets by their mean
    score. All subsets, splits and levels of one trait (and set) are trained
    together by train_batch; train_options go to it (e.g. n_iter, device, max_copies).
    Returns (noise_sd of each subset (k,), the chosen level times noise_scale(X) of the subset, for
    train; (k, n_traits, n_splits, n_levels) scores; and the trained traits before decorrelation,
    (k, n_traits, n_splits, n_levels, n) float32, on all n individuals); with a list of sets,
    noise_sd (one per set: (k, sets)), scores and traits have a set axis after the first. The
    chosen level of subset i is best_level(levels, scores[i]). The traits of a level do not depend
    on the other levels, so the traits and scores give the choice of any other rule over the
    levels without training again.
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
    scales = np.zeros((n_subsets, len(sets)))
    traits = np.zeros(shape + (len(groups),), dtype=np.float32)
    for k, measurements in enumerate(sets):
        measurements = np.asarray(measurements)
        x = torch.as_tensor(measurements, dtype=torch.float32)
        scale = [noise_scale(measurements[subset].reshape((subset.sum(), -1)))
                 for subset in subsets]
        scales[:, k] = scale
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
    chosen = np.array([best_level(levels, scores[i]) for i in range(n_subsets)])
    noise_sd = chosen[:, None] * scales
    if not isinstance(X, list | tuple):
        noise_sd, scores, traits = noise_sd[:, 0], scores[:, 0], traits[:, 0]
    return noise_sd, scores, traits


def score_linear_conv_levels(make_conv, X, groups, environment, n_traits, linear_noise_sd, levels,
                             subsets=None, n_splits=5, n_folds=5, seed=0, subgroups=None,
                             estimator='anova', split_units=None, **train_options):
    """Held-out heritability of LinearConvModel traits at each noise level of the convolutional
    branch, on validation splits as in select_noise_level.

    Within each subset of the individuals (subsets: (k, n) boolean, e.g. the training individuals
    of k outer folds; default: one subset of all), the groups (or split_units) are assigned to
    n_folds folds (group_folds), and each of the first n_splits folds is the held-out fold of one
    validation split. For each split and level, train_linear_conv_models trains traits
    0..n_traits-1 on the other folds of the subset: linear H2-opt with noise of standard deviation
    linear_noise_sd of the subset (length k, absolute standard deviations, e.g. noise_sd of
    select_noise_level), then the LinearConvModel with convolutional noise of standard deviation
    level * noise_scale(X) (X of the subset, flattened). The score of a trait is its heritability
    on the held-out fold, after it is made uncorrelated with the earlier traits on the training
    folds (synthetic_traits). One level per call is enough: the scores of a level do not depend on
    the other levels, so levels scored in separate runs can be compared (best_level). All copies
    (subset, split, level) are trained together; train_options go to train_linear_conv_models
    (e.g. n_linear_iter, n_iter, device, max_copies). Returns ((k, n_traits, n_splits, n_levels)
    scores, and the (k, n_traits, n_splits, n_levels, n) float32 traits of all n individuals,
    uncorrelated on the training folds).
    """
    X = np.asarray(X)
    groups = np.asarray(groups)
    environment = _as_environment(environment, len(groups))
    subgroups = None if subgroups is None else np.asarray(subgroups)
    units = groups if split_units is None else np.asarray(split_units)
    subsets = np.ones((1, len(groups)), dtype=bool) if subsets is None else np.asarray(subsets)
    linear_noise_sd = np.broadcast_to(np.asarray(linear_noise_sd, dtype=float), (len(subsets),))
    flat = X.reshape((len(X), -1))
    copies, train_test, linear_sd, conv_sd = [], [], [], []
    for i, subset in enumerate(subsets):
        fold = np.full(len(groups), -1)
        fold[subset] = group_folds(units[subset], n_folds, seed)
        scale = noise_scale(flat[subset])
        for split in range(n_splits):
            for a, level in enumerate(levels):
                copies.append((i, split, a))
                # 0: training folds, 1: held-out fold, 2: outside the subset
                train_test.append(np.where(subset & (fold != split), 0,
                                           np.where(fold == split, 1, 2)))
                linear_sd.append(linear_noise_sd[i])
                conv_sd.append(level * scale)
    torch.manual_seed(seed)
    models = train_linear_conv_models(
        make_conv, X, groups, environment, np.array(train_test),
        np.array(linear_sd)[:, None], np.array(conv_sd)[:, None], n_traits=n_traits,
        subgroups=subgroups, estimator=estimator, verbose=False, **train_options)
    scores = np.zeros((len(subsets), n_traits, n_splits, len(levels)))
    traits = np.zeros(scores.shape + (len(groups),), dtype=np.float32)
    for model, rows, (i, split, a) in zip(models, train_test, copies, strict=True):
        Y = synthetic_traits(model, X, rows == 0)
        held_out = rows == 1
        scores[i, :, split, a] = heritability(
            Y[held_out], groups[held_out], environment[held_out],
            None if subgroups is None else subgroups[held_out], estimator)
        traits[i, :, split, a] = Y.T
    return scores, traits
