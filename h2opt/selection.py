"""Choosing a regularization level by the held-out heritability of the first trait.

One procedure for every method (cross_validate_levels): the groups of each subset of the
individuals are split into validation folds; the method fits its first trait at each level on
the other folds, and the score is its heritability on the held-out fold. Only the first trait is
scored: a mean over all traits rewards more regularization, which spreads the heritable signal
over more traits (the first traits lose heritability and the later traits gain it). The first
trait also needs no earlier traits, so only it is fitted.

Two rules choose from the scores, one for each kind of choice:
- best_level, for the strength of one model's regularization: H2-opt's noise
  (select_noise_level) and PCH's ridge (h2opt.baselines.PCH.tune).
- highest_level_within_one_se, for the convolutional branch of LinearConvModel
  (score_linear_conv_levels): a high level makes the branch add almost nothing, so the choice is
  between the linear model and the linear model plus a CNN, and the simpler model is kept unless
  the CNN raises the held-out heritability by more than one standard error.
"""

import copy

import numpy as np
import torch

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
    the rule for the strength of one model's regularization. With the scores of levels from
    separate runs stacked on the last axis, it gives the choice of one run over all of them."""
    scores = np.asarray(scores)
    return levels[int(np.argmax(scores.reshape((-1, scores.shape[-1])).mean(axis=0)))]


def highest_level_within_one_se(levels, scores):
    """The highest level whose mean score is within one standard error of the best mean score,
    the rule for the convolutional branch of LinearConvModel.

    scores: (..., splits, levels); the axes before the splits (e.g. measurement sets) are
    averaged first. The standard error is that of the best level's mean over the splits."""
    levels = np.asarray(levels, dtype=float)
    scores = np.asarray(scores)
    per_split = scores.reshape((-1,) + scores.shape[-2:]).mean(axis=0)
    mean = per_split.mean(axis=0)
    best = int(np.argmax(mean))
    se = per_split[:, best].std(ddof=1) / np.sqrt(len(per_split))
    return float(levels[mean >= mean[best] - se].max())


def cross_validate_levels(fit, groups, environment, levels, subsets=None, n_splits=5, n_folds=5,
                          seed=0, subgroups=None, estimator='anova', split_units=None):
    """Held-out heritability of a method's first trait at each level, on validation splits.

    Within each subset of the individuals (subsets: (k, n) boolean, e.g. the training individuals
    of k outer folds; default: one subset of all), the groups (or the split_units, labels of the
    individuals; e.g. np.arange(n) to split individuals) are assigned to n_folds folds
    (group_folds with seed); each of the first n_splits folds is the held-out fold of one
    validation split. A copy is one (subset, split, level). fit(fit_rows, subset, level) fits the
    method once per copy, all copies in one call so that a method can batch them: fit_rows
    (copies, n) boolean, the training individuals of each copy; subset and level (copies,), the
    index of its subset and its level. It returns the first trait of each copy for all n
    individuals, (copies, n), or (sets, copies, n) for several measurement sets of the same
    individuals. The score of a copy is the heritability of its trait on its held-out fold
    (estimator and subgroups as in h2opt.train). Returns ((k, n_splits, n_levels) scores and the
    (k, n_splits, n_levels, n) float32 traits; with sets, a set axis after the first).
    """
    groups = np.asarray(groups)
    environment = _as_environment(environment, len(groups))
    subgroups = None if subgroups is None else np.asarray(subgroups)
    units = groups if split_units is None else np.asarray(split_units)
    subsets = np.ones((1, len(groups)), dtype=bool) if subsets is None else np.asarray(subsets)
    copies, fit_rows, held_out = [], [], []
    for i, subset in enumerate(subsets):
        fold = np.full(len(groups), -1)
        fold[subset] = group_folds(units[subset], n_folds, seed)
        for split in range(n_splits):
            for a in range(len(levels)):
                copies.append((i, split, a))
                fit_rows.append(subset & (fold != split))
                held_out.append(fold == split)
    traits = np.asarray(fit(np.array(fit_rows), np.array([i for i, _, _ in copies]),
                            np.array([levels[a] for _, _, a in copies], dtype=float)))
    one_set = traits.ndim == 2
    traits = traits.reshape((-1, len(copies), len(groups)))
    shape = (len(subsets), len(traits), n_splits, len(levels))
    scores = np.zeros(shape)
    out = np.zeros(shape + (len(groups),), dtype=np.float32)
    for k, set_traits in enumerate(traits):
        for c, (i, split, a) in enumerate(copies):
            rows = held_out[c]
            scores[i, k, split, a] = heritability(
                set_traits[c][rows, None], groups[rows], environment[rows],
                None if subgroups is None else subgroups[rows], estimator)[0]
            out[i, k, split, a] = set_traits[c]
    if one_set:
        scores, out = scores[:, 0], out[:, 0]
    return scores, out


def select_noise_level(make_model, X, groups, environment, subsets=None, levels=NOISE_LEVELS,
                       n_splits=5, n_folds=5, seed=0, subgroups=None, estimator='anova',
                       split_units=None, **train_options):
    """Noise of H2-opt, one level for all traits: cross_validate_levels and best_level.

    The method fits the first trait as in train: a model make_model() (a module mapping the
    measurements of n individuals to (n, 1)) with normal noise of standard deviation level *
    noise_scale(X) (X of the subset), all copies trained together by train_batch (train_options go
    to it, e.g. n_iter, device, max_copies). X is one array of measurements (first axis:
    individuals), or a list of arrays of the same individuals (e.g. one per date): then every set
    is trained and scored, and one level serves all sets by their mean score. The other arguments
    are as in cross_validate_levels. Returns (noise_sd of each subset (k,), the chosen level times
    noise_scale(X) of the subset, for train; the (k, n_splits, n_levels) scores; and the
    (k, n_splits, n_levels, n) float32 traits); with a list of sets, noise_sd (one per set:
    (k, sets)), scores and traits have a set axis after the first. The chosen level of subset i
    is best_level(levels, scores[i]).
    """
    sets = [np.asarray(x) for x in (X if isinstance(X, list | tuple) else [X])]
    all_rows = np.ones((1, len(sets[0])), dtype=bool) if subsets is None else np.asarray(subsets)
    scales = np.array([[noise_scale(x[rows].reshape((rows.sum(), -1))) for x in sets]
                       for rows in all_rows])

    def fit(fit_rows, subset, level):
        out = []
        for k, measurements in enumerate(sets):
            x = torch.as_tensor(measurements, dtype=torch.float32)
            torch.manual_seed(seed)
            start = make_model()
            models = [copy.deepcopy(start) for _ in level]
            train_batch(models, measurements, groups, environment, fit_rows,
                        level * scales[subset, k], noise='normal', subgroups=subgroups,
                        estimator=estimator, **train_options)
            traits = []
            for model in models:
                model.cpu()
                with torch.no_grad():
                    traits.append(model(x)[:, 0].double().numpy())
            out.append(traits)
        return np.array(out)

    scores, traits = cross_validate_levels(fit, groups, environment, levels, all_rows, n_splits,
                                           n_folds, seed, subgroups, estimator, split_units)
    chosen = np.array([best_level(levels, scores[i]) for i in range(len(all_rows))])
    noise_sd = chosen[:, None] * scales
    if not isinstance(X, list | tuple):
        noise_sd, scores, traits = noise_sd[:, 0], scores[:, 0], traits[:, 0]
    return noise_sd, scores, traits


def score_linear_conv_levels(make_conv, X, groups, environment, linear_noise_sd, levels,
                             subsets=None, n_splits=5, n_folds=5, seed=0, subgroups=None,
                             estimator='anova', split_units=None, **train_options):
    """Held-out heritability of the first LinearConvModel trait at each noise level of the
    convolutional branch: cross_validate_levels; the chosen level of subset i is
    highest_level_within_one_se(levels, scores[i]).

    The method fits the first trait by train_linear_conv_models: linear H2-opt with noise of
    standard deviation linear_noise_sd of the subset (length k, absolute standard deviations,
    e.g. noise_sd of select_noise_level), then the LinearConvModel with convolutional noise of
    standard deviation level * noise_scale(X) (X of the subset, flattened). All copies are trained
    together; train_options go to train_linear_conv_models (e.g. n_linear_iter, n_iter, device,
    max_copies). The other arguments are as in cross_validate_levels. One level per call is
    enough: the scores of a level do not depend on the other levels. Returns the
    (k, n_splits, n_levels) scores and the (k, n_splits, n_levels, n) float32 traits.
    """
    X = np.asarray(X)
    flat = X.reshape((len(X), -1))
    all_rows = np.ones((1, len(X)), dtype=bool) if subsets is None else np.asarray(subsets)
    linear_noise_sd = np.broadcast_to(np.asarray(linear_noise_sd, dtype=float), (len(all_rows),))
    scales = np.array([noise_scale(flat[rows]) for rows in all_rows])

    def fit(fit_rows, subset, level):
        torch.manual_seed(seed)
        models = train_linear_conv_models(
            make_conv, X, groups, environment, np.where(fit_rows, 0, 1),
            linear_noise_sd[subset][:, None], (level * scales[subset])[:, None], n_traits=1,
            subgroups=subgroups, estimator=estimator, verbose=False, **train_options)
        return np.array([synthetic_traits(model, X, rows)[:, 0]
                         for model, rows in zip(models, fit_rows, strict=True)])

    return cross_validate_levels(fit, groups, environment, levels, all_rows, n_splits, n_folds,
                                 seed, subgroups, estimator, split_units)
