"""Choosing a regularization strength by the held-out heritability of the first trait.

H2-opt's noise level (select_noise_level), one for all traits: the level whose first trait has
the highest mean held-out heritability over validation folds of the groups (h2opt.folds). The
first trait sets the level because a mean over all traits rewards spreading the heritable signal
over more traits: at high noise, the first traits lose heritability and the later traits gain
it. The first trait also needs no earlier traits, so only it is trained.

The noise of the convolutional branch of LinearConvModel (score_linear_conv_levels, with the
linear branch fixed): the highest level whose first trait is within one standard error of the
best (highest_level_within_one_se). A high level makes the convolutional branch add almost
nothing, so the model stays at the linear trait unless the convolutional branch clearly raises
the held-out heritability.
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
    the rule of select_noise_level. With the scores of levels trained in separate runs, it gives
    the choice of one run over all of them."""
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


def _validation_copies(groups, subsets, units, n_folds, n_splits, levels, seed):
    """(subset, split, level) of each copy, with its training and held-out individuals."""
    copies, fit, held_out = [], [], []
    for i, subset in enumerate(subsets):
        fold = np.full(len(groups), -1)
        fold[subset] = group_folds(units[subset], n_folds, seed)
        for split in range(n_splits):
            for a in range(len(levels)):
                copies.append((i, split, a))
                fit.append(subset & (fold != split))
                held_out.append(fold == split)
    return copies, fit, held_out


def select_noise_level(make_model, X, groups, environment, subsets=None, levels=NOISE_LEVELS,
                       n_splits=5, n_folds=5, seed=0, subgroups=None, estimator='anova',
                       split_units=None, **train_options):
    """Noise of H2-opt, one level for all traits, by the held-out heritability of the first trait.

    One selection runs on each subset of the individuals (subsets: (k, n) boolean, e.g. the training
    individuals of k outer folds; default: one selection on all). Within a subset, the groups (or
    the split_units, labels of the individuals; e.g. np.arange(n) to split individuals) are assigned
    to n_folds folds (group_folds); each of the first n_splits folds (default: all) is the held-out
    fold of one validation split. In each split and for each level, the first trait is trained on
    the other folds, as in train: a model make_model() (a module mapping the measurements of n
    individuals to (n, 1)) with normal noise of standard deviation level * noise_scale(X) (X of the
    subset). Its score is its heritability (estimator and subgroups as in train) on the held-out
    fold. The chosen level has the best score averaged over the splits (best_level). X is one
    array of measurements (first axis: individuals), or a list of arrays of the same individuals
    (e.g. one per date): then every set is trained and scored, and one level serves all sets by
    their mean score. All subsets, splits and levels are trained together by train_batch;
    train_options go to it (e.g. n_iter, device, max_copies).
    Returns (noise_sd of each subset (k,), the chosen level times noise_scale(X) of the subset, for
    train; (k, n_splits, n_levels) scores; and the trained traits, (k, n_splits, n_levels, n)
    float32, on all n individuals); with a list of sets, noise_sd (one per set: (k, sets)), scores
    and traits have a set axis after the first. The chosen level of subset i is
    best_level(levels, scores[i]). The traits of a level do not depend on the other levels, so the
    traits and scores give the choice of any other rule over the levels without training again.
    """
    sets = X if isinstance(X, list | tuple) else [X]
    groups = np.asarray(groups)
    environment = _as_environment(environment, len(groups))
    subgroups = None if subgroups is None else np.asarray(subgroups)
    units = groups if split_units is None else np.asarray(split_units)
    subsets = np.ones((1, len(groups)), dtype=bool) if subsets is None else np.asarray(subsets)
    n_subsets = len(subsets)
    copies, fit, held_out = _validation_copies(groups, subsets, units, n_folds, n_splits, levels,
                                               seed)
    shape = (n_subsets, len(sets), n_splits, len(levels))
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
        torch.manual_seed(seed)
        start = make_model()
        models = [copy.deepcopy(start) for _ in copies]
        train_batch(models, measurements, groups, environment, fit, sd, noise='normal',
                    subgroups=subgroups, estimator=estimator, **train_options)
        for c, (model, (i, split, a)) in enumerate(zip(models, copies, strict=True)):
            model.cpu()
            with torch.no_grad():
                trait = model(x)[:, :1].double().numpy()
            traits[i, k, split, a] = trait[:, 0]
            rows = held_out[c]
            scores[i, k, split, a] = heritability(
                trait[rows], groups[rows], environment[rows],
                None if subgroups is None else subgroups[rows], estimator)[0]
    chosen = np.array([best_level(levels, scores[i]) for i in range(n_subsets)])
    noise_sd = chosen[:, None] * scales
    if not isinstance(X, list | tuple):
        noise_sd, scores, traits = noise_sd[:, 0], scores[:, 0], traits[:, 0]
    return noise_sd, scores, traits


def score_linear_conv_levels(make_conv, X, groups, environment, linear_noise_sd, levels,
                             subsets=None, n_splits=5, n_folds=5, seed=0, subgroups=None,
                             estimator='anova', split_units=None, **train_options):
    """Held-out heritability of the first LinearConvModel trait at each noise level of the
    convolutional branch, on validation splits as in select_noise_level.

    Within each subset of the individuals (subsets: (k, n) boolean, e.g. the training individuals
    of k outer folds; default: one subset of all), the groups (or split_units) are assigned to
    n_folds folds (group_folds), and each of the first n_splits folds is the held-out fold of one
    validation split. For each split and level, train_linear_conv_models trains the first trait on
    the other folds of the subset: linear H2-opt with noise of standard deviation linear_noise_sd
    of the subset (length k, absolute standard deviations, e.g. noise_sd of select_noise_level),
    then the LinearConvModel with convolutional noise of standard deviation level * noise_scale(X)
    (X of the subset, flattened). The score is its heritability on the held-out fold. The chosen
    level of subset i is highest_level_within_one_se(levels, scores[i]). One level per call is
    enough: the scores of a level do not depend on the other levels, so levels scored in separate
    runs can be compared. All copies (subset, split, level) are trained together; train_options go
    to train_linear_conv_models (e.g. n_linear_iter, n_iter, device, max_copies). Returns
    ((k, n_splits, n_levels) scores, and the (k, n_splits, n_levels, n) float32 traits of all n
    individuals).
    """
    X = np.asarray(X)
    groups = np.asarray(groups)
    environment = _as_environment(environment, len(groups))
    subgroups = None if subgroups is None else np.asarray(subgroups)
    units = groups if split_units is None else np.asarray(split_units)
    subsets = np.ones((1, len(groups)), dtype=bool) if subsets is None else np.asarray(subsets)
    linear_noise_sd = np.broadcast_to(np.asarray(linear_noise_sd, dtype=float), (len(subsets),))
    flat = X.reshape((len(X), -1))
    scales = [noise_scale(flat[subset]) for subset in subsets]
    copies, fit, held_out = _validation_copies(groups, subsets, units, n_folds, n_splits, levels,
                                               seed)
    # 0: training folds, 1: held-out fold, 2: outside both
    train_test = np.where(np.array(fit), 0, np.where(np.array(held_out), 1, 2))
    linear_sd = [linear_noise_sd[i] for i, _, _ in copies]
    conv_sd = [levels[a] * scales[i] for i, _, a in copies]
    torch.manual_seed(seed)
    models = train_linear_conv_models(
        make_conv, X, groups, environment, train_test, np.array(linear_sd)[:, None],
        np.array(conv_sd)[:, None], n_traits=1, subgroups=subgroups, estimator=estimator,
        verbose=False, **train_options)
    scores = np.zeros((len(subsets), n_splits, len(levels)))
    traits = np.zeros(scores.shape + (len(groups),), dtype=np.float32)
    for model, rows, (i, split, a) in zip(models, train_test, copies, strict=True):
        Y = synthetic_traits(model, X, rows == 0)
        held = rows == 1
        scores[i, split, a] = heritability(
            Y[held], groups[held], environment[held],
            None if subgroups is None else subgroups[held], estimator)[0]
        traits[i, split, a] = Y[:, 0]
    return scores, traits
