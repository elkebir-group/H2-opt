"""Choosing a regularization strength by heritability on held-out groups.

H2-opt's noise levels (select_noise_levels), one per trait: for each trait in order, the level
whose trait has the highest mean held-out heritability over validation folds of the groups.
PCH penalizes ridge * v * |w|^2 (v the mean variance of the measurements); its ridge (PCH.tune,
through cross_validate) is chosen by the same rule. Also the assignment of groups to folds.
"""

import copy

import numpy as np
import torch

from .decorrelation import Decorrelation
from .heritability import _as_environment, anova_heritability
from .train import train_batch

# Noise levels of H2-opt: the standard deviation of the normal input noise as a fraction of
# noise_scale(X), the typical spread of a measurement.
NOISE_LEVELS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

# ridge 1 is the penalty of input noise with the standard deviation of a typical measurement.
RIDGES = tuple(float(r) for r in np.logspace(-6, 2, 17))


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


def select_noise_levels(make_model, X, groups, environment, n_traits, subsets=None,
                        levels=NOISE_LEVELS, n_splits=5, n_folds=5, seed=0, **train_options):
    """Noise level of each H2-opt trait by its held-out heritability.

    One selection runs on each subset of the individuals (subsets: (k, n) boolean, e.g. the
    training individuals of k outer folds; default: one selection on all). Within a subset, the
    groups are assigned to n_folds folds (group_folds); each of the first n_splits folds (default:
    all) is the held-out fold of one validation split. Traits are chosen in order. For trait t, in
    each split, a fresh model (make_model(), a module mapping (n, m) to (n, 1)) is trained on the
    other folds for each level, with normal noise of standard deviation level * noise_scale(X)
    (X of the subset), on top of traits 0..t-1 of that split (in the loss, the trait is its
    residual on them, as in train). Its score is the ANOVA heritability on the held-out fold of
    the trait made uncorrelated with traits 0..t-1 on the training folds (Decorrelation, as in
    synthetic_traits). The chosen level has the best mean score over the splits, as for the PCH
    ridge (PCH.tune). Each split then keeps its trait t trained at the chosen level. All subsets,
    splits and levels of one trait are trained together by train_batch; train_options go to it
    (e.g. n_iter, device).
    Returns (the chosen levels (k, n_traits), (k, n_traits, n_splits, n_levels) scores, and the
    trained traits before decorrelation, (k, n_traits, n_splits, n_levels, n) float32, on all n
    individuals). With the traits, the scores of any other rule's choice of trait t can be
    computed again without training, but the traits after t depend on the choice.
    """
    X = np.asarray(X)
    groups = np.asarray(groups)
    environment = _as_environment(environment, len(groups))
    subsets = np.ones((1, len(groups)), dtype=bool) if subsets is None else np.asarray(subsets)
    n_subsets = len(subsets)
    # (subset, split) pairs: training and held-out individuals of each validation split
    fit, held_out, scale = [], [], []
    for subset in subsets:
        fold = np.full(len(groups), -1)
        fold[subset] = group_folds(groups[subset], n_folds, seed)
        for split in range(n_splits):
            fit.append(subset & (fold != split))
            held_out.append(fold == split)
        scale.append(noise_scale(X[subset]))
    pairs = [(i, split) for i in range(n_subsets) for split in range(n_splits)]
    copies = [(p, a) for p in range(len(pairs)) for a in range(len(levels))]
    x = torch.as_tensor(X, dtype=torch.float32)
    earlier = [np.zeros((len(groups), 0)) for _ in pairs]
    chosen = np.zeros((n_subsets, n_traits))
    scores = np.zeros((n_subsets, n_traits, n_splits, len(levels)))
    traits = np.zeros((n_subsets, n_traits, n_splits, len(levels), len(groups)), dtype=np.float32)
    for t in range(n_traits):
        torch.manual_seed(seed + t)
        start = make_model()
        models = [copy.deepcopy(start) for _ in copies]
        train_batch(models, X, groups, environment, [fit[p] for p, _ in copies],
                    [levels[a] * scale[pairs[p][0]] for p, a in copies],
                    [earlier[p] for p, _ in copies], noise='normal', **train_options)
        outputs = {}
        for model, (p, a) in zip(models, copies, strict=False):
            model.cpu()
            with torch.no_grad():
                outputs[p, a] = model(x)[:, 0].double().numpy()
            Y = np.column_stack([earlier[p], outputs[p, a]])
            trait = Decorrelation().fit(Y[fit[p]]).transform(Y)[:, -1:]
            i, split = pairs[p]
            traits[i, t, split, a] = outputs[p, a]
            scores[i, t, split, a] = anova_heritability(trait[held_out[p]], groups[held_out[p]],
                                                        environment[held_out[p]])[0]
        for i in range(n_subsets):
            a = int(np.argmax(scores[i, t].mean(axis=0)))
            chosen[i, t] = levels[a]
            for p, (j, _) in enumerate(pairs):
                if j == i:
                    earlier[p] = np.column_stack([earlier[p], outputs[p, a]])
    return chosen, scores, traits
