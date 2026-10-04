import numpy as np
import pytest
import torch

import h2opt
from h2opt import baselines, selection


def test_group_folds_keep_groups_together_and_balance_folds():
    groups = np.repeat(np.arange(23), 3)
    fold = selection.group_folds(groups, n_folds=5, seed=1)
    for g in range(23):
        assert len(set(fold[groups == g])) == 1
    sizes = np.bincount(fold[::3], minlength=5)
    assert sizes.max() - sizes.min() <= 1


def test_cross_validate_averages_fold_scores():
    groups = np.repeat(np.arange(10), 2)
    X = np.arange(20.0)[:, None]

    def score_fold(X_fit, groups_fit, environment_fit, X_val, groups_val, environment_val):
        assert not set(groups_fit) & set(groups_val)
        assert environment_fit.shape == (len(X_fit), 0)
        return [len(X_val), X_val.sum()]

    scores, fold_scores = selection.cross_validate(X, groups, None, score_fold, n_folds=5)
    assert fold_scores.shape == (5, 2)
    assert fold_scores[:, 0].sum() == 20 and fold_scores[:, 1].sum() == X.sum()
    np.testing.assert_allclose(scores, fold_scores.mean(axis=0))


def test_pch_tune_scores_held_out_heritability(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    ridges = (1e-3, 1.0)
    ridge, scores, fold_scores = baselines.PCH.tune(X, groups, environment, 2, ridges=ridges,
                                                    n_folds=3)
    assert ridge == ridges[int(np.argmax(scores))]
    fold = selection.group_folds(groups, 3)
    fit, val = fold != 0, fold == 0
    pch = baselines.PCH(2, ridges[1]).fit(X[fit], groups[fit], environment[fit])
    traits = pch.transform(X[val])
    expected = h2opt.anova_heritability(traits, groups[val], environment[val]).mean()
    assert fold_scores[0, 1] == pytest.approx(expected)
    assert h2opt.selection.RIDGES[0] == pytest.approx(1e-4)


def test_select_noise_level_scores_held_out_heritability_of_trait_1(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    levels = (0.0, 0.1)

    def make_model():
        return h2opt.TraitModels(1, h2opt.LinearModel, X.shape[1])

    level, scores = selection.select_noise_level(make_model, X, groups, environment, levels,
                                                 n_splits=1, n_folds=3, n_iter=200)
    assert scores.shape == (1, 2) and level == levels[int(np.argmax(scores[0]))]
    assert selection.noise_scale(X) == pytest.approx(np.sqrt(X.var(axis=0).mean()))
    # the score of level 0.1 on split 0, by hand
    is_val = selection.group_folds(groups, 3) == 0
    torch.manual_seed(0)
    model = make_model()
    h2opt.train(model, X, groups, environment, is_val.astype(int), n_iter=200,
                noise_level=0.1 * selection.noise_scale(X), noise='normal', verbose=False)
    trait = h2opt.synthetic_traits(model, X, ~is_val)
    expected = h2opt.anova_heritability(trait[is_val], groups[is_val], environment[is_val])[0]
    assert scores[0, 1] == pytest.approx(expected)
