import numpy as np
import pytest

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
    assert h2opt.selection.RIDGES[0] == pytest.approx(1e-6)


def test_select_noise_levels_chooses_one_level_per_trait_by_held_out_heritability(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    levels = (0.0, 0.1)
    chosen, scores = selection.select_noise_levels(lambda: h2opt.LinearModel(X.shape[1]), X,
                                                   groups, environment, 2, levels=levels,
                                                   n_splits=2, n_folds=3, n_iter=100)
    assert chosen.shape == (1, 2) and scores.shape == (1, 2, 2, 2)
    assert np.isfinite(scores).all()
    for t in range(2):
        assert chosen[0, t] == levels[selection.one_standard_error_choice(scores[0, t])]
    assert selection.noise_scale(X) == pytest.approx(np.sqrt(X.var(axis=0).mean()))


def test_select_noise_levels_on_subsets_ignores_the_other_individuals(sorghum):
    # the selection on a subset is the same whether it runs alone or with another subset, and
    # whatever the measurements outside the subset are (at level 0: with noise, the shared draw
    # differs with the set of individuals)
    X, groups, environment = sorghum
    X = X[:, ::20]
    subset = selection.group_folds(groups, 3) != 0
    changed = X.copy()
    changed[~subset] = np.random.RandomState(0).normal(size=changed[~subset].shape)
    options = dict(levels=(0.0,), n_splits=2, n_folds=3, n_iter=100)

    def make_model():
        return h2opt.LinearModel(X.shape[1])

    _, together = selection.select_noise_levels(make_model, X, groups, environment, 1,
                                                [subset, ~subset], **options)
    _, alone = selection.select_noise_levels(make_model, changed, groups, environment, 1,
                                             [subset], **options)
    np.testing.assert_allclose(together[0], alone[0], atol=1e-4)


def test_one_standard_error_choice_takes_the_strongest_setting_near_the_best():
    scores = np.array([[0.70, 0.72, 0.71, 0.60],
                       [0.72, 0.74, 0.72, 0.62],
                       [0.68, 0.70, 0.70, 0.58]])
    # best: setting 1, mean 0.72, standard error 0.02 / sqrt(3) = 0.0115; setting 2 is at 0.71
    assert selection.one_standard_error_choice(scores) == 2
    assert selection.one_standard_error_choice(scores[:1]) == 1
