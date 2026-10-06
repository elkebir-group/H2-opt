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
    X = np.arange(20.0)

    def score_fold(fit, val):
        assert not set(groups[fit]) & set(groups[val])
        return [val.sum(), X[val].sum()]

    scores, fold_scores = selection.cross_validate(groups, score_fold, n_folds=5)
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


def test_select_noise_level_chooses_one_level_by_held_out_heritability(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    levels = (0.0, 0.1)
    chosen, scores, traits = selection.select_noise_level(lambda: h2opt.LinearModel(X.shape[1]),
                                                          X, groups, environment, 2,
                                                          levels=levels, n_splits=2, n_folds=3,
                                                          n_iter=100)
    assert chosen.shape == (1,) and scores.shape == (1, 2, 2, 2)
    assert traits.shape == (1, 2, 2, 2, len(groups))
    assert np.isfinite(scores).all()
    assert chosen[0] == levels[int(np.argmax(scores[0].mean(axis=(0, 1))))]
    # the saved traits 1 and 2 of split 0 at level 1 give the score of trait 2 again
    fold = selection.group_folds(groups, 3)
    fit, val = fold != 0, fold == 0
    Y = traits[0, :, 0, 1].T.astype(float)
    trait = h2opt.Decorrelation().fit(Y[fit]).transform(Y)[:, -1:]
    expected = h2opt.anova_heritability(trait[val], groups[val], environment[val])[0]
    assert scores[0, 1, 0, 1] == pytest.approx(expected, abs=1e-5)
    assert selection.noise_scale(X) == pytest.approx(np.sqrt(X.var(axis=0).mean()))
    assert selection.NOISE_LEVELS[4] ** 2 == pytest.approx(selection.RIDGES[4])


def test_select_noise_level_on_subsets_ignores_the_other_individuals(sorghum):
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

    _, together, _ = selection.select_noise_level(make_model, X, groups, environment, 1,
                                                   [subset, ~subset], **options)
    _, alone, _ = selection.select_noise_level(make_model, changed, groups, environment, 1,
                                                [subset], **options)
    np.testing.assert_allclose(together[0], alone[0], atol=1e-4)


def test_pch_tune_on_several_measurement_sets_averages_their_scores(sorghum):
    X, groups, environment = sorghum
    first_set, second_set = X[:, ::20], X[:, 5::20]
    options = dict(ridges=(1e-3, 1.0), n_folds=3, subgroups=np.arange(len(groups)) % 2,
                   estimator='henderson3')
    _, both, _ = baselines.PCH.tune([first_set, second_set], groups, environment, 1, **options)
    _, first, _ = baselines.PCH.tune(first_set, groups, environment, 1, **options)
    _, second, _ = baselines.PCH.tune(second_set, groups, environment, 1, **options)
    np.testing.assert_allclose(both, (first + second) / 2, rtol=1e-10)
