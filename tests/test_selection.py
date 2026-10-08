import numpy as np
import pytest

import h2opt
from h2opt import folds, selection


def test_select_noise_level_chooses_one_level_by_held_out_heritability(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    levels = (0.0, 0.1)
    noise_sd, scores, traits = selection.select_noise_level(
        lambda: h2opt.LinearModel(X.shape[1]), X, groups, environment, levels=levels,
        n_splits=2, n_folds=3, n_iter=100)
    assert noise_sd.shape == (1,) and scores.shape == (1, 2, 2)
    assert traits.shape == (1, 2, 2, len(groups))
    assert np.isfinite(scores).all()
    chosen = levels[int(np.argmax(scores[0].mean(axis=0)))]
    assert selection.best_level(levels, scores[0]) == chosen
    assert noise_sd[0] == pytest.approx(chosen * selection.noise_scale(X))
    # the saved trait of split 0 at level 2 gives its score again
    val = folds.group_folds(groups, 3) == 0
    trait = traits[0, 0, 1].astype(float)[:, None]
    expected = h2opt.anova_heritability(trait[val], groups[val], environment[val])[0]
    assert scores[0, 0, 1] == pytest.approx(expected, abs=1e-5)
    assert selection.noise_scale(X) == pytest.approx(np.sqrt(X.var(axis=0).mean()))


def test_highest_level_within_one_se():
    # splits x levels: level 1.0 is best (mean 0.54, SE 0.02); 10.0 (0.52) is within one SE
    scores = np.array([[0.50, 0.52, 0.49], [0.54, 0.56, 0.55]])
    assert selection.highest_level_within_one_se((0.1, 1.0, 10.0), scores) == 10.0
    assert selection.highest_level_within_one_se((0.1, 1.0, 10.0), scores - [0, 0, 0.05]) == 1.0
    # axes before the splits (e.g. sets) are averaged first
    stacked = np.stack([scores, scores - [0, 0, 0.1]])
    assert selection.highest_level_within_one_se((0.1, 1.0, 10.0), stacked) == 1.0


def test_select_noise_level_on_subsets_ignores_the_other_individuals(sorghum):
    # the selection on a subset is the same whether it runs alone or with another subset, and
    # whatever the measurements outside the subset are (at level 0: with noise, the shared draw
    # differs with the set of individuals)
    X, groups, environment = sorghum
    X = X[:, ::10]
    subset = folds.group_folds(groups, 3) != 0
    changed = X.copy()
    changed[~subset] = np.random.RandomState(0).normal(size=changed[~subset].shape)
    options = dict(levels=(0.0,), n_splits=2, n_folds=3, n_iter=100)

    def make_model():
        return h2opt.LinearModel(X.shape[1])

    _, together, _ = selection.select_noise_level(make_model, X, groups, environment,
                                                   [subset, ~subset], **options)
    _, alone, _ = selection.select_noise_level(make_model, changed, groups, environment,
                                                [subset], **options)
    np.testing.assert_allclose(together[0], alone[0], atol=1e-4)


def test_select_noise_level_on_several_measurement_sets_averages_their_scores(sorghum_small):
    X, groups, environment = sorghum_small
    first_set, second_set = X[:, ::20], X[:, 5::20]
    options = dict(levels=(0.0, 0.1), n_splits=2, n_folds=3, n_iter=50,
                   subgroups=np.arange(len(groups)) % 2, estimator='henderson3')

    def make_model():
        return h2opt.LinearModel(first_set.shape[1])

    noise_sd, both, traits = selection.select_noise_level(make_model, [first_set, second_set],
                                                          groups, environment, **options)
    _, first, _ = selection.select_noise_level(make_model, first_set, groups, environment,
                                               **options)
    assert both.shape == (1, 2, 2, 2) and traits.shape == (1, 2, 2, 2, len(groups))
    np.testing.assert_allclose(both[:, 0], first, atol=1e-4)
    chosen = options['levels'][int(np.argmax(both[0].mean(axis=(0, 1))))]
    # one level for both sets, each scaled by its own noise_scale
    np.testing.assert_allclose(noise_sd, [[chosen * selection.noise_scale(first_set),
                                           chosen * selection.noise_scale(second_set)]])


def _linear_conv_scores(X, groups, environment, subsets, **options):
    return selection.score_linear_conv_levels(
        lambda: h2opt.ConvModel(X.shape[1]), X, groups, environment, 0.01, (0.1, 1.0),
        subsets=subsets, n_splits=2, n_folds=3, n_linear_iter=50, n_iter=20, **options)


def test_score_linear_conv_levels_scores_held_out_heritability(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::10]
    scores, traits = _linear_conv_scores(X, groups, environment, None)
    assert scores.shape == (1, 2, 2) and traits.shape == (1, 2, 2, len(groups))
    assert np.isfinite(scores).all()
    # the saved trait of split 1 at level 2 gives its score again
    val = folds.group_folds(groups, 3) == 1
    Y = traits[0, 1, 1].astype(float)[:, None]
    expected = h2opt.anova_heritability(Y[val], groups[val], environment[val])
    np.testing.assert_allclose(scores[0, 1, 1], expected[0], atol=1e-5)


def test_score_linear_conv_levels_ignores_the_individuals_outside_the_subset(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::10]
    subset = folds.group_folds(groups, 3) != 0
    changed = X.copy()
    changed[~subset] = np.random.RandomState(0).normal(size=changed[~subset].shape)
    scores, _ = _linear_conv_scores(X, groups, environment, subset[None])
    again, _ = _linear_conv_scores(changed, groups, environment, subset[None])
    np.testing.assert_allclose(scores, again, atol=1e-4)
