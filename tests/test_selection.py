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
    splits = selection.N_PARTITIONS * 2
    assert noise_sd.shape == (1,) and scores.shape == (1, splits, 2)
    assert traits.shape == (1, splits, 2, len(groups))
    assert np.isfinite(scores).all()
    chosen = selection.highest_level_within_one_se(levels, scores[0])
    assert noise_sd[0] == pytest.approx(chosen * selection.noise_scale(X))
    # the saved trait of split 0 at level 2 gives its score again
    val = folds.group_folds(groups, 3) == 0
    trait = traits[0, 0, 1].astype(float)[:, None]
    expected = h2opt.anova_heritability(trait[val], groups[val], environment[val])[0]
    assert scores[0, 0, 1] == pytest.approx(expected, abs=1e-5)
    assert selection.noise_scale(X) == pytest.approx(np.sqrt(X.var(axis=0).mean()))


def test_cross_validate_levels_scores_the_held_out_first_trait():
    # a method whose first trait is a feature times its level, with the training rows checked
    rng = np.random.RandomState(0)
    groups = np.repeat(np.arange(30), 2)
    X = rng.normal(size=(60, 2)) + rng.normal(size=(30, 1))[groups]
    subsets = np.array([np.arange(60) < 40, np.arange(60) >= 20])
    calls = []

    def fit(fit_rows, subset, level):
        calls.append(len(level))
        for rows, i in zip(fit_rows, subset, strict=True):
            assert not (rows & ~subsets[i]).any()
        return np.array([X[:, 0] * lev for lev in level])

    scores, traits = selection.cross_validate_levels(fit, groups, None, (1.0, 2.0), subsets,
                                                     n_splits=2, n_folds=3)
    splits = selection.N_PARTITIONS * 2
    assert calls == [2 * splits * 2] and scores.shape == (2, splits, 2)
    assert traits.shape == (2, splits, 2, 60)
    # heritability does not depend on the scale, so both levels score the same
    np.testing.assert_allclose(scores[..., 0], scores[..., 1])
    fold = folds.group_folds(groups[subsets[1]], 3)
    val = np.zeros(60, dtype=bool)
    val[subsets[1]] = fold == 1
    expected = h2opt.anova_heritability(X[val, :1], groups[val])[0]
    assert scores[1, 1, 0] == pytest.approx(expected)
    # partition p uses seed p: split 2 + 1 is fold 1 of the partition with seed 1
    val[subsets[1]] = folds.group_folds(groups[subsets[1]], 3, 1) == 1
    expected = h2opt.anova_heritability(X[val, :1], groups[val])[0]
    assert scores[1, 2 + 1, 0] == pytest.approx(expected)


def test_highest_level_within_one_se_uses_the_paired_difference():
    # splits x levels: level 1.0 is best (mean 0.50). Level 10.0 is 0.01 lower on every split:
    # the paired difference has SE 0, so it is not within one SE, although the SE of the best
    # level's mean over the splits (0.1) is large
    scores = np.array([[0.50, 0.60, 0.59], [0.30, 0.40, 0.39]])
    levels = (0.1, 1.0, 10.0)
    assert selection.highest_level_within_one_se(levels, scores) == 1.0
    # differences -0.02 and 0.04 from the best: mean 0.01, SE 0.03, so 10.0 is within one SE
    scores = np.array([[0.50, 0.60, 0.62], [0.30, 0.40, 0.36]])
    assert selection.highest_level_within_one_se(levels, scores) == 10.0
    # axes before the splits (e.g. sets) are averaged first
    stacked = np.stack([scores, scores - [0, 0, 0.1]])
    assert selection.highest_level_within_one_se(levels, stacked) == 1.0
    with pytest.raises(ValueError, match='2 splits'):
        selection.highest_level_within_one_se(levels, scores[:1])


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
    splits = selection.N_PARTITIONS * 2
    assert both.shape == (1, 2, splits, 2) and traits.shape == (1, 2, splits, 2, len(groups))
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
    splits = selection.N_PARTITIONS * 2
    assert scores.shape == (1, splits, 2) and traits.shape == (1, splits, 2, len(groups))
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
