import numpy as np
import pytest

import h2opt
from h2opt import folds, selection


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
    fold = folds.group_folds(groups, 3)
    fit, val = fold != 0, fold == 0
    Y = traits[0, :, 0, 1].T.astype(float)
    trait = h2opt.Decorrelation().fit(Y[fit]).transform(Y)[:, -1:]
    expected = h2opt.anova_heritability(trait[val], groups[val], environment[val])[0]
    assert scores[0, 1, 0, 1] == pytest.approx(expected, abs=1e-5)
    assert selection.noise_scale(X) == pytest.approx(np.sqrt(X.var(axis=0).mean()))


def test_select_noise_level_on_subsets_ignores_the_other_individuals(sorghum):
    # the selection on a subset is the same whether it runs alone or with another subset, and
    # whatever the measurements outside the subset are (at level 0: with noise, the shared draw
    # differs with the set of individuals)
    X, groups, environment = sorghum
    X = X[:, ::20]
    subset = folds.group_folds(groups, 3) != 0
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


def test_select_noise_level_on_several_measurement_sets_averages_their_scores(sorghum):
    X, groups, environment = sorghum
    first_set, second_set = X[:, ::20], X[:, 5::20]
    options = dict(levels=(0.0, 0.1), n_splits=2, n_folds=3, n_iter=50,
                   subgroups=np.arange(len(groups)) % 2, estimator='henderson3')

    def make_model():
        return h2opt.LinearModel(first_set.shape[1])

    chosen, both, traits = selection.select_noise_level(make_model, [first_set, second_set],
                                                        groups, environment, 1, **options)
    _, first, _ = selection.select_noise_level(make_model, first_set, groups, environment, 1,
                                               **options)
    assert both.shape == (1, 2, 1, 2, 2) and traits.shape == (1, 2, 1, 2, 2, len(groups))
    np.testing.assert_allclose(both[:, 0], first, atol=1e-4)
    assert chosen[0] == options['levels'][int(np.argmax(both[0].mean(axis=(0, 1, 2))))]
