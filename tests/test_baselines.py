import numpy as np
import pytest
import torch

import h2opt
from h2opt import baselines, folds


def test_baselines(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::10]
    train = np.random.RandomState(0).randint(5, size=len(groups)) != 0
    herit = {}
    for name, baseline in [('pca', baselines.PCA(3)), ('gpca', baselines.GeneticPCA(3)),
                           ('lda', baselines.LDA(3)), ('pch', baselines.PCH(3)),
                           ('max', baselines.MaxHeritabilityFeatures(3))]:
        Y = baseline.fit(X[train], groups[train], environment[train]).transform(X)
        assert Y.shape == (len(groups), 3)
        # traits are uncorrelated on the training individuals
        np.testing.assert_allclose(np.corrcoef(Y[train].T), np.eye(3), atol=1e-6)
        traits = torch.tensor(Y[train]).float()
        herit[name] = h2opt.anova_heritability(traits, groups[train], environment[train]).numpy()
    # unregularized PCH maximizes the training heritability among linear traits
    assert all(herit['pch'][0] >= h[0] - 1e-4 for h in herit.values())


def test_max_features_never_picks_constant_or_spanned_measurements():
    # small-scale measurements (as metabolite intensities over their maximum): a zero column, a
    # copy of the most heritable column and many noisy ones; the picks stay distinct and
    # independent, so their decorrelation exists
    rng = np.random.RandomState(0)
    groups = np.repeat(np.arange(30), 2)
    genetic = rng.normal(size=30)[groups]
    X = 1e-5 * rng.normal(size=(60, 40))
    X[:, 1] += 1e-4 * genetic
    X[:, 0] = 0
    X[:, 2] = 3 * X[:, 1]
    model = baselines.MaxHeritabilityFeatures(10).fit(X, groups)
    assert model.features_[0] in (1, 2)
    assert 0 not in model.features_ and not {1, 2} <= set(model.features_)
    Y = model.transform(X)
    np.testing.assert_allclose(np.corrcoef(Y.T), np.eye(10), atol=1e-8)


def test_pch_more_measurements_than_individuals(sorghum):
    X, groups, environment = sorghum
    rows = np.arange(200)
    Y = baselines.PCH(2, ridge=1e-3).fit_transform(X[rows], groups[rows], environment[rows])
    assert Y.shape == (200, 2) and np.all(np.isfinite(Y))


def test_pch_henderson3_maximizes_henderson3_heritability(sorghum):
    X, groups, environment = sorghum
    X = X[:600, ::40]
    groups, environment = groups[:600], environment[:600]
    subgroups = np.arange(600) % 2
    pch = baselines.PCH(2, estimator='henderson3').fit(X, groups, environment, subgroups)
    model = h2opt.Henderson3(groups, environment, subgroups)
    best = model.heritability(torch.tensor(pch.transform(X)[:, :1])).item()
    w = np.random.RandomState(0).normal(size=(X.shape[1], 20))
    others = np.c_[X @ w, baselines.PCH(1).fit_transform(X, groups, environment)]
    assert np.all(model.heritability(torch.tensor(others)).numpy() <= best + 1e-8)


def test_pch_fit_ridges_equals_separate_fits(sorghum):
    X, groups, environment = sorghum
    X = X[:300, ::50]
    groups, environment = groups[:300], environment[:300]
    ridges = (1e-4, 1e-1)
    fitted = baselines.PCH.fit_ridges(X, groups, environment, None, 2, ridges)
    for pch, ridge in zip(fitted, ridges, strict=True):
        alone = baselines.PCH(2, ridge).fit_transform(X, groups, environment)
        np.testing.assert_allclose(pch.transform(X), alone, atol=1e-8)


def test_pch_tune_scores_held_out_heritability(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    ridges = (1e-3, 1.0)
    ridge, scores, fold_scores = baselines.PCH.tune(X, groups, environment, 2, ridges=ridges,
                                                    n_folds=3)
    assert ridge == ridges[int(np.argmax(scores))]
    fold = folds.group_folds(groups, 3)
    fit, val = fold != 0, fold == 0
    pch = baselines.PCH(2, ridges[1]).fit(X[fit], groups[fit], environment[fit])
    traits = pch.transform(X[val])
    expected = h2opt.anova_heritability(traits, groups[val], environment[val]).mean()
    assert fold_scores[0, 1] == pytest.approx(expected)
    assert baselines.RIDGES[0] == pytest.approx(1e-6) and baselines.RIDGES[-1] == pytest.approx(100)


def test_pch_tune_on_several_measurement_sets_averages_their_scores(sorghum):
    X, groups, environment = sorghum
    first_set, second_set = X[:, ::20], X[:, 5::20]
    options = dict(ridges=(1e-3, 1.0), n_folds=3, subgroups=np.arange(len(groups)) % 2,
                   estimator='henderson3')
    _, both, _ = baselines.PCH.tune([first_set, second_set], groups, environment, 1, **options)
    _, first, _ = baselines.PCH.tune(first_set, groups, environment, 1, **options)
    _, second, _ = baselines.PCH.tune(second_set, groups, environment, 1, **options)
    np.testing.assert_allclose(both, (first + second) / 2, rtol=1e-10)


def test_split_units_assign_individuals_to_validation_folds(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    plants = np.arange(len(groups))
    _, scores, fold_scores = baselines.PCH.tune(X, groups, environment, 1, ridges=(1e-3,),
                                                n_folds=3, split_units=plants)
    fold = folds.group_folds(plants, 3)
    fit, val = fold != 0, fold == 0
    assert set(groups[fit]) & set(groups[val])
    pch = baselines.PCH(1, 1e-3).fit(X[fit], groups[fit], environment[fit])
    expected = h2opt.anova_heritability(pch.transform(X[val]), groups[val], environment[val])
    assert fold_scores[0, 0] == pytest.approx(expected[0])
