import numpy as np
import torch

import h2opt
from h2opt import baselines


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


def test_pch_more_measurements_than_individuals(sorghum):
    X, groups, environment = sorghum
    rows = np.arange(200)
    Y = baselines.PCH(2, ridge=1e-3).fit_transform(X[rows], groups[rows], environment[rows])
    assert Y.shape == (200, 2) and np.all(np.isfinite(Y))


def test_pch_forms_give_anova_heritability(sorghum):
    X, groups, environment = sorghum
    X = X[:300, ::50]
    groups, environment = groups[:300], environment[:300]
    A, B = baselines.PCH.heritability_forms(X, groups, environment)
    w = np.random.RandomState(0).normal(size=(X.shape[1], 4))
    expected = h2opt.anova_heritability(X @ w, groups, environment)
    ratio = np.einsum('ik,ij,jk->k', w, A, w) / np.einsum('ik,ij,jk->k', w, B, w)
    np.testing.assert_allclose(ratio, expected, rtol=1e-8)


def test_decorrelation_is_sequential_least_squares():
    rng = np.random.RandomState(0)
    Y = rng.normal(size=(200, 3)) @ np.triu(np.ones((3, 3))) + 5
    fit = np.arange(200) < 150
    decorrelation = h2opt.Decorrelation().fit(Y[fit])
    out = decorrelation.transform(Y)
    np.testing.assert_allclose(np.corrcoef(out[fit].T), np.eye(3), atol=1e-10)
    # trait 3: residual of trait 3 on an intercept and traits 1-2, coefficients from the fit rows
    design = np.column_stack([np.ones(fit.sum()), Y[fit, :2]])
    coef = np.linalg.lstsq(design, Y[fit, 2], rcond=None)[0]
    residual = Y[:, 2] - np.column_stack([np.ones(200), Y[:, :2]]) @ coef
    np.testing.assert_allclose(out[:, 2], residual, atol=1e-10)
    np.testing.assert_allclose(out[:, 0], Y[:, 0] - Y[fit, 0].mean(), atol=1e-10)


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
