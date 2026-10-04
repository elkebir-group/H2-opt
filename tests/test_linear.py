import numpy as np
import pytest
import torch

import h2opt


def test_linear_h2opt_matches_pch_without_ridge(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    model = h2opt.LinearH2opt(2, ridge=1e-8, n_iter=30)
    Y = model.fit(X, groups, environment, validation=(X, groups, environment)).transform(X)
    herit = h2opt.anova_heritability(torch.tensor(Y), groups, environment).numpy()
    pch = h2opt.baselines.PCH(2).fit_transform(X, groups, environment)
    herit_pch = h2opt.anova_heritability(torch.tensor(pch), groups, environment).numpy()
    # L-BFGS converges to the PCH optimum, which maximizes heritability among linear traits
    assert herit[0] == pytest.approx(herit_pch[0], abs=1e-3)
    assert abs(np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]) < 1e-6
    assert model.validation_curves_.shape == (2, 30)
    assert model.noise_sd_ == pytest.approx(np.sqrt(1e-8 * X.var(axis=0).mean()), rel=1e-6)


def test_tune_picks_the_ridge_with_best_cross_validated_heritability(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    ridges = (1e-4, 1e-2, 1.0)
    ridge, scores, fold_scores = h2opt.LinearH2opt.tune(X, groups, environment, n_traits=2,
                                                        ridges=ridges, n_folds=3, n_iter=5)
    assert fold_scores.shape == (3, 3)
    np.testing.assert_allclose(scores, fold_scores.mean(axis=0))
    assert ridge == ridges[int(np.argmax(scores))]
    # the score of a fold: fit on the other folds, mean heritability on the held-out fold
    fold = h2opt.selection.group_folds(groups, 3)
    fit, val = fold != 0, fold == 0
    model = h2opt.LinearH2opt(2, ridge=ridges[0], n_iter=5)
    model.fit(X[fit], groups[fit], environment[fit], (X[val], groups[val], environment[val]))
    assert fold_scores[0, 0] == pytest.approx(model.validation_curves_[:, -1].mean())


def test_fit_stops_when_the_loss_no_longer_changes(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    model = h2opt.LinearH2opt(3, ridge=1e-2, n_iter=200).fit(X, groups, environment)
    assert np.all(model.n_steps_ < 200)
    # more steps would not change the traits
    capped = h2opt.LinearH2opt(3, ridge=1e-2, n_iter=int(model.n_steps_.max())).fit(
        X, groups, environment)
    np.testing.assert_array_equal(capped.weights_, model.weights_)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_gives_the_cpu_traits(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    Y = {device: h2opt.LinearH2opt(2, ridge=1e-2, device=device).fit_transform(
        X, groups, environment) for device in ('cpu', 'cuda')}
    # the objective does not depend on the scale of a trait, so compare standardized traits
    np.testing.assert_allclose(Y['cuda'] / Y['cuda'].std(0), Y['cpu'] / Y['cpu'].std(0),
                               atol=1e-5)
