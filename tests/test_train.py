import copy

import numpy as np
import pytest
import torch

import h2opt


def test_test_individuals_do_not_shape_training(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    train_test = (np.random.RandomState(0).randint(4, size=len(groups)) == 0).astype(int)
    changed = X.copy()
    changed[train_test == 1] = np.random.RandomState(1).normal(size=changed[train_test == 1].shape)
    weights = []
    for measurements in (X, changed):
        torch.manual_seed(0)
        model = h2opt.TraitModels(3, h2opt.LinearModel, X.shape[1])
        h2opt.train(model, measurements, groups, environment, train_test, n_traits=3, n_iter=20,
                    learning_rate=1e-3, noise_level=0.005, verbose=False)
        weights.append(torch.cat([m.lin1.weight for m in model.models]).detach())
    torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
    traits = h2opt.synthetic_traits(model, changed, train_test == 0)
    np.testing.assert_allclose(np.corrcoef(traits[train_test == 0].T), np.eye(3), atol=1e-6)


def test_train_increases_heritability(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    torch.manual_seed(0)
    model = h2opt.TraitModels(1, h2opt.LinearModel, X.shape[1])

    def heritability():
        traits = model(torch.tensor(X).float()).detach()
        return h2opt.anova_heritability(traits, groups, environment).item()

    before = heritability()
    h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_iter=50,
                learning_rate=1e-3, noise_level=0.005, verbose=False)
    assert heritability() > before


def test_train_with_normal_noise_and_no_clip(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    torch.manual_seed(0)
    model = h2opt.TraitModels(2, h2opt.LinearModel, X.shape[1])
    h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_traits=2,
                n_iter=20, learning_rate=1e-3, noise_level=0.02, noise='normal',
                verbose=False)
    traits = model(torch.tensor(X).float()).detach()
    assert torch.isfinite(traits).all()
    with pytest.raises(ValueError):
        h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int),
                    n_iter=1, noise='gaussian', verbose=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_train_on_cuda(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    model = h2opt.TraitModels(2, h2opt.LinearModel, X.shape[1])
    h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_traits=2,
                n_iter=5, device='cuda', verbose=False)
    assert next(model.parameters()).is_cuda


def test_adam_reaches_the_pch_trait(sorghum):
    # with normal noise of SD s and no clipping, the optimum of a linear trait is the PCH trait
    # with ridge s^2 / v (v: the mean variance of the measurements)
    X, groups, environment = sorghum
    X = X[:, ::20]
    s = 0.0014
    pch = h2opt.baselines.PCH(1, ridge=s ** 2 / X.var(axis=0).mean()).fit(X, groups, environment)
    torch.manual_seed(0)
    model = h2opt.TraitModels(1, h2opt.LinearModel, X.shape[1])
    h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_iter=10000,
                noise_level=s, noise='normal', verbose=False)
    trait = h2opt.synthetic_traits(model, X, np.ones(len(groups), dtype=bool))[:, 0]
    assert abs(np.corrcoef(trait, pch.transform(X)[:, 0])[0, 1]) > 0.9999


def test_rmsprop_and_unknown_optimizer(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    torch.manual_seed(0)
    model = h2opt.TraitModels(1, h2opt.LinearModel, X.shape[1])

    def heritability():
        traits = model(torch.tensor(X).float()).detach()
        return h2opt.anova_heritability(traits, groups, environment).item()

    before = heritability()
    h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_iter=50,
                optimizer='rmsprop', learning_rate=1e-3, noise_level=0.005, verbose=False)
    assert heritability() > before
    with pytest.raises(ValueError):
        h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_iter=1,
                    optimizer='sgd', verbose=False)


def test_train_batch_copies_with_shared_rows_train_as_alone(sorghum):
    # copies with the same training rows share each noise draw; each gets only its own gradient
    X, groups, environment = sorghum
    X = X[:, ::20]
    rows = np.random.RandomState(0).randint(4, size=len(groups)) != 0
    earlier = np.random.RandomState(1).normal(size=(len(groups), 1))
    torch.manual_seed(0)
    start = h2opt.LinearModel(X.shape[1])
    levels = [0.0, 0.002]

    def outputs(models):
        with torch.no_grad():
            return [m(torch.tensor(X).float())[:, 0].numpy() for m in models]

    together = [copy.deepcopy(start) for _ in levels]
    torch.manual_seed(1)
    h2opt.train_batch(together, X, groups, environment, [rows, rows], levels, [earlier, earlier],
                      n_iter=300)
    for a, level in enumerate(levels):
        alone = copy.deepcopy(start)
        torch.manual_seed(1)
        h2opt.train_batch([alone], X, groups, environment, [rows], [level], [earlier], n_iter=300)
        assert abs(np.corrcoef(outputs(together)[a], outputs([alone])[0])[0, 1]) > 0.9999
    # the two levels give different traits
    assert abs(np.corrcoef(*outputs(together))[0, 1]) < 0.9999


def test_train_takes_one_noise_level_per_trait(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    model = h2opt.TraitModels(2, h2opt.LinearModel, X.shape[1])
    h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_traits=2,
                n_iter=5, noise_level=[0.0, 0.01], noise='normal', verbose=False)
    with pytest.raises(ValueError):
        h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_traits=2,
                    n_iter=1, noise_level=[0.0, 0.01, 0.1], verbose=False)


def test_train_batch_linear_model_matches_the_generic_path(sorghum):
    # LinearModel takes a shortcut, (X + level Z) w = X w + level (Z w); a plain nn.Linear does not
    X, groups, environment = sorghum
    X = X[:, ::20]
    rows = np.random.RandomState(0).randint(4, size=len(groups)) != 0
    torch.manual_seed(0)
    linear = h2opt.LinearModel(X.shape[1])
    plain = copy.deepcopy(linear.lin1)
    for model in (linear, plain):
        torch.manual_seed(1)
        h2opt.train_batch([model], X, groups, environment, [rows], [0.002], n_iter=100)
    torch.testing.assert_close(linear.lin1.weight, plain.weight, rtol=0, atol=1e-5)


def test_train_models_trains_each_model_as_alone(sorghum):
    # without noise, models trained together on different splits match models trained alone
    X, groups, environment = sorghum
    X = X[:, ::20]
    splits = np.random.RandomState(0).randint(2, size=(2, len(groups)))
    torch.manual_seed(0)
    start = h2opt.TraitModels(2, h2opt.LinearModel, X.shape[1])
    together = [copy.deepcopy(start) for _ in splits]
    h2opt.train_models(together, X, groups, environment, splits, n_traits=2, n_iter=50,
                       noise_levels=0.0, verbose=False)
    for model, train_test in zip(together, splits, strict=True):
        alone = h2opt.train(copy.deepcopy(start), X, groups, environment, train_test, n_traits=2,
                            n_iter=50, noise_level=0.0, verbose=False)
        is_train = train_test == 0
        np.testing.assert_allclose(h2opt.synthetic_traits(model, X, is_train),
                                   h2opt.synthetic_traits(alone, X, is_train), atol=1e-4)
