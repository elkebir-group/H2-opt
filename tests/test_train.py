import copy
import sys
from functools import partial

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
                    learning_rate=1e-3, noise_sd=0.005, verbose=False)
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
                learning_rate=1e-3, noise_sd=0.005, verbose=False)
    assert heritability() > before


def test_train_with_normal_noise_and_no_clip(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    torch.manual_seed(0)
    model = h2opt.TraitModels(2, h2opt.LinearModel, X.shape[1])
    h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_traits=2,
                n_iter=20, learning_rate=1e-3, noise_sd=0.02,
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
                noise_sd=s, verbose=False)
    trait = h2opt.synthetic_traits(model, X, np.ones(len(groups), dtype=bool))[:, 0]
    assert abs(np.corrcoef(trait, pch.transform(X)[:, 0])[0, 1]) > 0.9999


def test_adam_reaches_the_henderson3_pch_trait(sorghum):
    # the same with Henderson's Method III: its group variance is unbiased, so pure noise adds
    # nothing to it in expectation, and the noise acts as a ridge in the denominator
    X, groups, environment = sorghum
    X = X[:, ::20]
    subgroups = np.arange(len(groups)) % 2
    s = 0.0014
    pch = h2opt.baselines.PCH(1, ridge=s ** 2 / X.var(axis=0).mean(), estimator='henderson3')
    pch.fit(X, groups, environment, subgroups)
    torch.manual_seed(0)
    model = h2opt.TraitModels(1, h2opt.LinearModel, X.shape[1])
    h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_iter=10000,
                noise_sd=s, verbose=False, subgroups=subgroups,
                estimator='henderson3')
    trait = h2opt.synthetic_traits(model, X, np.ones(len(groups), dtype=bool))[:, 0]
    assert abs(np.corrcoef(trait, pch.transform(X)[:, 0])[0, 1]) > 0.9999


def test_train_batch_in_chunks_trains_each_copy_as_together(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    rows = np.random.RandomState(0).rand(3, len(groups)) < 0.8
    torch.manual_seed(0)
    start = h2opt.LinearModel(X.shape[1])
    together = [copy.deepcopy(start) for _ in range(3)]
    chunked = [copy.deepcopy(start) for _ in range(3)]
    options = dict(n_iter=50, subgroups=np.arange(len(groups)) % 2, estimator='henderson3')
    h2opt.train_batch(together, X, groups, environment, rows, [0, 0, 0], **options)
    h2opt.train_batch(chunked, X, groups, environment, rows, [0, 0, 0], max_copies=2, **options)
    for a, b in zip(together, chunked, strict=True):
        torch.testing.assert_close(a.lin1.weight, b.lin1.weight, atol=1e-5, rtol=1e-4)


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
                optimizer='rmsprop', learning_rate=1e-3, noise_sd=0.005, verbose=False)
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


def test_train_takes_one_noise_sd_per_trait(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    model = h2opt.TraitModels(2, h2opt.LinearModel, X.shape[1])
    h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_traits=2,
                n_iter=5, noise_sd=[0.0, 0.01], verbose=False)
    with pytest.raises(ValueError):
        h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_traits=2,
                    n_iter=1, noise_sd=[0.0, 0.01, 0.1], verbose=False)


def test_train_batch_linear_model_matches_the_generic_path(sorghum):
    # LinearModel takes a shortcut, (X + level Z) w = X w + level (Z w); a plain nn.Linear does not.
    # With normal noise, LinearModel draws Z w directly, and its draws differ from the generic path.
    X, groups, environment = sorghum
    X = X[:, ::20]
    rows = np.random.RandomState(0).randint(4, size=len(groups)) != 0
    torch.manual_seed(0)
    linear = h2opt.LinearModel(X.shape[1])
    plain = copy.deepcopy(linear.lin1)
    for model in (linear, plain):
        torch.manual_seed(1)
        h2opt.train_batch([model], X, groups, environment, [rows], [0.002], n_iter=100,
                          noise='uniform')
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
                       noise_sd=0.0, verbose=False)
    for model, train_test in zip(together, splits, strict=True):
        alone = h2opt.train(copy.deepcopy(start), X, groups, environment, train_test, n_traits=2,
                            n_iter=50, noise_sd=0.0, verbose=False)
        is_train = train_test == 0
        np.testing.assert_allclose(h2opt.synthetic_traits(model, X, is_train),
                                   h2opt.synthetic_traits(alone, X, is_train), atol=1e-4)


def _image_data():
    rng = np.random.RandomState(0)
    groups = np.repeat(np.arange(15), 4)
    images = rng.normal(size=(60, 2, 15, 15)) + groups[:, None, None, None] / 10
    train_test = np.array([(np.arange(60) % 3 == k).astype(int) for k in range(2)])
    return images.astype(np.float32), groups, train_test


def test_vmap_randomness_leaves_models_without_draws_unchanged(monkeypatch):
    # train_batch lets the forward pass draw noise (vmap randomness 'different'); a model that
    # draws nothing trains as with the default randomness 'error', which forbids draws
    images, groups, train_test = _image_data()
    results = []
    for randomness in ('different', 'error'):
        if randomness == 'error':
            def default_vmap(function, randomness=None):
                return torch.func.vmap(function)
            monkeypatch.setattr(sys.modules['h2opt.train'], 'vmap', default_vmap)
        torch.manual_seed(0)
        models = [h2opt.TraitModels(2, h2opt.ImageConvModel, 2, 15) for _ in train_test]
        h2opt.train_models(models, images, groups, None, train_test, n_traits=2, n_iter=20,
                           noise_sd=0.3, verbose=False)
        results.append([h2opt.synthetic_traits(m, images, r == 0)
                        for m, r in zip(models, train_test, strict=True)])
    for a, b in zip(*results, strict=True):
        np.testing.assert_array_equal(a, b)


def start_scale(model, X, rows):
    """Standard deviation of the output of trait 1 of model on the training rows."""
    with torch.no_grad():
        return model.models[1](torch.tensor(X[rows == 0])).std(unbiased=False).item()


def test_train_linear_conv_models_starts_at_the_linear_traits():
    images, groups, train_test = _image_data()
    flat = images.reshape((len(images), -1))
    options = dict(n_traits=2, n_linear_iter=50, verbose=False)
    # with no second-stage steps, the traits are the first-stage linear traits, each with unit
    # standard deviation on the training individuals
    torch.manual_seed(0)
    start = h2opt.train_linear_conv_models(partial(h2opt.ImageConvModel, 2, 15), images, groups,
                                           None, train_test, 0.5, 0.3, n_iter=0, **options)
    torch.manual_seed(0)
    linear = [h2opt.TraitModels(2, h2opt.LinearModel, flat.shape[1]) for _ in train_test]
    h2opt.train_models(linear, flat, groups, None, train_test, n_iter=50, noise_sd=0.5,
                       verbose=False, n_traits=2)
    for model, reference, rows in zip(start, linear, train_test, strict=True):
        expected = h2opt.synthetic_traits(reference, flat, rows == 0)
        np.testing.assert_allclose(h2opt.synthetic_traits(model, images, rows == 0)[:, 0],
                                   expected[:, 0] / expected[rows == 0, 0].std(), atol=1e-4)
        np.testing.assert_allclose(h2opt.synthetic_traits(model, images, rows == 0)[:, 1:],
                                   expected[:, 1:] / start_scale(reference, flat, rows), atol=1e-4)
        assert torch.allclose(model.models[1].linear_noise_sd, torch.tensor(0.5))
        assert torch.allclose(model.models[1].conv_noise_sd, torch.tensor(0.3))
    torch.manual_seed(0)
    models = h2opt.train_linear_conv_models(partial(h2opt.ImageConvModel, 2, 15), images, groups,
                                            None, train_test, 0.5, 0.3, n_iter=20, **options)
    for model, first, rows in zip(models, start, train_test, strict=True):
        traits = h2opt.synthetic_traits(model, images, rows == 0)
        assert traits.shape == (60, 2) and np.isfinite(traits).all()
        assert not model.models[0].training
        # the linear map stays fixed and the CNN trains
        assert torch.equal(model.models[0].weight, first.models[0].weight)
        assert model.models[0].conv.lin1.weight.abs().sum() > 0


def test_train_batch_returns_every_copy_in_evaluation_mode():
    images, groups, train_test = _image_data()
    for max_copies in (None, 2):
        models = [h2opt.ImageConvModel(2, 15) for _ in range(4)]
        rows = np.concatenate([train_test, train_test])[:4] == 0
        h2opt.train_batch(models, images, groups, None, rows, np.full(4, 0.1), n_iter=3,
                          max_copies=max_copies)
        assert not any(m.training for m in models)


def test_train_linear_conv_models_returns_every_model_in_evaluation_mode():
    images, groups, train_test = _image_data()
    models = h2opt.train_linear_conv_models(partial(h2opt.ImageConvModel, 2, 15), images, groups,
                                            None, train_test, 0.5, 0.3, n_traits=2,
                                            n_linear_iter=5, n_iter=3, verbose=False)
    assert len(models) > 1
    assert not any(m.training for model in models for m in model.modules())


def test_train_linear_conv_models_rejects_a_constant_linear_trait():
    images, groups, train_test = _image_data()
    images = np.ones_like(images)
    with pytest.raises(ValueError, match='constant'):
        h2opt.train_linear_conv_models(partial(h2opt.ImageConvModel, 2, 15), images, groups, None,
                                       train_test, 0.5, 0.3, n_linear_iter=0, n_iter=0,
                                       verbose=False)
