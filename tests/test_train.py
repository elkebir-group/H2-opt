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
                n_iter=20, learning_rate=1e-3, noise_level=0.02, noise='normal', clip=None,
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
