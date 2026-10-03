import numpy as np
import pytest
import torch

import h2opt


def test_decorrelate_gives_orthonormal_columns():
    Y = np.random.RandomState(0).normal(size=(500, 4)) @ np.triu(np.ones((4, 4)))
    Y = h2opt.decorrelate(torch.tensor(Y).float())
    np.testing.assert_allclose((Y.T @ Y / Y.shape[0]).numpy(), np.eye(4), atol=1e-4)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_train_on_cuda(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    model = h2opt.TraitModels(2, h2opt.LinearModel, X.shape[1])
    h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_traits=2,
                n_iter=5, device='cuda', verbose=False)
    assert next(model.parameters()).is_cuda
