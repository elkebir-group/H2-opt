import numpy as np
import pytest
import torch

import h2opt

EXAMPLES = 'data/examples/'


@pytest.fixture(scope='module')
def sorghum():
    X = np.concatenate((h2opt.load_npz(EXAMPLES + 'X_file1.npz'), h2opt.load_npz(EXAMPLES + 'X_file2.npz')))
    return X, h2opt.load_npz(EXAMPLES + 'genotypes.npz'), h2opt.load_npz(EXAMPLES + 'environment.npz')


def test_heritability_matches_paper_example(sorghum):
    X, groups, environment = sorghum
    H = h2opt.anova_heritability(torch.tensor(X[:, :10]).float(), groups, environment)
    expected = [0.0530, 0.0322, 0.0545, 0.0715, 0.0774, 0.0755, 0.0564, 0.0422, 0.0479, 0.0503]
    np.testing.assert_allclose(H.numpy(), expected, atol=1e-4)


def test_heritability_does_not_modify_input(sorghum):
    X, groups, environment = sorghum
    Y = torch.tensor(X[:, :3]).float()
    Y_copy = Y.clone()
    h2opt.anova_heritability(Y, groups, environment, return_variance=True)
    assert torch.equal(Y, Y_copy)


def test_heritability_of_pure_genetic_trait_is_one():
    groups = np.repeat(np.arange(50), 4)
    Y = torch.tensor(np.random.RandomState(0).normal(size=50)[groups]).float().reshape((-1, 1))
    assert h2opt.anova_heritability(Y, groups, None).item() == pytest.approx(1.0, abs=1e-5)


def test_decorrelate_gives_orthonormal_columns():
    Y = torch.tensor(np.random.RandomState(0).normal(size=(500, 4)) @ np.triu(np.ones((4, 4)))).float()
    Y = h2opt.decorrelate(Y)
    np.testing.assert_allclose((Y.T @ Y / Y.shape[0]).numpy(), np.eye(4), atol=1e-4)


def test_train_increases_heritability(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    torch.manual_seed(0)
    model = h2opt.TraitModels(1, h2opt.LinearModel, X.shape[1])
    before = h2opt.anova_heritability(model(torch.tensor(X).float()).detach(), groups, environment).item()
    h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_iter=50,
                learning_rate=1e-3, noise_level=0.005, verbose=False)
    after = h2opt.anova_heritability(model(torch.tensor(X).float()).detach(), groups, environment).item()
    assert after > before


def test_conv_model_output_shape(sorghum):
    X, _, _ = sorghum
    model = h2opt.TraitModels(2, h2opt.ConvModel, X.shape[1])
    assert model(torch.tensor(X[:5]).float()).shape == (5, 2)


def test_encode_latent_shape(sorghum):
    X, _, _ = sorghum
    autoencoder = h2opt.AutoEncoder.load(EXAMPLES + 'autoencoder.pt')
    latent = h2opt.load_npz(EXAMPLES + 'simulatedLatentTraits.npz')
    assert h2opt.encode_latent(latent, autoencoder, X).shape == (latent.shape[0], X.shape[1])
