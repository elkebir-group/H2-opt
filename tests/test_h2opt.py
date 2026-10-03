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


def test_baselines(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::10]
    train = np.random.RandomState(0).randint(5, size=len(groups)) != 0
    herit = {}
    for name, baseline in [('pca', h2opt.baselines.PCA(3)), ('gpca', h2opt.baselines.GeneticPCA(3)),
                           ('lda', h2opt.baselines.LDA(3)), ('pch', h2opt.baselines.PCH(3)),
                           ('max', h2opt.baselines.MaxHeritabilityFeatures(3))]:
        Y = baseline.fit(X[train], groups[train], environment[train]).transform(X)
        assert Y.shape == (len(groups), 3)
        # traits are uncorrelated on the training individuals
        np.testing.assert_allclose(np.corrcoef(Y[train].T), np.eye(3), atol=1e-6)
        herit[name] = h2opt.anova_heritability(torch.tensor(Y[train]).float(), groups[train], environment[train]).numpy()
    # unregularized PCH maximizes the training heritability among linear traits
    assert all(herit['pch'][0] >= h[0] - 1e-4 for h in herit.values())


def test_pch_more_measurements_than_individuals(sorghum):
    X, groups, environment = sorghum
    rows = np.arange(200)
    Y = h2opt.baselines.PCH(2, ridge=1e-3).fit_transform(X[rows], groups[rows], environment[rows])
    assert Y.shape == (200, 2) and np.all(np.isfinite(Y))


def test_henderson3_is_unbiased_with_confounded_environment():
    groups = np.repeat(np.arange(60), 15)
    subgroups = np.tile(np.repeat(np.arange(3), 5), 60)
    plot = groups * 3 + subgroups
    perm = np.random.RandomState(0).permutation(180)
    environment = np.c_[(perm // 90)[plot], (perm % 30)[plot]]
    model = h2opt.Henderson3(groups, environment, subgroups=subgroups)
    estimates = []
    for seed in range(200):
        r = np.random.RandomState(seed)
        y = (r.normal(size=60)[groups] + 0.7 * r.normal(size=180)[plot] + r.normal(size=900)
             + r.normal(size=30)[environment[:, 1]] + 2 * environment[:, 0])
        estimates.append(model.components(torch.tensor(y)).numpy()[:, 0])
    np.testing.assert_allclose(np.mean(estimates, axis=0), [1.0, 0.49, 1.0], atol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_train_on_cuda(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    model = h2opt.TraitModels(2, h2opt.LinearModel, X.shape[1])
    h2opt.train(model, X, groups, environment, np.zeros(len(groups), dtype=int), n_traits=2, n_iter=5,
                device='cuda', verbose=False)
    assert next(model.parameters()).is_cuda


def test_linear_h2opt_matches_pch_without_ridge(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    model = h2opt.LinearH2opt(2, ridge=1e-8, n_iter=30).fit(X, groups, environment, validation=(X, groups, environment))
    Y = model.transform(X)
    herit = h2opt.anova_heritability(torch.tensor(Y), groups, environment).numpy()
    pch = h2opt.baselines.PCH(2).fit_transform(X, groups, environment)
    herit_pch = h2opt.anova_heritability(torch.tensor(pch), groups, environment).numpy()
    # L-BFGS converges to the PCH optimum, which maximizes heritability among linear traits
    assert herit[0] == pytest.approx(herit_pch[0], abs=1e-3)
    assert abs(np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]) < 1e-6
    assert model.validation_curves_.shape == (2, 30)
