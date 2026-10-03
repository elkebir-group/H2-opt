import numpy as np
import pytest
import torch

import h2opt


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
