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
    h2opt.anova_heritability(Y, groups, environment)
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


def test_henderson3_forms_give_its_heritability():
    rng = np.random.RandomState(1)
    groups = np.repeat(np.arange(20), 6)
    subgroups = np.tile(np.repeat(np.arange(2), 3), 20)
    environment = rng.randint(4, size=(120, 1))
    model = h2opt.Henderson3(groups, environment, subgroups=subgroups)
    A, B = model.forms()
    Y = rng.normal(size=(120, 5)) + rng.normal(size=(20, 5))[groups]
    ratio = np.einsum('ik,ij,jk->k', Y, A, Y) / np.einsum('ik,ij,jk->k', Y, B, Y)
    np.testing.assert_allclose(ratio, model.heritability(torch.tensor(Y)).numpy(), rtol=1e-10)
    np.testing.assert_allclose(
        h2opt.heritability(Y, groups, environment, subgroups, 'henderson3'), ratio, rtol=1e-10)
    np.testing.assert_allclose(h2opt.heritability(Y, groups, environment),
                               h2opt.anova_heritability(Y, groups, environment), rtol=1e-12)


def test_henderson3_rows_give_each_column_its_own_individuals():
    rng = np.random.RandomState(2)
    groups = np.repeat(np.arange(30), 4)
    subgroups = np.tile([0, 0, 1, 1], 30)
    environment = rng.randint(3, size=120)
    rows = rng.rand(120, 3) < [0.6, 0.8, 0.6]
    rows[:, 2] = rows[:, 0]
    Y = torch.tensor(rng.normal(size=(120, 3)) + rng.normal(size=(30, 3))[groups])
    together = h2opt.Henderson3(groups, environment, subgroups, rows=rows).heritability(Y)
    for j in range(3):
        r = rows[:, j]
        alone = h2opt.Henderson3(groups[r], environment[r], subgroups[r]).heritability(Y[r, j])
        assert together[j].item() == pytest.approx(alone.item(), rel=1e-10)
    # one column of rows serves every column of Y
    one = h2opt.Henderson3(groups, environment, subgroups, rows=rows[:, :1]).heritability(Y)
    for j in range(3):
        r = rows[:, 0]
        alone = h2opt.Henderson3(groups[r], environment[r], subgroups[r]).heritability(Y[r, j])
        assert one[j].item() == pytest.approx(alone.item(), rel=1e-10)


def test_anova_forms_give_its_heritability(sorghum):
    X, groups, environment = sorghum
    X, groups, environment = X[:300, ::50], groups[:300], environment[:300]
    A, B = h2opt.AnovaDesign(groups, environment).forms()
    w = np.random.RandomState(0).normal(size=(X.shape[1], 4))
    Y = X @ w
    expected = h2opt.anova_heritability(Y, groups, environment)
    ratio = np.einsum('ik,ij,jk->k', Y, A, Y) / np.einsum('ik,ij,jk->k', Y, B, Y)
    np.testing.assert_allclose(ratio, expected, rtol=1e-8)
