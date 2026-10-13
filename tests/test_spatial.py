import numpy as np
import torch

import h2opt
from h2opt.spatial import papadakis, papadakis_torch


def field_trial(seed=0, clustered=False, genetic_noise=0.3):
    """Two reps of 60 field columns, one plot of 4 plants per column, 30 groups with 2 plots per
    rep. Measurements: a smooth field effect along the columns, a group effect, and 3 noise
    features. clustered: the plots of a group are next to each other (field and group confounded);
    otherwise they are placed at random. genetic_noise: the plant noise of the group effect."""
    rng = np.random.RandomState(seed)
    column = np.tile(np.repeat(np.arange(60), 4), 2)
    rep = np.repeat([0, 1], 240)
    plot = rep * 60 + column
    plot_group = []
    for _ in range(2):
        order = np.repeat(np.arange(30), 2)
        plot_group.append(order if clustered else rng.permutation(order))
    groups = np.concatenate(plot_group)[plot]
    field = np.sin(column / 6.0) * 2 + rep
    genetic = rng.normal(size=30)[groups]
    X = np.column_stack([field + 0.3 * rng.normal(size=480),
                         genetic + genetic_noise * rng.normal(size=480),
                         rng.normal(size=(480, 3))])
    neighbors = ((rep[:, None] == rep[None]) & (np.abs(column[:, None] - column[None]) <= 3)
                 & (plot[:, None] != plot[None]) & (groups[:, None] != groups[None])).astype(float)
    return X, groups, neighbors


def test_papadakis_leaves_no_part_predicted_by_the_neighbors():
    X, _, neighbors = field_trial()
    y = X[:, 0] + X[:, 1]
    rows = np.random.RandomState(1).rand(len(y)) < 0.5
    x = neighbors @ y / neighbors.sum(1)
    adjusted = papadakis(y, neighbors, rows)
    assert abs(np.corrcoef(adjusted[rows], x[rows])[0, 1]) < 1e-10
    assert np.var(adjusted[rows]) < np.var(y[rows])


def test_papadakis_torch_subtracts_the_centered_neighbor_mean():
    X, _, neighbors = field_trial()
    Y = X[:, :2] - X[:, :2].mean(0)
    x = neighbors @ Y / neighbors.sum(1, keepdims=True)
    got = papadakis_torch(torch.tensor(Y), torch.tensor(neighbors),
                          torch.ones(Y.shape, dtype=torch.float64))
    np.testing.assert_allclose(got.numpy(), Y - (x - x.mean(0)), atol=1e-12)


def test_papadakis_torch_ignores_individuals_outside_the_rows():
    X, _, neighbors = field_trial()
    Y = torch.tensor(X[:, :1] - X[:, :1].mean(0))
    mask = torch.tensor((np.arange(len(Y)) % 3 != 0)[:, None], dtype=torch.float64)
    changed = Y.clone()
    changed[mask[:, 0] == 0] = 100.0
    first = papadakis_torch(Y * mask, torch.tensor(neighbors), mask)
    second = papadakis_torch(changed * mask, torch.tensor(neighbors), mask)
    torch.testing.assert_close(first, second, rtol=0, atol=1e-12)


def test_training_with_neighbors_raises_the_adjusted_heritability():
    # the field effect looks more heritable than the group effect unless it is adjusted for
    X, groups, neighbors = field_trial(clustered=True, genetic_noise=2.0)
    every = np.ones(len(groups), dtype=bool)

    def adjusted_heritability(model):
        y = model(torch.tensor(X).float()).detach().numpy()[:, 0].astype(float)
        return np.ravel(h2opt.anova_heritability(papadakis(y, neighbors, every)[:, None],
                                                 groups))[0]

    scores = {}
    for name, nb in (('plain', None), ('adjusted', neighbors)):
        torch.manual_seed(0)
        model = h2opt.TraitModels(1, h2opt.LinearModel, X.shape[1])
        h2opt.train(model, X, groups, None, np.zeros(len(groups), dtype=int), n_iter=400,
                    learning_rate=1e-2, noise_sd=0.05, verbose=False, neighbors=nb)
        scores[name] = adjusted_heritability(model)
    assert scores['adjusted'] > scores['plain']


def test_select_noise_level_with_neighbors_runs_and_scores_adjusted_traits():
    X, groups, neighbors = field_trial(clustered=True)
    levels = np.array([0.01, 1.0])
    _, plain, _ = h2opt.selection.select_noise_level(
        lambda: h2opt.LinearModel(X.shape[1]), X, groups, None, levels=levels, n_splits=2,
        n_iter=20, split_units=np.arange(len(groups)))
    _, adjusted, _ = h2opt.selection.select_noise_level(
        lambda: h2opt.LinearModel(X.shape[1]), X, groups, None, levels=levels, n_splits=2,
        n_iter=20, split_units=np.arange(len(groups)), neighbors=neighbors)
    assert plain.shape == adjusted.shape
    assert not np.allclose(plain, adjusted)
