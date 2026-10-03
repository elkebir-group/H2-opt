import numpy as np
import torch

import h2opt
from h2opt import baselines


def test_baselines(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::10]
    train = np.random.RandomState(0).randint(5, size=len(groups)) != 0
    herit = {}
    for name, baseline in [('pca', baselines.PCA(3)), ('gpca', baselines.GeneticPCA(3)),
                           ('lda', baselines.LDA(3)), ('pch', baselines.PCH(3)),
                           ('max', baselines.MaxHeritabilityFeatures(3))]:
        Y = baseline.fit(X[train], groups[train], environment[train]).transform(X)
        assert Y.shape == (len(groups), 3)
        # traits are uncorrelated on the training individuals
        np.testing.assert_allclose(np.corrcoef(Y[train].T), np.eye(3), atol=1e-6)
        traits = torch.tensor(Y[train]).float()
        herit[name] = h2opt.anova_heritability(traits, groups[train], environment[train]).numpy()
    # unregularized PCH maximizes the training heritability among linear traits
    assert all(herit['pch'][0] >= h[0] - 1e-4 for h in herit.values())


def test_pch_more_measurements_than_individuals(sorghum):
    X, groups, environment = sorghum
    rows = np.arange(200)
    Y = baselines.PCH(2, ridge=1e-3).fit_transform(X[rows], groups[rows], environment[rows])
    assert Y.shape == (200, 2) and np.all(np.isfinite(Y))
