import numpy as np
import pytest
import torch

import h2opt


def test_linear_h2opt_matches_pch_without_ridge(sorghum):
    X, groups, environment = sorghum
    X = X[:, ::20]
    model = h2opt.LinearH2opt(2, ridge=1e-8, n_iter=30)
    Y = model.fit(X, groups, environment, validation=(X, groups, environment)).transform(X)
    herit = h2opt.anova_heritability(torch.tensor(Y), groups, environment).numpy()
    pch = h2opt.baselines.PCH(2).fit_transform(X, groups, environment)
    herit_pch = h2opt.anova_heritability(torch.tensor(pch), groups, environment).numpy()
    # L-BFGS converges to the PCH optimum, which maximizes heritability among linear traits
    assert herit[0] == pytest.approx(herit_pch[0], abs=1e-3)
    assert abs(np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]) < 1e-6
    assert model.validation_curves_.shape == (2, 30)
