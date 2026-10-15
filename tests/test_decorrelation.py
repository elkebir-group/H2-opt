import numpy as np

import h2opt


def test_decorrelation_is_sequential_least_squares():
    rng = np.random.RandomState(0)
    Y = rng.normal(size=(200, 3)) @ np.triu(np.ones((3, 3))) + 5
    fit = np.arange(200) < 150
    decorrelation = h2opt.Decorrelation().fit(Y[fit])
    out = decorrelation.transform(Y)
    np.testing.assert_allclose(np.corrcoef(out[fit].T), np.eye(3), atol=1e-10)
    # trait 3: residual of trait 3 on an intercept and traits 1-2, coefficients from the fit rows
    design = np.column_stack([np.ones(fit.sum()), Y[fit, :2]])
    coef = np.linalg.lstsq(design, Y[fit, 2], rcond=None)[0]
    residual = Y[:, 2] - np.column_stack([np.ones(200), Y[:, :2]]) @ coef
    np.testing.assert_allclose(out[:, 2], residual, atol=1e-10)
    np.testing.assert_allclose(out[:, 0], Y[:, 0] - Y[fit, 0].mean(), atol=1e-10)
