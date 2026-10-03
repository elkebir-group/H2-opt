import torch

import h2opt


def test_conv_model_output_shape(sorghum):
    X, _, _ = sorghum
    model = h2opt.TraitModels(2, h2opt.ConvModel, X.shape[1])
    assert model(torch.tensor(X[:5]).float()).shape == (5, 2)
