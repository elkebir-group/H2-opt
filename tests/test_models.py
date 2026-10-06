import torch

import h2opt


def test_conv_model_output_shape(sorghum):
    X, _, _ = sorghum
    model = h2opt.TraitModels(2, h2opt.ConvModel, X.shape[1])
    assert model(torch.tensor(X[:5]).float()).shape == (5, 2)


def test_image_conv_model_output_shape():
    model = h2opt.TraitModels(2, h2opt.ImageConvModel, 6, 63)
    assert model.models[0].lin1.in_features == 360
    assert model(torch.rand(5, 6, 63, 63)).shape == (5, 2)
