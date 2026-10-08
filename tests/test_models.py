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


def test_linear_conv_model_starts_at_the_linear_map():
    torch.manual_seed(0)
    linear = h2opt.LinearModel(6 * 20 * 20)
    model = h2opt.LinearConvModel(h2opt.ImageConvModel(6, 20), linear.lin1.weight.detach(),
                                  linear.lin1.bias.detach(), 0.5)
    x = torch.rand(5, 6, 20, 20)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x), linear(x.reshape((5, -1))), rtol=0, atol=0)
    # in training mode the linear branch is noisy, and the gradient is finite at w = 0
    model.train()
    assert not torch.equal(model(x), model(x))
    zero = h2opt.LinearConvModel(h2opt.ConvModel(200), torch.zeros(200), torch.zeros(1), 0.5)
    zero(torch.rand(5, 200)).sum().backward()
    assert torch.isfinite(zero.V.grad).all()
