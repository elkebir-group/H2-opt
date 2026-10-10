import torch

import h2opt


def test_conv_model_output_shape(sorghum):
    X, _, _ = sorghum
    model = h2opt.TraitModels(2, h2opt.ConvModel, X.shape[1])
    assert model(torch.tensor(X[:5]).float()).shape == (5, 2)


def test_image_conv_model_pools_over_the_image():
    model = h2opt.TraitModels(2, h2opt.ImageConvModel, 6)
    assert model.models[0].lin1.in_features == 10
    assert model(torch.rand(5, 6, 63, 63)).shape == (5, 2)
    # any image size of at least 15 x 15
    assert model(torch.rand(5, 6, 15, 20)).shape == (5, 2)



def test_position_image_conv_model_has_a_weight_per_position():
    model = h2opt.TraitModels(2, h2opt.PositionImageConvModel, 6, 63)
    # 63 -> 20 -> 6 positions per side, 10 channels
    assert model.models[0].lin1.in_features == 10 * 6 * 6
    assert model(torch.rand(5, 6, 63, 63)).shape == (5, 2)
    assert h2opt.PositionImageConvModel(6, (63, 30))(torch.rand(5, 6, 63, 30)).shape == (5, 1)


def test_linear_conv_model_is_the_linear_map_plus_conv():
    torch.manual_seed(0)
    linear = h2opt.LinearModel(6 * 20 * 20)
    conv = h2opt.ImageConvModel(6)
    model = h2opt.LinearConvModel(conv, linear.lin1.weight.detach(), linear.lin1.bias.detach(),
                                  0.5, 0.3)
    x = torch.rand(5, 6, 20, 20)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x), linear(x.reshape((5, -1))) + conv(x))
    # both branches train: the linear map is a parameter, and conv keeps its random start
    names = {name.split('.')[0] for name, _ in model.named_parameters()}
    assert names == {'weight', 'bias', 'conv'}
    assert conv.lin1.weight.abs().sum() > 0
    # in training mode each branch is noisy
    model.train()
    assert not torch.equal(model(x), model(x))
    only_conv = h2opt.LinearConvModel(h2opt.ImageConvModel(6), torch.zeros(2400),
                                      torch.zeros(1), 0.0, 0.3)
    only_conv.train()
    assert not torch.equal(only_conv(x), only_conv(x))
