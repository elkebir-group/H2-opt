"""Models mapping HTP measurements to synthetic traits."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    """Linear map from n_features measurements to n_out traits."""

    def __init__(self, n_features, n_out=1):
        super().__init__()
        self.lin1 = nn.Linear(n_features, n_out)

    def forward(self, x):
        return self.lin1(x)


class ConvModel(nn.Module):
    """Two 1D convolutions (kernel 20, stride 5, 5 and 10 channels, ReLU), then a linear layer.

    Input is (n, n_features), e.g. a spectrum per individual.
    """

    def __init__(self, n_features, n_out=1):
        super().__init__()
        self.nonlin = nn.ReLU()
        self.conv1 = nn.Conv1d(1, 5, 20, 5)
        self.conv2 = nn.Conv1d(5, 10, 20, 5)
        length = ((n_features - 20) // 5 + 1 - 20) // 5 + 1
        self.lin1 = nn.Linear(10 * length, n_out)

    def forward(self, x):
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        x = self.nonlin(self.conv1(x))
        x = self.nonlin(self.conv2(x))
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        return self.lin1(x)


class ImageConvModel(nn.Module):
    """Two 2D convolutions (10 channels each; kernel 6, stride 3, then kernel 4, stride 3; leaky
    ReLU), then a linear layer.

    Input is (n, n_channels, height, width), e.g. a multispectral image per individual;
    image_size is height (= width) or (height, width).
    """

    def __init__(self, n_channels, image_size, n_out=1):
        super().__init__()
        self.nonlin = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(n_channels, 10, 6, stride=3)
        self.conv2 = nn.Conv2d(10, 10, 4, stride=3)
        height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
        size = [((length - 6) // 3 + 1 - 4) // 3 + 1 for length in (height, width)]
        self.lin1 = nn.Linear(10 * size[0] * size[1], n_out)

    def forward(self, x):
        x = self.nonlin(self.conv1(x))
        x = self.nonlin(self.conv2(x))
        return self.lin1(x.reshape((x.shape[0], -1)))


class LinearConvModel(nn.Module):
    """The sum of a linear map of the flattened input and a convolutional model.

    conv is a module with a final linear layer lin1 and one output, e.g. ConvModel or
    ImageConvModel. The weight of the linear map is w = weight + V, with the fixed buffers weight
    (1, m) and bias (1,), and the parameter V (1, m); m is the number of input values per
    individual. V starts at zero, and the constructor sets the weight and bias of conv.lin1 to
    zero, so the model starts at exactly the linear map x w^T + bias.

    In training mode, each branch draws its own noise, so the two branches can have any noise
    levels: the linear branch gets normal noise of standard deviation linear_noise_sd |w| per
    individual on its output, the same as input noise of standard deviation linear_noise_sd for a
    linear map; conv gets normal input noise of standard deviation conv_noise_sd. Train it with
    no input noise from train_batch (noise_sd 0).
    """

    def __init__(self, conv, weight, bias, linear_noise_sd, conv_noise_sd):
        super().__init__()
        weight = torch.as_tensor(weight, dtype=torch.float32).reshape((1, -1))
        self.register_buffer('weight', weight)
        self.register_buffer('bias', torch.as_tensor(bias, dtype=torch.float32).reshape((1,)))
        self.register_buffer('linear_noise_sd', torch.tensor(float(linear_noise_sd)))
        self.register_buffer('conv_noise_sd', torch.tensor(float(conv_noise_sd)))
        self.V = nn.Parameter(torch.zeros_like(weight))
        self.conv = conv
        nn.init.zeros_(conv.lin1.weight)
        nn.init.zeros_(conv.lin1.bias)

    def forward(self, x):
        w = self.weight + self.V
        linear = F.linear(x.reshape((x.shape[0], -1)), w, self.bias)
        if self.training:
            # |w| as the root of a sum with a small constant: the gradient of the norm at w = 0
            # is NaN
            linear = linear + (torch.randn((x.shape[0], 1), device=x.device)
                               * self.linear_noise_sd * torch.sqrt((w ** 2).sum() + 1e-12))
            x = x + torch.randn_like(x) * self.conv_noise_sd
        return linear + self.conv(x)


class TraitModels(nn.Module):
    """One independent model per synthetic trait.

    TraitModels(n_traits, model_class, *args) builds model_class(*args) for each trait;
    forward(x, traits) returns an (n, len(traits)) tensor with the output of the selected traits.
    """

    def __init__(self, n_traits, model_class, *args, **kwargs):
        super().__init__()
        self.n_traits = n_traits
        self.models = nn.ModuleList([model_class(*args, **kwargs) for _ in range(n_traits)])

    def forward(self, x, traits=None):
        if traits is None:
            traits = range(self.n_traits)
        out = torch.zeros((x.shape[0], len(traits))).to(x.device)
        for a, trait in enumerate(traits):
            out[:, a] = self.models[trait](x)[:, 0]
        return out
