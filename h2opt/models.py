"""Models mapping HTP measurements to synthetic traits, and the autoencoder used for simulation."""

import torch
import torch.nn as nn


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
    image_size is height (= width) or (height, width). The model of the paper's Miscanthus images
    (6 channels, 63 x 63 pixels).
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


class AutoEncoder(nn.Module):
    """Tanh autoencoder with one hidden layer, which embeds latent traits into simulated spectra."""

    def __init__(self, n_features, n_latent, hidden=100):
        super().__init__()
        self.nonlin = torch.tanh
        self.linE1 = nn.Linear(n_features, hidden)
        self.linE2 = nn.Linear(hidden, n_latent)
        self.linD1 = nn.Linear(n_latent, hidden)
        self.linD2 = nn.Linear(hidden, n_features)

    def forward(self, x):
        x = self.encode(x)
        x = x + 0.005 * torch.randn(x.shape).to(x.device)
        return self.decode(x)

    def encode(self, x):
        return self.nonlin(self.linE2(self.nonlin(self.linE1(x))))

    def decode(self, x):
        return self.linD2(self.nonlin(self.linD1(x)))

    def save(self, path):
        torch.save({'n_features': self.linE1.in_features, 'n_latent': self.linE2.out_features,
                    'hidden': self.linE1.out_features, 'state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, weights_only=True)
        model = cls(checkpoint['n_features'], checkpoint['n_latent'], checkpoint['hidden'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
