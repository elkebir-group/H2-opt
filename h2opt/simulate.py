"""Simulated HTP measurements: latent traits decoded by a pretrained autoencoder."""

import numpy as np
import torch
import torch.nn as nn


def latent_scale(autoencoder, reference):
    """Mean and standard deviation of each latent dimension over the encodings of reference, an
    (n_ref, m) array of real measurements."""
    with torch.no_grad():
        encoded = autoencoder.encode(torch.tensor(reference).float()).numpy()
    means = np.mean(encoded, axis=0)
    return means, np.mean((encoded - means) ** 2, axis=0) ** 0.5


def encode_latent(latent, autoencoder, scale):
    """Decode latent traits into simulated measurements.

    latent: (n, k) latent values in standard units, k = the autoencoder's latent size.
    scale: (means, standard deviations) of the latent dimensions, from latent_scale. Latent
    dimension j is set to means[j] + standard deviations[j] * latent[:, j] and decoded. Returns an
    (n, m) array.
    """
    means, stds = scale
    with torch.no_grad():
        return autoencoder.decode(torch.tensor(latent * stds + means).float()).numpy()


class AutoEncoder(nn.Module):
    """Tanh autoencoder with one hidden layer, which embeds latent traits into simulated spectra."""

    def __init__(self, n_features, n_latent, hidden=100):
        super().__init__()
        self.nonlin = torch.tanh
        self.linE1 = nn.Linear(n_features, hidden)
        self.linE2 = nn.Linear(hidden, n_latent)
        self.linD1 = nn.Linear(n_latent, hidden)
        self.linD2 = nn.Linear(hidden, n_features)

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
