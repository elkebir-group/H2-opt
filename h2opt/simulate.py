"""Simulated HTP measurements: embed latent traits into spectra with a pretrained autoencoder."""

import numpy as np
import torch


def encode_latent(latent, autoencoder, reference):
    """Decode latent traits into simulated measurements.

    latent: (n, k) latent traits, k = the autoencoder's latent size (extra columns are ignored).
    autoencoder: an AutoEncoder. reference: (n_ref, m) real measurements whose latent encodings
    set the mean and scale of each latent dimension. Each latent trait is standardized and then
    rescaled to the mean and standard deviation of the corresponding encoded reference dimension.
    Returns an (n, m) array.
    """
    with torch.no_grad():
        encoded = autoencoder.encode(torch.tensor(reference).float()).numpy()
    means = np.mean(encoded, axis=0)
    stds = np.mean((encoded - means) ** 2, axis=0) ** 0.5

    latent = latent - np.mean(latent, axis=0).reshape((1, -1))
    latent = latent / (np.mean(latent ** 2, axis=0).reshape((1, -1)) ** 0.5)
    latent = latent[:, :len(means)] * stds.reshape((1, -1)) + means.reshape((1, -1))

    with torch.no_grad():
        return autoencoder.decode(torch.tensor(latent).float()).numpy()
