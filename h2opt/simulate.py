"""Simulated HTP measurements: embed latent traits into spectra with a pretrained autoencoder."""

import numpy as np
import torch


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
