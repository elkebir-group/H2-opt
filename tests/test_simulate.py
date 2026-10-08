import numpy as np

import h2opt


def test_encode_latent(sorghum, examples):
    X, _, _ = sorghum
    autoencoder = h2opt.AutoEncoder.load(examples / 'autoencoder.pt')
    latent = h2opt.load_npz(examples / 'simulatedLatentTraits.npz')
    latent = (latent - latent.mean(axis=0)) / latent.std(axis=0)
    means, stds = h2opt.latent_scale(autoencoder, X)
    simulated = h2opt.encode_latent(latent, autoencoder, (means, stds))
    assert simulated.shape == (latent.shape[0], X.shape[1])
    # the encodings of the simulated measurements follow the reference scale
    encoded_means, _ = h2opt.latent_scale(autoencoder, simulated)
    assert np.abs(encoded_means - means).max() < 0.5 * stds.max()
    # the measurements are an exact linear function of the latent traits
    design = np.c_[np.ones(len(latent)), latent]
    coefficients, *_ = np.linalg.lstsq(design, simulated, rcond=None)
    residual = simulated - design @ coefficients
    assert (residual ** 2).sum() < 1e-8 * ((simulated - simulated.mean(0)) ** 2).sum()
