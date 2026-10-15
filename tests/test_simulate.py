import numpy as np

import h2opt


def test_encode_latent(sorghum, examples):
    X, _, _ = sorghum
    autoencoder = h2opt.AutoEncoder.load(examples / 'autoencoder.pt')
    latent = h2opt.load_npz(examples / 'simulatedLatentTraits.npz')
    latent = (latent - latent.mean(axis=0)) / latent.std(axis=0)
    means, stds = h2opt.latent_scale(autoencoder, X)
    residuals = h2opt.reconstruction_residuals(autoencoder, X)
    assert residuals.shape == X.shape
    np.testing.assert_allclose(residuals.mean(axis=0), 0, atol=1e-6)
    simulated = h2opt.encode_latent(latent, autoencoder, (means, stds), residuals,
                                    np.random.RandomState(0))
    assert simulated.shape == (latent.shape[0], X.shape[1])
    # the encodings of the simulated measurements follow the reference scale
    encoded_means, _ = h2opt.latent_scale(autoencoder, simulated)
    assert np.abs(encoded_means - means).max() < 0.5 * stds.max()
    # without the residuals, the measurements are an exact linear function of the latent traits
    drawn = residuals[np.random.RandomState(0).randint(len(residuals), size=len(latent))]
    decoded = simulated - drawn
    design = np.c_[np.ones(len(latent)), latent]
    coefficients, *_ = np.linalg.lstsq(design, decoded, rcond=None)
    remainder = decoded - design @ coefficients
    assert (remainder ** 2).sum() < 1e-8 * ((decoded - decoded.mean(0)) ** 2).sum()
