import h2opt


def test_encode_latent_shape(sorghum, examples):
    X, _, _ = sorghum
    autoencoder = h2opt.AutoEncoder.load(examples / 'autoencoder.pt')
    latent = h2opt.load_npz(examples / 'simulatedLatentTraits.npz')
    assert h2opt.encode_latent(latent, autoencoder, X).shape == (latent.shape[0], X.shape[1])
