# H2-opt
H2-opt: A novel self-supervised algorithm to mine high-throughput phenotyping data for genetically-driven traits

<p align="center">
  <img width="400" height="220" src="./overview.png">
</p>

H2-opt learns synthetic traits from high-throughput phenotyping (HTP) measurements that maximize heritability.
It needs only the measurements and labels of genetically related groups (e.g. clones or families); no genotype data is used in training.

The code that reproduces the analyses and figures of the paper is in [H2-opt-analysis](https://github.com/elkebir-group/H2-opt-analysis).

## Installation

H2-opt requires Python 3.10+ with PyTorch and numpy.

```bash
pip install .
```

## Calculating ANOVA heritability

Let n be the number of individuals.
`traits` is an n by k PyTorch tensor of phenotypes, `groups` a length-n array of labels of genetically related groups such as clones, and `environment` an n by g array of categorical environmental variables (or `None`).

```python
from h2opt import anova_heritability
H = anova_heritability(traits, groups, environment)
```

If groups are clones, this is the broad-sense heritability. If the individuals in a group have genetic relatedness r, the narrow-sense heritability is H / r.
Groups with a single individual are ignored.

`Henderson3(groups, environment, subgroups=None)` estimates heritability by Henderson's Method III, treating environment as fixed and groups (and optional subgroups nested in groups, e.g. plots of one family) as random effects.
Unlike `anova_heritability`, it stays unbiased when environment and groups are confounded, and it separates shared subgroup effects from genetic variance.
It precomputes n by n matrices for a fixed set of individuals, then `Henderson3(...).heritability(traits)` is differentiable.

Example: the heritability of the first 10 wavelengths of the sorghum hyperspectral data in `data/examples` (the measurements are split over two files because of GitHub file size limits).

```python
import numpy as np
import torch
from h2opt import load_npz, anova_heritability

X = np.concatenate((load_npz('data/examples/X_file1.npz'), load_npz('data/examples/X_file2.npz')))
groups = load_npz('data/examples/genotypes.npz')
environment = load_npz('data/examples/environment.npz')

H = anova_heritability(torch.tensor(X[:, :10]).float(), groups, environment)
# tensor([0.0530, 0.0322, 0.0545, 0.0715, 0.0774, 0.0755, 0.0564, 0.0422, 0.0479, 0.0503])
```

## Optimizing heritability

`train` learns synthetic traits one at a time, each orthogonal to the earlier ones.
`model` holds one model per trait (`TraitModels`), `X` is the n by m array of measurements, and `train_test` a length-n array with 0 for training and 1 for test individuals.
The heritability is only optimized on the training individuals.

Example: train a linear model extracting 10 synthetic traits from the sorghum data.

```python
import numpy as np
import torch
from h2opt import load_npz, train, TraitModels, LinearModel

X = np.concatenate((load_npz('data/examples/X_file1.npz'), load_npz('data/examples/X_file2.npz')))
groups = load_npz('data/examples/genotypes.npz')
environment = load_npz('data/examples/environment.npz')
train_test = np.zeros(X.shape[0], dtype=int)

n_traits = 10
model = TraitModels(n_traits, LinearModel, X.shape[1])
train(model, X, groups, environment, train_test, model_file='model.pt', n_traits=n_traits,
      n_iter=10000, learning_rate=1e-5, noise_level=0.005)

traits = model(torch.tensor(X).float())  # n by n_traits synthetic traits (before decorrelation)
```

Options: `n_iter` steps per trait (default 10000), `learning_rate` for RMSprop (default 1e-4), `noise_level` for data augmentation, the maximum of the uniform noise added to the measurements at each step (default 0.1), and `model_file` to save the model every `save_every` steps.
`ConvModel` is a convolutional alternative to `LinearModel` for spectra.

## Baselines

`h2opt.baselines` has linear baselines with a common interface: `fit(X, groups, environment)` on the training individuals, then `transform(X)`.
Their traits are centered and made uncorrelated on the training individuals, like H2-opt's.

- `PCA(n_traits)`: principal components of the measurements.
- `GeneticPCA(n_traits)`: principal components of the ANOVA estimate of the genetic covariance.
- `LDA(n_traits)`: linear discriminant analysis of the groups (unregularized).
- `PCH(n_traits, ridge)`: principal components of heritability, maximizing the ANOVA heritability above with ridge regularization.
- `MaxHeritabilityFeatures(n_traits)`: the most heritable individual features, selected greedily.

```python
from h2opt.baselines import PCH

is_train = train_test == 0
pch = PCH(n_traits=5, ridge=1e-4).fit(X[is_train], groups[is_train], environment[is_train])
traits = pch.transform(X)
```

## Generating simulated hyperspectral measurements

`encode_latent` embeds latent traits into simulated spectra with a pretrained autoencoder.
Each latent trait is matched to the mean and spread of one latent dimension of a reference set of real measurements.
The pretrained autoencoder in `data/examples/autoencoder.pt` has 5 latent dimensions.

Example with the latent traits simulated with simplePHENOTYPES in the paper:

```python
import numpy as np
from h2opt import load_npz, encode_latent, AutoEncoder

autoencoder = AutoEncoder.load('data/examples/autoencoder.pt')
reference = np.concatenate((load_npz('data/examples/X_file1.npz'), load_npz('data/examples/X_file2.npz')))
latent = load_npz('data/examples/simulatedLatentTraits.npz')

X_simulated = encode_latent(latent, autoencoder, reference)
```

## Tests

```bash
pip install .[test]
pytest tests
```
