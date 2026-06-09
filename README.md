# H2-opt
H2-opt: A novel self-supervised algorithm to mine high-throughput phenotyping data for genetically-driven traits

<p align="center">
  <img width="400" height="220" src="./overview.png">
</p>

## Running H2-opt

### Requirements

The H2-opt software requires Python3 with the packages PyTorch and numpy. 

### Calculating ANOVA heritability 
Let n be the number of individuals. Define "traits" as an n by k PyTorch tensor of phenotypes. Define "groups" as a length n integer array representing genetically related groups such as clones. Define "environments" as an n by g matrix of categorical variables, where g is the number of environmental variables (and can be zero). Then, the heritability can be calculated as follows in Python. 

```python
from shared import ANOVAHeritability
H = ANOVAHeritability(traits, groups, environment)
```
Specifically, if groups represent clones, this directly gives the broad-sense heritability. If groups have genetic relatedness Gamma, then narrow-sense heritability is H / Gamma. 
A full example of calculating the heritability of the first 10 wavelengths in our sorghum hyperspectral measurement dataset is given below.
```python
import numpy as np
import torch
from H2-opt import loadnpz, ANOVAHeritability


X = np.concatenate((loadnpz('./data/examples/X_file1.npz'), loadnpz('./data/examples/X_file2.npz')), axis=0)
groups = loadnpz('./data/examples/genotypes.npz')
environment = loadnpz('./data/examples/environment.npz')
X_example = torch.tensor(X[:, :10]).float()

heritability = ANOVAHeritability(X_example, groups, environment)
```
The output "heritability" is the tensor of the 10 heritability values tensor([0.0530, 0.0322, 0.0545, 0.0715, 0.0774, 0.0755, 0.0564, 0.0422, 0.0479, 0.0503]). 
As a minor aside, the measurement data X is split into two files due to GitHub file size limits. 

### Optimizing heritability 
Let n be the number of individuals. Define "groups" as a length n integer array representing genetically related groups such as clones. Define "environments" as an n by g matrix of categorical variables, where g is the number of environmental variables (and can be zero). Define "model" as the PyTorch model that determines the synthetic traits and will be trained. Define "X" as the HTP measurement data tensor (with the first axis having length n). Define "trainTest" as a numpy array of length n, with values 0 indicating individuals in the training set and values 1 indicating individuals in the test set. Define "modelFile" as the location for the trained model to be saved. The minimal usage of H2-opt heritability optimization is as below in Python. 
```python
from shared import trainModel
trainModel(model, X, groups, environment, trainTest, modelFile)
```
Additional optional parameters include the following. ``Nphen`` is the number of phenotypes to extract, which is denoted by k in the H2-opt manuscript. By default, Nphen = 1. "learningRate" is the Pytorch learning rate with a default of 1e-4. noiseLevel is data augmentation-based regularization level with a default value of 0.1. The below code sets these values. 
```python
from shared import trainModel
trainModel(model, X, groups, environment, trainTest, modelFile, Nphen=Nphen, learningRate=learningRate, noiseLevel=noiseLevel)
```
Below is a full example of training a linear model to extract 10 synthetic traits on our sorghum hyperspectral measurement dataset. 
```python
import numpy as np
from H2-opt import loadnpz, trainModel, multiConv, simpleModel

X = np.concatenate((loadnpz('./data/examples/X_file1.npz'), loadnpz('./data/examples/X_file2.npz')), axis=0)
groups = loadnpz('./data/examples/genotypes.npz')
environment = loadnpz('./data/examples/environment.npz')
trainTest = np.zeros(genotype.shape[0], dtype=int)

Nphen = 10
model = multiConv(Nphen, [X.shape[1], 1], simpleModel)

trainModel(model, X, groups, environment, trainTest, './model.pt', Niter=10000, doPrint=True, Nphen=Nphen, learningRate=1e-5, noiseLevel=0.005)
```

This code results in the final model being trained and saved as a PyTorch model file in ./model.pt. An example model is saved in "./data/examples/mode.pt". 



### Generating simulated hyperspectral measurements 

Let "modelFile" be the location of the autoencoder being used, with "modelFile = './data/examples/autoencoder.pt'" being used for our pretrained autoencoder. Let "latentTraits" be the array of traits one wants to embed in hyperspectral measurements. The number of traits in "latentTraits" should be 5 (including any random traits) if one uses "modelFile = './data/examples/autoencoder.pt'". Let "referenceMeasurements" be our reference set of hyperspectral measurements. Then, one can use "encodeValues" to encode hyperspectral measurements. Below is an example using the simulated latent traits (generated using simplePHENOTYPES) from our manuscript. 
```python
import numpy as np
from H2-opt import loadnpz, encodeValues

modelFile = './data/examples/autoencoder.pt'
referenceMeasurements = np.concatenate((loadnpz('./data/examples/X_file1.npz'), loadnpz('./data/examples/X_file2.npz')), axis=0)

X_latent = loadnpz('./data/examples/simulatedLatentTraits.npz')
X_final = encodeValues(X_latent, modelFile, referenceMeasurements)
```

