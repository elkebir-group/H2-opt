"""H2-opt: self-supervised discovery of heritable traits in high-throughput phenotyping data."""

from . import baselines, selection
from .decorrelation import Decorrelation
from .heritability import (
    AnovaDesign,
    Henderson3,
    anova_heritability,
    genetic_covariance,
    grouped_variance,
    remove_environment,
)
from .io import load_npz
from .models import AutoEncoder, ConvModel, LinearModel, TraitModels
from .simulate import encode_latent, latent_scale
from .train import synthetic_traits, train, train_batch

__all__ = [
    "AnovaDesign",
    "AutoEncoder",
    "ConvModel",
    "Decorrelation",
    "Henderson3",
    "LinearModel",
    "TraitModels",
    "anova_heritability",
    "baselines",
    "selection",
    "encode_latent",
    "genetic_covariance",
    "grouped_variance",
    "latent_scale",
    "load_npz",
    "remove_environment",
    "synthetic_traits",
    "train",
    "train_batch",
]
