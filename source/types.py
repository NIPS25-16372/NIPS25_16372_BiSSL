from typing import Literal

PretextTaskTypes = Literal["simclr", "byol"]
DownstreamTaskTypes = Literal["classification", "object_detection"]

ConvBackbones = Literal["resnet18", "resnet50"]
BackboneArchs = Literal[ConvBackbones]

### Datasets ###
DatasetsClassification = Literal[
    "stl10",
    "food",
    "cars",
    "dtd",
    "pets",
    "flowers",
    "aircrafts",
    "cifar10",
    "cifar100",
    "imagenet",
    "caltech101",
    "sun397",
    "voc07",
    "cub200",
]

DownstreamDatasets = Literal[DatasetsClassification]
PretrainDatasets = Literal["imagenet", "stl10"]
Datasets = Literal[PretrainDatasets, DownstreamDatasets]
DatasetSplits = Literal["unlabeled", "train", "val", "test", "trainval"]

OptimizerChoices = Literal["sgd", "adam", "adamw", "lars"]

# BiSSL Specific
HinvSolverTypes = Literal["cg"]
UpperLower = Literal["upper", "lower"]

# General Types
BinaryChoices = Literal[0, 1]
FloatBetween0and1 = float
FloatNonNegative = float
