from typing import Literal

PretextTaskTypes = Literal["simclr", "byol"]
DownstreamTaskTypes = Literal["classification", "object_detection"]

ConvBackbones = Literal["resnet18", "resnet50"]
BackboneArchs = Literal[ConvBackbones]
DetectionBackbones = Literal["resnet50"]

### Datasets ###
DatasetsClassification = Literal[
    "stl10",
    "food",
    "cars",
    "dtd",
    "pets",
    "flowers",
    "aircrafts",
    "celeba",
    "cifar10",
    "cifar100",
    "imagenet",
    "caltech101",
    "sun397",
    "voc07",
    "cub200",
]
DatasetsDetection = Literal["voc07detection", "voc12detection", "voc07+12detection"]
DownstreamDatasets = Literal[DatasetsClassification, DatasetsDetection]
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
