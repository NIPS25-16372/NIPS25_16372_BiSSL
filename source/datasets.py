from typing import Literal, Any, Optional, Callable, List
import urllib

import numpy as np
from torchvision import datasets
from torchvision.transforms import Compose, Grayscale
from torch.utils.data import Dataset

from source.types import Datasets, DatasetSplits
from source.dataset_attr.classes import flowers_classes
from source.dataset_attr.modified_datasets import (
    Caltech101GreyToRGB,
    SUN397WPartitions,
    VOC2007Classification,
    Cub2011,
    VOCObjectDetectionDataset,
)


class GetData:
    def __init__(self, root: str, download: bool = False):
        self.root = root
        self.download = download

    def _get_partitioned_data(
        self,
        dataset_name: Datasets,
        split: Literal["train", "val", "test"],
        data: List[Any] | np.ndarray[Any, np.dtype[Any]],
        return_separate_data_labels: bool = False,
    ) -> (
        tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]
        | tuple[tuple[Any, ...], tuple[Any, ...]]
        | List[np.ndarray[Any, np.dtype[Any]] | Any]
    ):

        return_np_arrays = True if isinstance(data, np.ndarray) else False

        dataset_idxs: np.ndarray[Any, np.dtype[Any]] = np.loadtxt(
            fname=self.root + "/tv_splits/" + dataset_name + "_" + split + "_split.txt",
            dtype=int,
        )

        samples_subset = [data[i] for i in dataset_idxs]

        if return_separate_data_labels:
            samps, labels = zip(*samples_subset)
            if return_np_arrays:
                return np.array(samps), np.array(labels)
            return samps, labels

        return samples_subset

    def __call__(
        self,
        dataset_name: Datasets,
        split: DatasetSplits = "train",
        transform: Optional[Callable] = None,
    ) -> Dataset:
        match dataset_name:
            case "stl10":
                assert split in ("unlabeled", "train", "test", "val")
                dataset = datasets.STL10(
                    root=self.root,
                    split=split if split != "val" else "train",
                    download=self.download,
                    transform=transform,
                )
                if split in ("train", "val"):
                    dataset.data, dataset.labels = self._get_partitioned_data(  # type: ignore
                        dataset_name=dataset_name,
                        split=split,
                        data=list(zip(dataset.data, dataset.labels)),  # type: ignore
                        return_separate_data_labels=True,
                    )
                return dataset

            case "food":
                assert split in ("train", "test", "val")

                dataset = datasets.Food101(
                    root=self.root,
                    split=split if split != "val" else "train",
                    download=self.download,
                    transform=transform,
                )
                if split in ("train", "val"):
                    dataset._image_files, dataset._labels = self._get_partitioned_data(  # type: ignore
                        dataset_name=dataset_name,
                        split=split,
                        data=list(zip(dataset._image_files, dataset._labels)),
                        return_separate_data_labels=True,
                    )

                return dataset

            case "cars":
                assert split in ("train", "test", "val")
                dataset = datasets.StanfordCars(
                    root=self.root,
                    split=split if split != "val" else "train",
                    download=self.download,
                    transform=transform,
                )
                if split in ("train", "val"):
                    dataset._samples = self._get_partitioned_data(  # type: ignore
                        dataset_name=dataset_name,
                        split=split,
                        data=dataset._samples,
                    )
                return dataset

            case "dtd":
                assert split in ("train", "test", "val")
                dataset = datasets.DTD(
                    root=self.root,
                    split=split,
                    download=self.download,
                    transform=transform,
                )
                return dataset

            case "pets":
                assert split in ("train", "test", "val")
                split_get = "trainval" if split in ("train", "val") else "test"
                dataset = datasets.OxfordIIITPet(
                    root=self.root,
                    split=split_get,
                    download=self.download,
                    transform=transform,
                )

                if split in ("train", "val"):
                    dataset._images, dataset._labels = self._get_partitioned_data(  # type: ignore
                        dataset_name=dataset_name,
                        split=split,
                        data=list(zip(dataset._images, dataset._labels)),
                        return_separate_data_labels=True,
                    )

                return dataset

            case "flowers":
                assert split in ("train", "val", "test")
                dataset = datasets.Flowers102(
                    root=self.root,
                    split=split,
                    download=self.download,
                    transform=transform,
                )
                dataset.classes = flowers_classes  # type: ignore

                return dataset

            case "aircrafts":
                assert split in ("train", "trainval", "test", "val")
                dataset = datasets.FGVCAircraft(
                    root=self.root,
                    split=split,
                    download=self.download,
                    transform=transform,
                )

                return dataset

            case "cifar10":
                assert split in ("train", "test", "val")
                train = True if split in ("train", "val") else False

                dataset = datasets.CIFAR10(
                    root=self.root,
                    train=train,
                    download=self.download,
                    transform=transform,
                )
                if split in ("train", "val"):
                    dataset.data, dataset.targets = self._get_partitioned_data(  # type: ignore
                        dataset_name=dataset_name,
                        split=split,
                        data=list(zip(dataset.data, dataset.targets)),
                        return_separate_data_labels=True,
                    )

                return dataset

            case "cifar100":
                assert split in ("train", "test", "val")
                train = True if split in ("train", "val") else False

                dataset = datasets.CIFAR100(
                    root=self.root,
                    train=train,
                    download=self.download,
                    transform=transform,
                )

                if split in ("train", "val"):
                    dataset.data, dataset.targets = self._get_partitioned_data(  # type: ignore
                        dataset_name=dataset_name,
                        split=split,
                        data=list(zip(dataset.data, dataset.targets)),
                        return_separate_data_labels=True,
                    )
                return dataset

            case "imagenet":
                assert split in ("train", "test")

                split = "val" if split == "test" else "train"

                dataset = datasets.ImageNet(
                    root=self.root + "/imagenet",
                    # root="/tmp/imagenet",
                    split=split,
                    transform=transform,
                )

                return dataset

            case "caltech101":
                assert split in ("train", "test", "val")
                dataset = Caltech101GreyToRGB(
                    root=self.root,
                    # root="/tmp",
                    download=self.download,
                    transform=transform,
                )
                dataset.index, dataset.y = self._get_partitioned_data(  # type: ignore
                    dataset_name=dataset_name,
                    split=split,
                    data=list(zip(dataset.index, dataset.y)),
                    return_separate_data_labels=True,
                )
                return dataset

            case "sun397":
                assert split in ("train", "test", "val")
                dataset = SUN397WPartitions(
                    root=self.root,
                    split=split if split != "val" else "train",
                    download=self.download,
                    transform=transform,
                )
                if split in ("train", "val"):
                    dataset._image_files, dataset._labels = self._get_partitioned_data(  # type: ignore
                        dataset_name=dataset_name,
                        split=split,
                        data=list(zip(dataset._image_files, dataset._labels)),
                        return_separate_data_labels=True,
                    )

                return dataset

            case "voc07":
                assert split in ("train", "test", "val")
                dataset = VOC2007Classification(
                    root=self.root,
                    image_set=split,
                    download=self.download,
                    transform=transform,
                )

                return dataset

            case "voc07detection":
                assert split in ("train", "test", "val")
                dataset = VOCObjectDetectionDataset(
                    root=self.root,
                    year="2007",
                    image_set=split,
                    download=self.download,
                    transform=transform,
                )
                return dataset

            case "voc12detection":
                assert split in ("train", "test", "val")
                dataset = VOCObjectDetectionDataset(
                    root=self.root,
                    year="2012",
                    image_set=split,  # type: ignore
                    download=self.download,
                    transform=transform,
                )
                return dataset

            case "voc07+12detection":
                assert split in ("train", "test", "val")
                dataset = VOCObjectDetectionDataset(
                    root=self.root,
                    year="2007+2012",
                    image_set=split,  # type: ignore
                    download=self.download,
                    transform=transform,
                )
                return dataset

            case "cub200":
                assert split in ("train", "test", "val")
                dataset = Cub2011(
                    root=self.root,
                    train=False if split == "test" else True,
                    download=self.download,
                    transform=transform,
                )
                if split in ("train", "val"):
                    dataset.images, dataset.targets = self._get_partitioned_data(  # type: ignore
                        dataset_name=dataset_name,
                        split=split,
                        data=list(zip(dataset.images, dataset.targets)),
                        return_separate_data_labels=True,
                    )

                return dataset

            case _:
                raise ValueError(f'"{dataset_name}" is an invalid dataset.')
