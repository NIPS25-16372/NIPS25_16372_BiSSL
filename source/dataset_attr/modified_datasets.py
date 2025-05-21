import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import pandas as pd
import torch
from PIL import Image
from typing import Literal
from numpy.typing import NDArray
import numpy as np
from torchvision.datasets.folder import default_loader
from torchvision import datasets
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from source.dataset_attr.classes import voc_classes

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse


class Caltech101GreyToRGB(datasets.Caltech101):
    def __init__(
        self,
        root: str,
        target_type: Union[List[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            target_type=target_type,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.classes = self.categories

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )
        ).convert(
            "RGB"
        )  # This is the only altered line. This is necessary  as some images are in greyscale.

        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(
                    os.path.join(
                        self.root,
                        "Annotations",
                        self.annotation_categories[self.y[index]],
                        f"annotation_{self.index[index]:04d}.mat",
                    )
                )
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SUN397WPartitions(datasets.SUN397):
    def __init__(
        self,
        root,
        split: Literal["train", "test"] = "train",
        partition: int = 1,
        transform=None,
        target_transform=None,
        download=False,
    ):
        """This extended dataclass allows for the use of the SUN397 dataset with the provided partitions of the original dataset
        (in a similar style as to the DTD dataset). It requires the user to manually add the Partition folder in the root directory
        of the dataset. The partition folder can be obtained from the official website: https://vision.princeton.edu/projects/2010/SUN/

        Args:
            root (_type_): _description_
            split (str, optional): _description_. Defaults to "train".
            partition (int, optional): _description_. Defaults to 1.
            transform (_type_, optional): _description_. Defaults to None.
            target_transform (_type_, optional): _description_. Defaults to None.
            download (bool, optional): _description_. Defaults to False.
        """
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        assert partition in np.arange(10) + 1
        assert split in ["train", "test"]

        self.partition = partition
        self.split = split

        if split == "train":
            self._partition_file = self._data_dir / Path(
                f"Partitions/Training_{partition:02}.txt"
            )
        else:
            self._partition_file = self._data_dir / Path(
                f"Partitions/Testing_{partition:02}.txt"
            )

        assert os.path.isfile(self._partition_file)

        self._image_files = [
            self._data_dir / Path(fpath[1:])
            for fpath in np.loadtxt(self._partition_file, dtype=str)
        ]

        self._labels = [
            self.class_to_idx[
                "/".join(Path(path).relative_to(self._data_dir).parts[1:-1])
            ]
            for path in self._image_files
        ]


class VOC2007Classification(datasets.VOCDetection):
    """This is a modified version of the VOCDetection dataset which only utilises the 2007 partition for image classification.
       This modified version only returns the class integers as target (1: present, 0: abscent).

    Args:
        datasets (_type_): _description_
    """

    def __init__(
        self,
        root: str,
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            root=root,
            image_set=image_set,
            download=download,
            year="2007",
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

        self.classes = voc_classes
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target_full = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())  # type: ignore
        target = torch.zeros(len(self.classes))
        for obj in target_full["annotation"]["object"]:
            target[self.class_to_idx[obj["name"]]] = 1

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class VOCObjectDetectionDataset(datasets.VOCDetection):
    def __init__(
        self,
        root,
        year: Literal["2007", "2012", "2007+2012"] = "2007",
        image_set: Literal["train", "val", "trainval", "test"] = "train",
        transform=None,
        download=False,
    ):

        if year == "2007+2012":
            super().__init__(
                root,
                year="2007",
                image_set=image_set,
                download=download,
            )
            if image_set in ["train", "val", "trainval"]:
                data_2012 = datasets.VOCDetection(
                    root,
                    "2012",
                    image_set,
                    download,
                )

                self.images += data_2012.images
                self.targets += data_2012.targets

                del data_2012

        else:
            super().__init__(
                root,
                year=year,
                image_set=image_set,
                download=download,
            )
        self.transforms = transform

        self.classes = voc_classes

        self.class_to_idx = {
            cls: i + 1 for i, cls in enumerate(self.classes)
        }  # Start from 1 (0 is reserved for background)

        self.num_classes = len(self.classes) + 1

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")

        # Convert annotations to be compatible with Faster R-CNN format
        target = self.parse_voc_xml(ET_parse(self.annotations[idx]).getroot())  # type: ignore
        targets = dict(
            boxes=torch.zeros((len(target["annotation"]["object"]), 4)),
            labels=torch.zeros(
                (len(target["annotation"]["object"])), dtype=torch.int64
            ),
        )
        for j, obj in enumerate(target["annotation"]["object"]):
            bbox = obj["bndbox"]
            targets["boxes"][j] = torch.tensor(
                [
                    float(bbox["xmin"]),
                    float(bbox["ymin"]),
                    float(bbox["xmax"]),
                    float(bbox["ymax"]),
                ]
            )
            targets["labels"][j] = torch.tensor(
                self.class_to_idx[obj["name"]], dtype=torch.int64
            )

        if self.transforms:
            img, targets = self.transforms(img, targets)

        return img, targets


class Cub2011(Dataset):
    """Inspired by this implementation: https://github.com/TDeVries/cub2011_dataset

    Args:
        datasets (_type_): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """

    base_folder = "CUB_200_2011/images"
    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self, root, train=True, transform=None, loader=default_loader, download=True
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.images = [
            os.path.join(self.root, self.base_folder, dp.filepath)
            for _, dp in self.data.iterrows()
        ]
        self.targets = [
            dp.target - 1 for _, dp in self.data.iterrows()
        ]  # Targets start at 1 by default, so shift to 0

        class_to_idx = {}

        for img in self.images:
            cl_idx = int(img.split("/")[-2].split(".")[0]) - 1
            cl = img.split("/")[-2].split(".")[1]
            if cl not in class_to_idx.keys():
                class_to_idx[cl] = cl_idx

        self.class_to_idx = class_to_idx
        self.classes = list(class_to_idx.keys())

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.loader(self.images[idx])
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target
