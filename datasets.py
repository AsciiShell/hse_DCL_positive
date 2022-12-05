import typing

import torch
import numpy as np
from PIL import Image
from torchvision.datasets import STL10, CIFAR10


class STL10Pair(STL10):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: typing.Optional[typing.Callable] = None,
                 download: bool = False):
        super().__init__(root, split, None, transform, None, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target


class CIFAR10Pair(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: typing.Optional[typing.Callable] = None,
                 download: bool = False):
        super().__init__(root, train, transform, None, download)
        self.targets = np.array(self.targets)
        self.labels = self.targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target


class STL10NoisePair(STL10):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: typing.Optional[typing.Callable] = None,
                 download: bool = False, tau=None):
        super().__init__(root, split, None, transform, None, download)
        self.tau = tau

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)

            if np.random.rand() < self.tau:
                index = np.where((self.labels != target)
                                 | (self.labels == -1))[0]
                assert len(index) > 1000, "Bad query"
                img_rand = self.data[np.random.choice(index)]
                img_rand = Image.fromarray(np.transpose(img_rand, (1, 2, 0)))
                pos_2 = self.transform(img_rand)
            else:
                pos_2 = self.transform(img)

        return pos_1, pos_2, target


class CIFAR10NoisePair(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: typing.Optional[typing.Callable] = None,
                 download: bool = False,
                 tau=None):
        super().__init__(root, train, transform, None, download)
        self.targets = np.array(self.targets)
        self.labels = self.targets
        self.tau = tau

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)

            if np.random.rand() < self.tau:
                index = np.where(self.targets != target)[0]
                assert len(index) > 1000, "Bad query"
                img_rand = self.data[np.random.choice(index)]
                img_rand = Image.fromarray(img_rand)
                pos_2 = self.transform(img_rand)
            else:
                pos_2 = self.transform(img)

        return pos_1, pos_2, target


def get_dataset(name: str, root: str, split: str, transform=None, tau=None) -> torch.utils.data.Dataset:
    kwargs = {"root": root, "transform": transform}
    if "CIFAR" in name:
        kwargs["train"] = "train" in split
    if "Noise" in name:
        kwargs["tau"] = tau
    if name == "STL10":
        base_cls = STL10Pair
    elif name == "CIFAR10":
        base_cls = CIFAR10Pair
    elif name == "STL10Noise":
        base_cls = STL10NoisePair
    elif name == "CIFAR10Noise":
        base_cls = CIFAR10NoisePair
    else:
        raise Exception("Unknown dataset {}".format(name))
    return base_cls(**kwargs)
