import typing

import torch
import numpy as np
from PIL import Image
from torchvision.datasets import STL10, CIFAR10


class STL10Pair(STL10):
    def __init__(self,
                 root: str,
                 split: str,
                 num_pos: int,
                 transform: typing.Callable,
                 download: bool = False,
                 noise_frac: typing.Optional[float] = None):
        super().__init__(root, split, None, transform, None, download)
        self.noise_frac = noise_frac
        self.num_pos = num_pos
        assert self.transform is not None, "Empty transform"

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        pos_1 = self.transform(img)

        if self.noise_frac is not None and np.random.rand() < self.noise_frac:
            index = np.where((self.labels != target) | (self.labels == -1))[0]
            assert len(index) > 1000, "Bad query"
            img_rand = self.data[np.random.choice(index)]
            img_rand = Image.fromarray(np.transpose(img_rand, (1, 2, 0)))
            pos_2 = self.transform(img_rand)
        else:
            pos_2 = self.transform(img)

        pos_m = []
        for _ in range(self.num_pos - 1):
            pos_m.append(self.transform(img))

        return pos_1, pos_2, pos_m, target


class CIFAR10Pair(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool,
                 num_pos: int,
                 transform: typing.Callable,
                 download: bool = False,
                 noise_frac: typing.Optional[float] = None):
        super().__init__(root, train, transform, None, download)
        self.targets = np.array(self.targets)
        self.labels = self.targets
        self.noise_frac = noise_frac
        self.num_pos = num_pos
        assert self.transform is not None, "Empty transform"

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        pos_1 = self.transform(img)

        if self.noise_frac is not None and np.random.rand() < self.noise_frac:
            index = np.where(self.targets != target)[0]
            assert len(index) > 1000, "Bad query"
            img_rand = self.data[np.random.choice(index)]
            img_rand = Image.fromarray(img_rand)
            pos_2 = self.transform(img_rand)
        else:
            pos_2 = self.transform(img)
        pos_m = []
        for _ in range(self.num_pos - 1):
            pos_m.append(self.transform(img))

        return pos_1, pos_2, pos_m, target


def get_dataset(name: str, root: str, split: str, num_pos: int, transform=None, noise_frac=None) -> torch.utils.data.Dataset:
    kwargs = {"root": root, "transform": transform, "num_pos": num_pos}
    if "CIFAR" in name:
        kwargs["train"] = "train" in split
    else:
        kwargs["split"] = split
    if "Noise" in name:
        kwargs["noise_frac"] = noise_frac
    if name == "STL10":
        base_cls = STL10Pair
    elif name == "CIFAR10":
        base_cls = CIFAR10Pair
    else:
        raise Exception("Unknown dataset {}".format(name))
    return base_cls(**kwargs)
