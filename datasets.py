import typing

import numpy as np
from PIL import Image
from torchvision.datasets import STL10, CIFAR10


class STL10Pair(STL10):
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target


class CIFAR10Pair(CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = self.targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target
    

class CIFAR10NoisePair(CIFAR10):
    def __init__(self, tau, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = self.targets
        self.tau = tau

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            
            if np.random.rand() < self.tau:
                img_rand = self.data[np.random.choice(self.data.shape[0])]
                img_rand = Image.fromarray(img_rand)
                pos_2 = self.transform(img_rand)
            else:
                pos_2 = self.transform(img)

        return pos_1, pos_2, target


def get_dataset(name: str) -> typing.Union[typing.Type[STL10Pair], typing.Type[CIFAR10Pair], typing.Type[CIFAR10NoisePair]]:
    if name == "STL10":
        return STL10Pair
    if name == "CIFAR10":
        return CIFAR10Pair
    if name == "CIFAR10Noise":
        return CIFAR10NoisePair
    raise Exception("Unknown dataset {}".format(name))
