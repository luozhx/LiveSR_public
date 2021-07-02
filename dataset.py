import random

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class OnlineDataset(Dataset):
    def __init__(self, max_size=2048):
        self.max_size = max_size

        self._scale = 2
        self._data = [[] for _ in range(4)]
        self.transform = Compose([ToTensor(), ])

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        assert val in [1, 2, 3, 4]
        self._scale = val

    @property
    def data(self):
        return self._data[self.scale - 1]

    @data.setter
    def data(self, val):
        self._data[self.scale - 1] = val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index = random.randrange(0, len(self.data))
        return self.data[index]

    def empty(self) -> bool:
        return len(self.data) == 0

    def size(self) -> int:
        return len(self.data)

    def put(self, lr, hr):
        if len(self) >= self.max_size:
            self.data = self.data[self.max_size // 4:]
        lr = self.transform(lr)
        hr = self.transform(hr)
        self.data.append((lr, hr))

    def put_patches(self, patches):
        for lr, hr in patches:
            self.put(lr, hr)
