import os
import torch
import numpy as np
import cv2
import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import AverageMeter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from loss import *
import torchvision
import scipy.io as io
from PIL import Image


class SaliconT(Dataset):
    def __init__(self, root, df_x, df_y, transform=None, size=(352, 352)) -> None:
        super().__init__()
        self.root = root
        self.img_transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        self.size = size
        self.images = df_x.tolist()
        self.maps = df_y.tolist()

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.images[idx])
        map_path = os.path.join(self.root, 'maps', self.maps[idx])

        image = Image.open(img_path).convert("RGB")
        # ground-truth
        map = np.array(Image.open(map_path).convert("L"))
        map = map.astype('float')
        map2 = cv2.resize(map, (self.size[0] // 8, self.size[1] // 8))
        map3 = cv2.resize(map, (self.size[0] // 4, self.size[1] // 4))
        map = cv2.resize(map, (self.size[1], self.size[0]))

        # transform
        image = self.img_transform(image)
        if np.max(map) > 1.0:
            map = map / 255.0
        assert np.min(map) >= 0.0 and np.max(map) <= 1.0

        return image, torch.FloatTensor(map), torch.FloatTensor(map2), torch.FloatTensor(map3)

    def __len__(self):
        return len(self.images)


class SaliconVal(Dataset):
    def __init__(self, root, df_x, df_y, transform=None, size=(352, 352)) -> None:
        super().__init__()
        self.root = root
        self.img_transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        self.size = size
        self.images = df_x.tolist()
        self.maps = df_y.tolist()

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.images[idx])
        map_path = os.path.join(self.root, 'maps', self.maps[idx])

        image = Image.open(img_path).convert("RGB")
        # ground-truth
        map = np.array(Image.open(map_path).convert("L"))
        map = map.astype('float')
        map = cv2.resize(map, (self.size[1], self.size[0]))

        # transform
        image = self.img_transform(image)
        if np.max(map) > 1.0:
            map = map / 255.0
        assert np.min(map) >= 0.0 and np.max(map) <= 1.0, "Ground-truth not in [0,1].{} {}".format(np.min(map),
                                                                                                   np.max(map))

        return image, torch.FloatTensor(map)

    def __len__(self):
        return len(self.images)
