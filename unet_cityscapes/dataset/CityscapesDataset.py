import torch
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import numpy as np
import torch.nn.functional as F


class CityscapesDataset(Dataset):
    def __init__(self, data_root_path: str, subset: str):
        self.images_paths = glob.glob(os.path.join(data_root_path, "img", subset, "*.png"))
        self.masks_paths = glob.glob(os.path.join(data_root_path, "mask", subset, "*.png"))
        sorted(self.images_paths)
        sorted(self.masks_paths)
        assert (len(self.images_paths) == len(self.masks_paths))

    def __normalize_img(self, img):
        return img / 255.0

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        mask_path = self.masks_paths[idx]
        img = np.array(Image.open(img_path))
        img = torch.as_tensor(img, dtype=torch.float32)
        mask = np.array(Image.open(mask_path))
        mask = torch.as_tensor(mask, dtype=torch.int64)
        one_hot_mask = F.one_hot(mask, num_classes=34)
        return self.__normalize_img(img), one_hot_mask
