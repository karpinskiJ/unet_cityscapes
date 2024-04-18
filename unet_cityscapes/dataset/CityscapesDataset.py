import torch
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np


class CityscapesDataset(Dataset):
    def __init__(self, data_root_path: str, subset: str, samples_cnt, num_classes, shape,resize=False):
        self.images_paths = sorted(glob.glob(os.path.join(data_root_path, "img", subset, "*.png")))[:samples_cnt]
        self.masks_paths = sorted(glob.glob(os.path.join(data_root_path, "mask", subset, "*.png")))[:samples_cnt]
        self.shape = shape
        self.num_classes = num_classes
        self.resize = resize

        assert (len(self.images_paths) == len(self.masks_paths))

    def __len__(self):
        return len(self.images_paths)

    def transform_img(self, input):
        transformations = [
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32)
            , T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if self.resize:
            transformations.append(T.Resize(size=self.shape))

        return T.Compose(transformations)(input)

    def transform_mask(self, input):
        transformations = [
            T.PILToTensor()
        ]
        if self.resize:
            transformations.append(T.Resize(size=self.shape))
        return T.Compose(transformations)(input)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        mask_path = self.masks_paths[idx]

        img_tensor = self.transform_img(Image.open(img_path))

        mask_tensor = self.transform_mask(Image.open(mask_path))[0]
        return img_tensor, mask_tensor.long()
