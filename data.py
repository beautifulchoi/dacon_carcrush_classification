import cv2
import torch

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from config import CFG

import warnings
warnings.filterwarnings(action='ignore')

# https://dacon.io/en/competitions/official/236064/codeshare/7572?page=1&dtype=recent#


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.label_list = label_list
        self.transforms = transforms
        self.img_path_list = img_path_list

    def __getitem__(self, index):
        images = self.get_frames(self.img_path_list[index])

        if self.transforms is not None:
            res = self.transforms(**images)
            # print(images)
            images = torch.zeros((len(images), 3, CFG.img_size, CFG.img_size))
            # print(images.shape)
            images[0, :, :, :] = torch.Tensor(res["image"])
            for i in range(1, len(images)):
                images[i, :, :, :] = res[f"image{i}"]

        images = images.permute(1, 0, 2, 3)

        if self.label_list is not None:
            label = self.label_list[index]

            return images, label
        else:
            return images

    def __len__(self):
        return len(self.img_path_list)

    def get_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        imgs = []
        for fidx in range(frames):
            _, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        ret = {f"image{i}": imgs[i] for i in range(1, len(imgs))}
        ret['image'] = imgs[0]

        return ret


class Transforms:
    weather = A.Compose([
        # A.CenterCrop(480,854,p=1.0),
        A.Resize(height=CFG.img_size, width=CFG.img_size),
        # A.Superpixels(p=0.3),
        # A.HorizontalFlip(0.5),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ], p=1, additional_targets={f"image{i}": "image" for i in range(1, 50)})

    other = A.Compose([
        # A.CenterCrop(480,854,p=1.0),
        A.Resize(height=CFG.img_size, width=CFG.img_size),
        # A.HorizontalFlip(0.5),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ], p=1, additional_targets={f"image{i}": "image" for i in range(1, 50)})

    test = A.Compose([
        # A.CenterCrop(480,854,p=1.0),
        A.Resize(height=CFG.img_size, width=CFG.img_size),
        # A.HorizontalFlip(0.5),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ], p=1, additional_targets={f"image{i}": "image" for i in range(1, 50)})
    rain = A.Compose([
        A.Resize(height=CFG.img_size, width=CFG.img_size),
        A.RandomRain(brightness_coefficient=0.9,
                     drop_width=1, blur_value=3, p=1),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ], p=1, additional_targets={f"image{i}": "image" for i in range(1, 50)})

    snow = A.Compose([
        A.Resize(height=CFG.img_size, width=CFG.img_size),
        A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3,
                     snow_point_upper=0.5, p=1),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ], p=1, additional_targets={f"image{i}": "image" for i in range(1, 50)})

    darken = A.Compose([
        A.Resize(height=CFG.img_size, width=CFG.img_size),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0), contrast_limit=(-0.7, -0.6), p=1),
        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1),
        # A.ToGray(0.5),
        # A.RandomSunFlare(flare_roi=(0, 0.5, 1, 1),angle_lower=0.5, src_radius=CFG.img_size//2, p=0.3),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ], p=1, additional_targets={f"image{i}": "image" for i in range(1, 50)})
