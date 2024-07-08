from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AugPiDataset(Dataset):
    def __init__(self, img_path: Path, iterations: int, deviation: float = 0.0):
        self.img = Image.open(img_path)
        self.iterations = iterations
        to_tenso_trans = transforms.ToTensor()
        self.img_tensor = to_tenso_trans(self.img)
        self.xs = torch.from_numpy(np.load("/mnt/e/learning/pi_xs.npy"))
        self.ys = torch.from_numpy(np.load("/mnt/e/learning/pi_ys.npy"))
        self.rgb_values = self.img_tensor[:, self.xs, self.ys]
        self.deviation = deviation

    def __len__(self) -> int:
        return self.iterations

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        aug_img = torch.zeros_like(self.img_tensor)
        delta_x = self.deviation * torch.randn_like(self.xs, dtype=torch.float)
        delta_y = self.deviation * torch.randn_like(self.ys, dtype=torch.float)
        new_xs = (self.xs + delta_x).to(torch.int)
        new_ys = (self.ys + delta_y).to(torch.int)
        aug_img[:, new_xs, new_ys] = self.rgb_values
        return aug_img, aug_img


class PiDataset(Dataset):
    def __init__(self, img_path: Path, iterations: int):
        self.img = Image.open(img_path)
        self.iterations = iterations
        to_tenso_trans = transforms.ToTensor()
        self.img_tensor = to_tenso_trans(self.img)

    def __len__(self) -> int:
        return self.iterations

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.img_tensor, self.img_tensor
