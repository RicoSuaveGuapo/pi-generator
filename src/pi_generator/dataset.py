from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
