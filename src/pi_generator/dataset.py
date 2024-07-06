import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def input() -> np.ndarray:
    xs = np.load("/Users/rico.li/Job/gen_ml_quiz_content/pi_xs.npy")
    ys = np.load("/Users/rico.li/Job/gen_ml_quiz_content/pi_ys.npy")
    image_array = np.array(
        Image.open("/Users/rico.li/Job/gen_ml_quiz_content/sparse_pi_colored.jpg")
    )
    return image_array[xs, ys]


class PiDataset(Dataset):
    def __init__(self, img_path: str, versions: int):
        self.img = Image.open(img_path)
        self.versions = versions
        to_tenso_trans = transforms.ToTensor()
        self.img_tensor = to_tenso_trans(self.img)

    def __len__(self) -> int:
        return self.versions

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.img_tensor, self.img_tensor
