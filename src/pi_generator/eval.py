from pathlib import Path

import numpy as np
import torch
from pytorch_fid.fid_score import (
    calculate_activation_statistics,
    calculate_frechet_distance,
)
from pytorch_fid.inception import InceptionV3

"""
adjust pytorch_fid's compute_statistics_of_path to our case
"""


def compute_statistics_of_path(
    path: Path,
    model: InceptionV3,
    batch_size: int,
    dims: int,
    device: str,
    num_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    m, s = calculate_activation_statistics(
        [Path(path)], model, batch_size, dims, device, num_workers
    )

    return m, s


def calculate_fid_given_paths(
    paths: list[Path],
    batch_size: int,
    device: str,
    dims: int,
    num_workers: int = 1,
) -> float:
    """Calculates the FID of two paths"""
    for p in paths:
        if not p.exists():
            raise RuntimeError("Invalid path: %s" % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(
        paths[0], model, batch_size, dims, device, num_workers
    )
    m2, s2 = compute_statistics_of_path(
        paths[1], model, batch_size, dims, device, num_workers
    )
    return calculate_frechet_distance(m1, s1, m2, s2)


def compute_fid(real_image: Path, generated_image: Path) -> float:
    return calculate_fid_given_paths(
        [real_image, generated_image],
        batch_size=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dims=2048,
    )


def compute_fid_command(real_images_path: Path, generated_images_path: Path):
    fid = compute_fid(real_images_path, generated_images_path)
    print(f"FID Score: {fid}")
