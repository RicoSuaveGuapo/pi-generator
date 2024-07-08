import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import ConvexHull
from torch.nn import functional


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    mu and logvar are the outputs from model, we can sample
    z from the gaussian distribusion, however this ops is not
    differentiable. Therefore, VAE authors come out the idea that
    we can actually use another independent variable (eps) with mean = 0,
    and variance = 1, to construct z by scaling and translation transformation,
    like below. Then since this transformation is differentiable, then graident
    flow can go through.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.rand_like(std)
    return mu + eps * std


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    recon_loss = functional.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss *= kl_beta
    return recon_loss + kl_loss, {"recon": recon_loss, "kl_loss": kl_loss}


def draw_min_enclose_polygon():
    # Example point cloud
    xs = np.load("/mnt/e/learning/pi_xs.npy")
    ys = np.load("/mnt/e/learning/pi_ys.npy")
    points = np.stack((xs, ys), axis=-1)

    # Compute the convex hull
    hull = ConvexHull(points)

    # Plotting the points and the convex hull
    plt.plot(points[:, 0], points[:, 1], ".")
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], "k-")

    # Optionally, fill the convex hull
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], "c", alpha=0.3)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Convex Hull of 2D Point Cloud")
    plt.savefig("tmp.jpg")
