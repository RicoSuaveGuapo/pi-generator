import torch
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
