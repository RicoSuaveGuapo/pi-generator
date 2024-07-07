from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import typer
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from pi_generator.dataset import PiDataset
from pi_generator.model import TransformerDecoder, TransformerVAE
from pi_generator.utils import vae_loss

THE_PI_IMG = Path("/mnt/e/learning/sparse_pi_colored.jpg")


def inferce_from_decoder(
    decoder: TransformerDecoder, latent_dim: int, seq_length: int
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(1, latent_dim).to(device)

    with torch.no_grad():
        generate_image = decoder.decode_from_latent(z, seq_length)

    generate_image = (generate_image - generate_image.min()) / (
        generate_image.max() - generate_image.min()
    )
    return generate_image.view(3, 300, 300).cpu().numpy().transpose(1, 2, 0)


def generate_pi_img(ckpt_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerVAE(
        input_dim=300 * 300 * 3, embed_dim=64, nhead=8, num_layers=3
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path))
    generate_image = inferce_from_decoder(
        model.decoder, latent_dim=64, seq_length=300 * 300 * 3
    )
    generate_image = (generate_image * 255.0).astype(np.uint8)
    img = Image.fromarray(generate_image)
    img_path = "generate_pi.jpg"
    img.save(img_path)
    print(f"\nImage is generated at {img_path}")


def train(img_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerVAE(
        input_dim=300 * 300 * 3, embed_dim=64, nhead=2, num_layers=3
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for src, tgt in tqdm(
        DataLoader(PiDataset(img_path, iterations=1000)), desc="Training"
    ):
        _src = src.view(src.size(0), -1).to(device)  # (1, 300 * 300 * 3)
        _tgt = tgt.view(tgt.size(0), -1).to(device)

        output, mu, logvar = model(_src, _tgt)
        loss = vae_loss(output, _tgt, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    Path("./ckpts").mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), "./ckpts/a_model.pth")


def main(
    ckpt_path: Path = typer.Option("./ckpts/a_model.pth", "--ckpt-path"),
    *,
    is_train: bool = False,
    is_generate: bool = False,
):
    if is_train:
        train(THE_PI_IMG)
    if is_generate:
        generate_pi_img(ckpt_path)


if __name__ == "__main__":
    typer.run(main)
