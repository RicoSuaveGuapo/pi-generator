from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import typer
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from pi_generator.dataset import PiDataset
from pi_generator.eval import compute_fid_command
from pi_generator.model import TransformerDecoder, TransformerVAE
from pi_generator.utils import vae_loss

THE_PI_IMG = Path("./sparse_pi_colored.jpg")

"""Example
python -m pi_generator.main --is-train  --is-generate
"""


def inference_from_decoder(
    decoder: TransformerDecoder, latent_dim: int, seq_length: int
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(1, latent_dim).to(device)

    with torch.no_grad():
        generate_image = decoder.decode_from_latent(z, seq_length)

    generate_image = torch.clamp(generate_image, min=0.0)
    generate_image = (generate_image - generate_image.min()) / (
        generate_image.max() - generate_image.min()
    )
    return generate_image.view(3, 300, 300).cpu().numpy().transpose(1, 2, 0)


def generate_pi_img(ckpt_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerVAE(
        input_dim=300 * 300 * 3, embed_dim=64, nhead=2, num_layers=3
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    generate_image = inference_from_decoder(
        model.decoder, latent_dim=64, seq_length=300 * 300 * 3
    )
    generate_image = (generate_image * 255.0).astype(np.uint8)
    img = Image.fromarray(generate_image)
    img_path = "generate_pi.jpg"
    img.save(img_path)
    print(f"\nImage is generated at {img_path}")
    compute_fid_command(THE_PI_IMG, Path(img_path))


def train(img_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerVAE(
        input_dim=300 * 300 * 3, embed_dim=64, nhead=2, num_layers=3
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    kl_beta = 0.0
    pbar = tqdm(DataLoader(PiDataset(img_path, iterations=3000)))
    for idx, (src, tgt) in enumerate(pbar):
        _src = src.view(src.size(0), -1).to(device)  # (1, 300 * 300 * 3)
        _tgt = tgt.view(tgt.size(0), -1).to(device)

        output, mu, logvar = model(_src, _tgt)
        kl_beta = min(1.0, idx / len(pbar))
        loss, loss_items = vae_loss(output, _tgt, mu, logvar, kl_beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 50 == 0:
            pbar.set_description(
                f"recon loss: {loss_items['recon']:.2f}, kl_loss: {loss_items['kl_loss']:.2g}"
            )

    Path("./ckpts").mkdir(exist_ok=True, parents=True)
    model_path = "./ckpts/a_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"model is saved in {model_path}")


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
