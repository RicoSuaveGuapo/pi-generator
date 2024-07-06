from pathlib import Path

import torch
import torch.optim as optim
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from pi_generator.dataset import PiDataset
from pi_generator.model import TransformerVAE
from pi_generator.utils import vae_loss

THE_PI_IMG = "/Users/rico.li/Job/gen_ml_quiz_content/sparse_pi_colored.jpg"


def generate(): ...


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerVAE(
        input_dim=300 * 300 * 3, embed_dim=64, nhead=2, num_layers=3
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1
    for _ in tqdm(range(num_epochs)):
        model.train()
        for src, tgt in DataLoader(PiDataset(THE_PI_IMG, versions=100)):
            _src = src.view(src.size(0), -1).to(device)
            _tgt = tgt.view(tgt.size(0), -1).to(device)

            output, mu, logvar = model(_src, _tgt)
            loss = vae_loss(output, tgt, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    Path("./ckpts").mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), "./ckpts/a_model.pth")


if __name__ == "__main__":
    typer.run(train)
