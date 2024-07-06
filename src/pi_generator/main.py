import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def input() -> np.ndarray:
    xs = np.load("/Users/rico.li/Job/gen_ml_quiz_content/pi_xs.npy")
    ys = np.load("/Users/rico.li/Job/gen_ml_quiz_content/pi_ys.npy")
    image_array = np.array(
        Image.open("/Users/rico.li/Job/gen_ml_quiz_content/sparse_pi_colored.jpg")
    )
    return image_array[xs, ys]


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, nhead: int, num_layers: int):
        super().__init__()
        # linear trans from raw input to embedding
        # y = Mx + b, where input_dim depends on the src, and embed_dim decides by me
        self.embedding = nn.Linear(input_dim, embed_dim)
        # notice that the output dim of above linear trans should be able to take
        # as input for transformer, that us d_model = embed_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=128,  # just for small encoder
        )
        # repeatly construct encoder_layer num_layers times
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embedding(src)
        return self.transformer_encoder(src)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        nhead: int,
        num_layers: int,
        output_dmi: int,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=128,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        # use linear trans back to desired dim
        self.fc_out = nn.Linear(embed_dim, output_dmi)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        embed_tgt = self.embedding(tgt)
        decode_output = self.transformer_decoder(embed_tgt, memory)
        return self.fc_out(decode_output)
