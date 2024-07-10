import math

import torch
import torch.nn as nn

from pi_generator.utils import reparameterize


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int = 1, max_len: int = 5000):
        super().__init__()
        self.embed_dim = embed_dim
        if embed_dim > 1:
            pe = torch.zeros(max_len, embed_dim)  # (max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
                1
            )  # (max_len, 1)
            div_term = torch.exp(
                torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
            ).unsqueeze(0)  # (1, embed_dim/2)
            pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, embeddim / 2)
            pe[:, 1::2] = torch.cos(position * div_term)
        elif embed_dim == 1:
            pe = torch.zeros(max_len)  # (max_len)
            position = torch.arange(0, max_len / 2, dtype=torch.float)  # (max_len/2,)
            div_term = 1 / 10000.0
            pe[0::2] = torch.sin(position * div_term)
            pe[1::2] = torch.cos(position * div_term)
        else:
            raise ValueError

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)  # not learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embed_dim > 1:
            return x + self.pe[:, : x.size(1), :]

        return x + self.pe[:, : x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int = 1, max_len: int = 5000):
        super().__init__()
        self.embed_dim = embed_dim
        if embed_dim > 1:
            self.pe = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        elif embed_dim == 1:
            self.pe = nn.Parameter(torch.zeros(1, max_len))
        else:
            raise ValueError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embed_dim > 1:
            return x + self.pe[:, : x.size(1), :]
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, nhead: int, num_layers: int):
        super().__init__()
        # linear trans from raw input to embedding
        # y = Mx + b, where input_dim depends on the src, and embed_dim decides by me
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = LearnablePositionalEncoding()
        # notice that the output dim of above linear trans should be able to take
        # as input for transformer, that us d_model = embed_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=128,  # just for small encoder
            batch_first=True,
        )
        # repeatly construct encoder_layer num_layers times
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # VAE components
        self.mu = nn.Linear(embed_dim, embed_dim)
        self.logvar = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, src: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = self.embedding(src)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        mu = self.mu(output)
        logvar = self.logvar(output)
        return output, mu, logvar


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
        self.positional_encoding = LearnablePositionalEncoding()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        # use linear trans back to desired dim
        self.fc_out = nn.Linear(embed_dim, output_dmi)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # tgt is our desired output, memory is the output from encoder
        embed_tgt = self.embedding(tgt)
        embed_tgt = self.positional_encoding(embed_tgt)
        decode_output = self.transformer_decoder(embed_tgt, memory)
        return self.fc_out(decode_output)

    def decode_from_latent(self, z: torch.Tensor, seq_length: int) -> torch.Tensor:
        tgt = torch.zeros(1, seq_length).to(z.device)
        return self.forward(tgt, z)


class TransformerVAE(nn.Module):
    def __init__(
        self, input_dim: int, embed_dim: int, nhead: int, num_layers: int
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, embed_dim, nhead, num_layers)
        self.decoder = TransformerDecoder(
            input_dim, embed_dim, nhead, num_layers, output_dmi=input_dim
        )

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory, mu, logvar = self.encoder(src)
        z = reparameterize(mu, logvar)
        output = self.decoder(tgt, memory + z)
        return output, mu, logvar
