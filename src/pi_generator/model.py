import torch
import torch.nn as nn

from pi_generator.utils import reparameterize


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
