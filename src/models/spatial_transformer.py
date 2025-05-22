# src/models/spatial_transformer.py

import torch
import torch.nn as nn
from src.core.base_module import BaseModule


class SpatialTransformer(BaseModule):
    """
    Spatial transformer network that attends across the flattened
    spatial–temporal feature sequence and aligns similar patterns
    regardless of direction.
    """

    def build(self) -> None:
        cfg = self.config

        # Build a standard TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (B, seq_len, embed_dim)
        Returns:
            aligned_feats: Tensor of same shape (B, seq_len, embed_dim)
        """
        # Transformer expects (S, B, E)
        x = features.permute(1, 0, 2)           # (seq_len, B, embed_dim)
        x = self.transformer(x)                 # (seq_len, B, embed_dim)
        aligned_feats = x.permute(1, 0, 2)      # (B, seq_len, embed_dim)
        return aligned_feats
