# src/models/bev_former.py

import torch
import torch.nn as nn
from src.core.base_module import BaseModule


class BEVFormer(BaseModule):
    """
    Bird’s-Eye View transformer that projects spatial–temporal features
    into a fixed BEV grid via cross-attention.
    """

    def build(self) -> None:
        cfg = self.config
        # BEV grid resolution
        H, W = cfg.bev_height, cfg.bev_width
        self.bev_size = H * W

        # Learnable BEV query embeddings (seq_len_bev, bev_embed_dim)
        self.bev_queries = nn.Parameter(
            torch.randn(self.bev_size, cfg.bev_embed_dim)
        )

        # Input projection if feature dim != bev_embed_dim
        # We'll infer feature_dim at build time placeholder; actual check in forward
        self.input_proj = nn.Identity()

        # TransformerDecoder for cross-attention: tgt=BEV queries, memory=features
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.bev_embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_layers)

        # Optional dropout on output
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (B, seq_len, E_in)
        Returns:
            bev_feats: Tensor of shape (B, bev_height*bev_width, bev_embed_dim)
        """
        B, seq_len, E_in = features.shape
        cfg = self.config

        # Project input features if needed
        if not isinstance(self.input_proj, nn.Identity):
            features = self.input_proj(features)
        elif E_in != cfg.bev_embed_dim:
            # replace identity with a real linear projection
            self.input_proj = nn.Linear(E_in, cfg.bev_embed_dim).to(features.device)
            features = self.input_proj(features)

        # Prepare memory and target
        # memory: (S, B, E) for transformer (seq_len, B, bev_embed_dim)
        mem = features.permute(1, 0, 2)

        # tgt/bev queries: (T, B, E)
        bev_q = self.bev_queries.unsqueeze(1).repeat(1, B, 1)

        # Cross-attention decoding
        out = self.decoder(tgt=bev_q, memory=mem)  # (bev_seq, B, E)

        # Permute back to (B, bev_seq, E)
        bev_feats = out.permute(1, 0, 2)
        bev_feats = self.dropout(bev_feats)

        return bev_feats
