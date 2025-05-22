# src/models/fusion_transformer.py

import torch
import torch.nn as nn
from typing import Optional, Dict
from src.core.base_module import BaseModule


class FusionTransformer(BaseModule):
    """
    Multi‐modal fusion transformer: merges BEV features (vision)
    with sensor readings and optional map embeddings into a single
    unified representation.
    """

    def build(self) -> None:
        cfg = self.config
        embed_dim = cfg.embed_dim

        # Projection layers for non‐vision modalities
        # Assumes sensor_data dict will have keys matching cfg.modalities
        # with tensor shapes as documented below.
        self.modality_projs = nn.ModuleDict()
        for mod in cfg.modalities:
            if mod == "vision":
                continue
            # define expected input dims for each modality
            # you may adjust these to match your preprocessor output
            input_dims = {
                "imu": 6,         # e.g., acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
                "gnss": 3,        # lat, lon, alt
                "speed": 1,       # scalar
                "steering": 1,    # scalar
            }
            in_dim = input_dims.get(mod, 1)
            self.modality_projs[mod] = nn.Linear(in_dim, embed_dim)

        # Build the core TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers
        )

    def forward(
        self,
        bev_feats: torch.Tensor,
        sensor_data: Dict[str, torch.Tensor],
        map_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            bev_feats: (B, seq_bev, embed_dim) — output of BEVFormer
            sensor_data: dict of modality→Tensor, where each Tensor is
                         (B, input_dim) matching the projections defined in build()
            map_feats: optional (B, seq_map, embed_dim) — e.g., learned
                       embeddings of OpenStreetMap waypoints

        Returns:
            fused_feats: (B, seq_total, embed_dim)
        """
        B, _, E = bev_feats.shape
        tokens = [bev_feats]

        # Embed each sensor modality as a single token
        for mod, proj in self.modality_projs.items():
            if mod in sensor_data:
                x = sensor_data[mod]              # (B, in_dim)
                tok = proj(x).unsqueeze(1)        # (B, 1, embed_dim)
                tokens.append(tok)

        # Append map features if provided
        if map_feats is not None:
            tokens.append(map_feats)             # (B, seq_map, embed_dim)

        # Concatenate into one long sequence of tokens
        seq = torch.cat(tokens, dim=1)           # (B, seq_total, embed_dim)

        # Transformer expects shape (S, B, E)
        seq = seq.permute(1, 0, 2)
        fused = self.transformer(seq)            # (S, B, E)
        fused = fused.permute(1, 0, 2)           # (B, seq_total, embed_dim)

        return fused
