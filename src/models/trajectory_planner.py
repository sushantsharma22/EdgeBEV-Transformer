# src/models/trajectory_planner.py

import torch
import torch.nn as nn
from src.core.base_module import BaseModule


class TrajectoryPlanner(BaseModule):
    """
    Path-planning transformer that generates a sequence of future
    trajectory points (x, y), heading, and velocity over a fixed horizon.
    """

    def build(self) -> None:
        cfg = self.config
        self.horizon = cfg.horizon_steps
        embed_dim = cfg.embed_dim

        # Learnable trajectory query embeddings (horizon, embed_dim)
        self.query_embeds = nn.Parameter(
            torch.randn(self.horizon, embed_dim)
        )

        # If fused feature dim != embed_dim, project
        self.input_proj = nn.Identity()

        # Transformer decoder: queries attend over fused feature memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=cfg.num_layers)

        # Prediction heads
        # (x, y) offsets in vehicle frame
        self.traj_head     = nn.Linear(embed_dim, 2)
        # heading (rad)
        self.heading_head  = nn.Linear(embed_dim, 1)
        # velocity (m/s)
        self.velocity_head = nn.Linear(embed_dim, 1)

    def forward(self, fused_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_feats: (B, seq_len, embed_dim) — output of FusionTransformer
        Returns:
            traj: Tensor of shape (B, horizon, 4), where each point is
                  [x_offset, y_offset, heading, velocity]
        """
        B, seq_len, E_in = fused_feats.shape
        cfg = self.config

        # Project features if needed
        if not isinstance(self.input_proj, nn.Identity):
            feats = self.input_proj(fused_feats)
        elif E_in != cfg.embed_dim:
            self.input_proj = nn.Linear(E_in, cfg.embed_dim).to(fused_feats.device)
            feats = self.input_proj(fused_feats)
        else:
            feats = fused_feats

        # Prepare memory for decoder: (S, B, E)
        memory = feats.permute(1, 0, 2)

        # Prepare trajectory queries: (T, B, E)
        queries = self.query_embeds.unsqueeze(1).repeat(1, B, 1)

        # Decode: cross-attend queries to memory
        decoded = self.decoder(tgt=queries, memory=memory)  # (T, B, E)

        # Permute back: (B, T, E)
        decoded = decoded.permute(1, 0, 2)

        # Apply heads
        xy       = self.traj_head(decoded)     # (B, T, 2)
        heading  = self.heading_head(decoded)  # (B, T, 1)
        velocity = self.velocity_head(decoded) # (B, T, 1)

        # Concatenate into final trajectory tensor
        traj = torch.cat([xy, heading, velocity], dim=-1)  # (B, T, 4)
        return traj
