# src/models/image_encoder.py

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.base_module import BaseModule


class ImageEncoder(BaseModule):
    """
    Multi‐camera, multi‐temporal image encoder.
    Takes a tensor of shape (B, cameras, T, C, H, W)
    and returns a sequence of features of shape (B, seq_len, F),
    where seq_len = cameras * T * H' * W' and F = out_channels.
    """

    def build(self) -> None:
        cfg = self.config
        in_ch = cfg.input_channels
        out_ch = cfg.out_channels

        # Simple conv‐backbone: reduce H, W by 4× each
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # Positional encoding buffer (initialized lazily)
        self.register_buffer("pos_enc", torch.empty(0), persistent=False)
        self.use_pe = cfg.use_positional_encoding
        self.pe_dim = cfg.out_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor shaped (B, cameras, T, C, H, W)
        Returns:
            features: Tensor shaped (B, seq_len, F)
        """
        B, cams, T, C, H, W = images.shape
        x = images.view(B * cams * T, C, H, W)       # (B*cams*T, C, H, W)
        feat = self.backbone(x)                      # (B*cams*T, F, H', W')
        _, F, Hf, Wf = feat.shape
        feat = feat.view(B, cams * T, F, Hf * Wf)    # (B, cams*T, F, H'*W')
        feat = feat.permute(0, 1, 3, 2)              # (B, cams*T, H'*W', F)
        seq = feat.contiguous().view(B, cams * T * Hf * Wf, F)

        if self.use_pe:
            seq = seq + self._get_positional_encoding(seq)

        return seq

    def _get_positional_encoding(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Lazily build or retrieve a sinusoidal positional encoding
        for this sequence length and feature dimension.
        Returns a tensor of shape (1, seq_len, F).
        """
        B, seq_len, F = seq.shape

        # If we've already built a buffer of the correct size, reuse it
        if self.pos_enc.numel() == seq_len * F:
            return self.pos_enc.unsqueeze(0)

        # Otherwise, compute a new sinusoidal encoding
        pe = torch.zeros(seq_len, F, device=seq.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=seq.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, F, 2, dtype=torch.float, device=seq.device)
            * (-math.log(10000.0) / F)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store for reuse
        self.register_buffer("pos_enc", pe, persistent=False)
        return pe.unsqueeze(0)
