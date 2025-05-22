# tests/test_models/test_image_encoder.py

import torch
import pytest
from config.model_config import ModelConfig
from src.models.image_encoder import ImageEncoder

def test_image_encoder_output_shape():
    # Arrange
    cfg = ModelConfig().image_encoder
    encoder = ImageEncoder(cfg)
    # Create dummy input: (B, cameras, temporal_window, C, H, W)
    B = 2
    cams = cfg.camera_count
    T = cfg.temporal_window
    C = cfg.input_channels
    # Use spatial dims divisible by 4 (backbone downsamples by 4×)
    H, W = 64, 64
    dummy = torch.randn(B, cams, T, C, H, W)
    # Act
    out = encoder(dummy)
    # Compute expected sequence length and feature dim
    Hf, Wf = H // 4, W // 4
    expected_seq_len = cams * T * Hf * Wf
    # Assert
    assert out.shape == (B, expected_seq_len, cfg.out_channels)
