"""
Basic end-to-end inference example.
"""
import numpy as np
from config.model_config import ModelConfig
from src.pipeline.perception_pipeline import PerceptionPipeline
from src.data.data_loader import get_dataloader

# 1. Load config & model
cfg = ModelConfig.load_from_yaml("config/model_config.yaml")
pipeline = PerceptionPipeline(cfg).eval()

# 2. Prepare one batch
dl = get_dataloader("data", "manifest.json", batch_size=1,
                    temporal_window=cfg.image_encoder.temporal_window,
                    stride=cfg.temporal_stride,
                    num_workers=0,
                    shuffle=False)
batch = next(iter(dl))

# 3. Run inference
out = pipeline(batch)
traj = out["trajectories"].cpu().numpy()  # (1, T, 4)
print("Predicted trajectory:", traj)
