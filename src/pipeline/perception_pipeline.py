# src/pipeline/perception_pipeline.py

import time
from typing import Dict, Any, Optional

import torch
from torch import nn

from src.models.image_encoder import ImageEncoder
from src.models.spatial_transformer import SpatialTransformer
from src.models.bev_former import BEVFormer
from src.models.fusion_transformer import FusionTransformer
from src.models.trajectory_planner import TrajectoryPlanner
from src.core.metrics_logger import MetricsLogger
from src.core.data_structures import BatchData, Trajectory


class PerceptionPipeline(nn.Module):
    """
    Orchestrates the end-to-end perception stack:
     1) Encode multi-camera, multi-temporal images
     2) Spatial alignment via transformer
     3) BEV projection
     4) Multi-modal fusion
     5) Trajectory planning
    Also collects timing metrics and exposes intermediate outputs.
    """

    def __init__(self, config: Any, device: Optional[torch.device] = None):
        """
        Args:
            config: ModelConfig instance with sub-configs for each module.
            device: torch.device to place modules on (defaults to CPU).
        """
        super().__init__()
        self.config = config
        self.device = device or torch.device(config.device)
        # Instantiate modules
        self.image_encoder       = ImageEncoder(config.image_encoder).to(self.device)
        self.spatial_transformer = SpatialTransformer(config.spatial_transformer).to(self.device)
        self.bev_former          = BEVFormer(config.bev_former).to(self.device)
        self.fusion_transformer  = FusionTransformer(config.fusion_transformer).to(self.device)
        self.trajectory_planner  = TrajectoryPlanner(config.trajectory_planner).to(self.device)

        # Metrics logger
        self.metrics = MetricsLogger(output_dir=config.output_dir, filename="pipeline_metrics.csv")

        # Set eval mode by default; training loops can switch back to train()
        self.eval()

    def forward(self, batch: BatchData) -> Dict[str, Any]:
        """
        Run a forward pass through the entire pipeline.

        Args:
            batch: BatchData containing images, sensors, map_data.
        Returns:
            Dict with keys:
              - trajectories: Tensor (B, T, 4)
              - intermediates: Dict of raw module outputs
        """
        timings: Dict[str, float] = {}
        x = batch.images  # list of lists of np.ndarray

        # Convert numpy images to Torch and reshape
        # Expected shape: (B, cameras, T, C, H, W)
        imgs = torch.from_numpy(
            np.stack(x).reshape(
                len(batch.images),
                self.config.image_encoder.camera_count,
                self.config.image_encoder.temporal_window,
                self.config.image_encoder.input_channels,
                *self.config.image_encoder.image_size[::-1]
            )
        ).float().to(self.device)

        # 1) Image encoding
        t0 = time.time()
        feats = self.image_encoder(imgs)
        timings["encode_ms"] = (time.time() - t0) * 1e3

        # 2) Spatial transformer
        t0 = time.time()
        aligned = self.spatial_transformer(feats)
        timings["spatial_ms"] = (time.time() - t0) * 1e3

        # 3) BEV projection
        t0 = time.time()
        bev = self.bev_former(aligned)
        timings["bev_ms"] = (time.time() - t0) * 1e3

        # 4) Fusion of vision + sensors + map
        t0 = time.time()
        sensor_dict = {
            "imu":    torch.tensor([s.imu for s in batch.sensors],    device=self.device),
            "speed":  torch.tensor([s.speed for s in batch.sensors],  device=self.device).unsqueeze(-1),
            "steering": torch.tensor(
                [s.steering_angle for s in batch.sensors], device=self.device
            ).unsqueeze(-1),
            "gnss":   torch.tensor([list(s.gnss.values()) for s in batch.sensors], device=self.device),
        }
        # map features optional: could be pre-embedded
        fused = self.fusion_transformer(bev, sensor_dict)
        timings["fusion_ms"] = (time.time() - t0) * 1e3

        # 5) Trajectory planning
        t0 = time.time()
        traj = self.trajectory_planner(fused)
        timings["plan_ms"] = (time.time() - t0) * 1e3

        # Log metrics
        # step is None here; user can supply externally if needed
        self.metrics.log(
            step=-1,
            phase="inference",
            metrics={
                "encode_ms":   timings["encode_ms"],
                "spatial_ms":  timings["spatial_ms"],
                "bev_ms":      timings["bev_ms"],
                "fusion_ms":   timings["fusion_ms"],
                "plan_ms":     timings["plan_ms"],
            },
            batch_size=len(batch.images),
        )

        return {
            "trajectories": traj,
            "intermediates": {
                "features": feats,
                "aligned": aligned,
                "bev": bev,
                "fused": fused,
            },
            "timings_ms": timings,
        }

    def save_metrics(self, path: Optional[str] = None) -> None:
        """
        Flush accumulated metrics to CSV.
        """
        self.metrics.save_csv(path or None)
