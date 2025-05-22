# config/model_config.py

from dataclasses import dataclass, field
from typing import Dict, List
from config.base_config import BaseConfig


@dataclass
class ImageEncoderConfig:
    # Multi-camera & temporal settings
    camera_count: int = 6
    temporal_window: int = 3  # number of frames per camera
    input_channels: int = 3

    # Convolutional backbone
    backbone: str = "resnet18"  # or "efficientnet_b0", etc.
    pretrained: bool = False
    out_channels: int = 256  # feature dimension per frame

    # Positional encoding
    use_positional_encoding: bool = True
    pos_encoding_dim: int = 64


@dataclass
class SpatialTransformerConfig:
    # Transformer settings for spatial alignment
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1

    # Pattern-matching window
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1


@dataclass
class BEVFormerConfig:
    # Bird’s Eye View generation
    bev_height: int = 64
    bev_width: int = 64
    bev_embed_dim: int = 256

    # Transformer specifics
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 512
    dropout: float = 0.1

    # Fusion of multi-view features
    attention_type: str = "cross"  # or "self"


@dataclass
class FusionTransformerConfig:
    # Combine vision + sensor modalities
    modalities: List[str] = field(default_factory=lambda: ["vision", "imu", "speed", "gnss", "steering"])
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 3
    ffn_dim: int = 512
    dropout: float = 0.1


@dataclass
class TrajectoryPlannerConfig:
    # Output trajectory length & resolution
    horizon_steps: int = 30
    step_dt: float = 0.1  # seconds per step

    # Planning transformer
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1

    # Loss weighting (for multi-task objectives)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "trajectory": 1.0,
        "heading": 0.5,
        "velocity": 0.5,
    })


@dataclass
class ModelConfig(BaseConfig):
    """
    Full model configuration, including general settings
    inherited from BaseConfig and all per-module configs.
    """
    # Time-aggregation parameter: k frames back per camera
    temporal_stride: int = 1

    # Sub-configurations
    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    spatial_transformer: SpatialTransformerConfig = field(default_factory=SpatialTransformerConfig)
    bev_former: BEVFormerConfig = field(default_factory=BEVFormerConfig)
    fusion_transformer: FusionTransformerConfig = field(default_factory=FusionTransformerConfig)
    trajectory_planner: TrajectoryPlannerConfig = field(default_factory=TrajectoryPlannerConfig)

    # Learning parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    total_steps: int = 100_000

    # Scheduler
    lr_scheduler: str = "linear"  # or "cosine", "step"
    scheduler_params: Dict[str, float] = field(default_factory=lambda: {
        "step_size": 30_000,
        "gamma": 0.1,
    })

    # Override BaseConfig name
    config_name: str = "model_config"

    @classmethod
    def load_from_yaml(cls, path: str) -> "ModelConfig":
        """
        Extends BaseConfig loader: also handles nested sub-configs.
        """
        import yaml
        from dataclasses import fields

        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}

        # Helper: recursively map dicts to dataclasses
        def _populate(dc_cls, values):
            init_kwargs = {}
            for f in fields(dc_cls):
                name = f.name
                if name in values:
                    val = values[name]
                    # If the field type is itself a dataclass, recurse
                    if hasattr(f.type, "__dataclass_fields__"):
                        init_kwargs[name] = _populate(f.type, val or {})
                    else:
                        init_kwargs[name] = val
            return dc_cls(**init_kwargs)

        return _populate(cls, cfg_dict)
