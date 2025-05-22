# config/deployment_config.py

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from config.base_config import BaseConfig


@dataclass
class DeploymentConfig(BaseConfig):
    """
    Deployment configuration for edge device optimization.
    Inherits common parameters (e.g. device, batch_size) from BaseConfig.
    """

    # Target hardware
    target_device: str = "cpu"  # options: "cpu", "cuda", "tpu", "npu", etc.
    gpu_memory_fraction: Optional[float] = 0.9  # for CUDA contexts

    # Optimization goals
    optimize_for: str = "latency"  # options: "latency", "throughput", "size"
    memory_limit_mb: Optional[int] = None     # cap total RAM usage on device

    # Precision / quantization
    use_fp16: bool = False                   # half-precision inference
    quantization: bool = False               # enable post-training quantization
    quantization_mode: str = "dynamic"       # "dynamic" or "static"

    # Pruning
    prune: bool = False
    prune_ratio: float = 0.2                  # fraction of weights to prune

    # Model export settings
    compile_backend: str = "onnx"             # "onnx", "tflite", "torchscript"
    onnx_opset: int = 12
    tflite_optimizations: List[str] = field(
        default_factory=lambda: ["DEFAULT"]
    )
    torchscript_jit: bool = True

    # Output
    export_dir: str = "./exported_models"

    # Runtime threads
    num_inference_threads: int = 1

    # Extra flags passed to the compiler/runtime
    additional_compile_args: Dict[str, str] = field(default_factory=dict)

    # Override config name
    config_name: str = "deployment_config"

    def validate(self) -> None:
        """
        Sanity‐check parameter combinations.
        """
        if self.quantization and self.use_fp16:
            raise ValueError("Cannot enable both FP16 and quantization at once.")
        if self.quantization_mode not in ("dynamic", "static"):
            raise ValueError(
                f"quantization_mode must be 'dynamic' or 'static', got '{self.quantization_mode}'"
            )
        if self.prune:
            if not (0.0 < self.prune_ratio < 1.0):
                raise ValueError("prune_ratio must be between 0.0 and 1.0.")

    @classmethod
    def load_from_yaml(cls, path: str) -> "DeploymentConfig":
        """
        Load and validate a YAML‐based deployment config.
        """
        cfg = super().load_from_yaml(path)
        cfg.validate()
        return cfg
