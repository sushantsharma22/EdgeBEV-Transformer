"""
Load an exported TorchScript/ONNX model and run on simulated edge device.
"""
import numpy as np
import torch
from config.deployment_config import DeploymentConfig
from src.pipeline.inference_engine import InferenceEngine

# Load configs
deploy_cfg = DeploymentConfig.load_from_yaml("config/deployment_config.yaml")
# Path to TorchScript or ONNX
model_path = "exported_models/pipeline.pt"
engine = InferenceEngine(
    model_cfg=None,  # only deployment config used
    deploy_cfg=deploy_cfg,
    weights_path=None
)
# Replace pipeline with the loaded script
engine.pipeline = torch.jit.load(model_path)
engine.pipeline.eval()

# Simulate one batch
dummy = np.zeros((1, deploy_cfg.batch_size, 3, 64, 64), dtype=np.float32)
# Note: replace with actual BatchData for real use
# Run inference
out = engine.run_batch(dummy, with_timings=True)
print("Latency (ms):", out["timings_ms"])
