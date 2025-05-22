#!/usr/bin/env python3
# scripts/deploy.py

import os
import argparse
import torch
import json
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from config.deployment_config import DeploymentConfig
from src.pipeline.inference_engine import InferenceEngine
from src.data.data_loader import get_dataloader
from src.core.data_structures import BatchData


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and export model for edge deployment")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML model + deployment config")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained checkpoint (.pt or .pth)")
    parser.add_argument("--manifest", type=str, required=False,
                        help="Optional manifest JSON to sample a batch for ONNX export")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root dir for manifest images/sensors (for sampling)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to use when sampling a representative batch")
    parser.add_argument("--export-dir", type=str, default=None,
                        help="Directory to write exported artifacts (overrides config)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configs
    model_cfg = ModelConfig.load_from_yaml(args.config)
    deploy_cfg = DeploymentConfig.load_from_yaml(args.config)
    export_dir = args.export_dir or deploy_cfg.export_dir
    os.makedirs(export_dir, exist_ok=True)

    # Build inference engine
    engine = InferenceEngine(model_cfg, deploy_cfg, weights_path=args.weights)

    sample_batch = None
    # If exporting ONNX, need a representative batch
    if deploy_cfg.compile_backend == "onnx":
        if not args.manifest:
            raise ValueError("--manifest is required for ONNX export to sample a batch")
        # build a single-batch DataLoader
        dl: DataLoader = get_dataloader(
            data_root=args.data_root,
            manifest_path=args.manifest,
            batch_size=args.batch_size,
            temporal_window=model_cfg.image_encoder.temporal_window,
            stride=model_cfg.temporal_stride,
            num_workers=0,
            shuffle=False
        )
        # take first batch
        try:
            sample_batch = next(iter(dl))
        except StopIteration:
            raise RuntimeError("Manifest is empty, cannot sample batch for ONNX export")

    # Export model
    exported_path = engine.export(export_dir=export_dir, sample_batch=sample_batch)
    print(f"Exported model to: {exported_path}")

    # Optionally package everything
    archive_path = os.path.join(export_dir, "deployment_package.tar.gz")
    os.system(f"tar -czf {archive_path} -C {export_dir} .")
    print(f"Packaged deployment artifacts at: {archive_path}")


if __name__ == "__main__":
    main()
