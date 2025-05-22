#!/usr/bin/env python3
# scripts/evaluate.py

import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.model_config import ModelConfig
from config.deployment_config import DeploymentConfig
from src.data.data_loader import get_dataloader
from src.pipeline.perception_pipeline import PerceptionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Transformer Perception Pipeline")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML model config (model_config.yaml)")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model checkpoint (.pt or .pth)")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to dataset manifest JSON")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root directory for image and sensor data")
    parser.add_argument("--output-dir", type=str, default="./eval_outputs",
                        help="Where to write metrics & predictions")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override num_workers from config")
    parser.add_argument("--temporal-window", type=int, default=None,
                        help="Override temporal_window from config")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit number of batches for quick evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model config
    model_cfg = ModelConfig.load_from_yaml(args.config)
    if args.batch_size is not None:
        model_cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        model_cfg.num_workers = args.num_workers
    if args.temporal_window is not None:
        model_cfg.image_encoder.temporal_window = args.temporal_window

    # Deployment config (for device choice)
    deploy_cfg = DeploymentConfig.load_from_yaml(args.config)

    # Build DataLoader
    dataloader: DataLoader = get_dataloader(
        data_root=args.data_root,
        manifest_path=args.manifest,
        batch_size=model_cfg.batch_size,
        temporal_window=model_cfg.image_encoder.temporal_window,
        stride=model_cfg.temporal_stride,
        num_workers=model_cfg.num_workers,
        shuffle=False,
    )

    # Build pipeline and load weights
    device = torch.device(deploy_cfg.device)
    pipeline = PerceptionPipeline(model_cfg, device=device).eval()
    # load checkpoint
    ckpt = torch.load(args.weights, map_location=device)
    # support both full state_dict or dict with 'model_state'
    state_dict = ckpt.get("model_state", ckpt)
    pipeline.load_state_dict(state_dict)

    # Run inference
    all_trajectories = []
    pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
    for i, batch in enumerate(pbar):
        if args.max_batches is not None and i >= args.max_batches:
            break
        out = pipeline(batch)
        traj = out["trajectories"].cpu().numpy()  # (B, T, 4)
        all_trajectories.append(traj)
        # Optionally log progress of metrics (timings are logged internally)
        avg_time = pipeline.metrics.get_latest().get("plan_ms", None)
        if avg_time is not None:
            pbar.set_postfix({"last_plan_ms": f"{avg_time:.1f}"})

    # Concatenate and save trajectories
    all_trajectories = np.concatenate(all_trajectories, axis=0)  # (N_samples, T, 4)
    traj_out_path = os.path.join(args.output_dir, "predicted_trajectories.npy")
    np.save(traj_out_path, all_trajectories)
    print(f"Saved trajectories → {traj_out_path}")

    # Save timing/metrics CSV
    metrics_csv = os.path.join(args.output_dir, "eval_metrics.csv")
    pipeline.save_metrics(metrics_csv)
    print(f"Saved evaluation metrics → {metrics_csv}")


if __name__ == "__main__":
    main()
