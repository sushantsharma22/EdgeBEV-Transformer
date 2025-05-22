#!/usr/bin/env python3
# scripts/benchmark.py

import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from config.deployment_config import DeploymentConfig
from src.pipeline.inference_engine import InferenceEngine
from src.data.data_loader import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference latency & throughput")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML model + deployment config")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Manifest JSON for test dataset")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root dir for images/sensors")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override num_workers from config")
    parser.add_argument("--max-batches", type=int, default=50,
                        help="Max batches to run for benchmarking")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations to skip timing")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configs
    model_cfg = ModelConfig.load_from_yaml(args.config)
    deploy_cfg = DeploymentConfig.load_from_yaml(args.config)
    if args.batch_size:
        model_cfg.batch_size = args.batch_size
    if args.num_workers:
        model_cfg.num_workers = args.num_workers

    # Prepare DataLoader
    dl: DataLoader = get_dataloader(
        data_root=args.data_root,
        manifest_path=args.manifest,
        batch_size=model_cfg.batch_size,
        temporal_window=model_cfg.image_encoder.temporal_window,
        stride=model_cfg.temporal_stride,
        num_workers=model_cfg.num_workers,
        shuffle=False
    )

    # Build engine
    engine = InferenceEngine(model_cfg, deploy_cfg, weights_path=args.weights)

    # Warmup
    for i, batch in enumerate(dl):
        _ = engine.run_batch(batch, with_timings=False)
        if i >= args.warmup:
            break

    # Benchmark loop
    latencies = []
    samples = 0
    for i, batch in enumerate(dl):
        if i >= args.max_batches:
            break
        t0 = time.time()
        out = engine.run_batch(batch, with_timings=False)
        t1 = time.time()
        latencies.append((t1 - t0) * 1e3)
        samples += len(batch.images)

    latencies = np.array(latencies)
    print(f"Ran {len(latencies)} batches, {samples} samples total")
    print(f"Batch latency (ms): mean={latencies.mean():.2f}, p50={np.percentile(latencies,50):.2f}, p90={np.percentile(latencies,90):.2f}, p99={np.percentile(latencies,99):.2f}")
    throughput = samples / (latencies.sum() / 1e3)
    print(f"Throughput: {throughput:.1f} samples/sec")


if __name__ == "__main__":
    main()
