#!/usr/bin/env python3
# scripts/train.py

import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.model_config import ModelConfig
from config.base_config import BaseConfig
from src.data.data_loader import get_dataloader
from src.pipeline.perception_pipeline import PerceptionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Transformer Perception Pipeline")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML model config (model_config.yaml)")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to dataset manifest JSON")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root directory for image and sensor data")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write logs, checkpoints, metrics")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override num_workers from config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load default base config for output dirs and device
    base_cfg = BaseConfig.load_from_yaml(args.config) if os.path.isfile(args.config) else BaseConfig()
    output_dir = args.output_dir or base_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load model config (with nested sections)
    model_cfg = ModelConfig.load_from_yaml(args.config)
    if args.batch_size is not None:
        model_cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        model_cfg.num_workers = args.num_workers
    if args.lr is not None:
        model_cfg.learning_rate = args.lr
    model_cfg.output_dir = output_dir

    # Build data loader
    dataloader: DataLoader = get_dataloader(
        data_root=args.data_root,
        manifest_path=args.manifest,
        batch_size=model_cfg.batch_size,
        temporal_window=model_cfg.image_encoder.temporal_window,
        stride=model_cfg.temporal_stride,
        num_workers=model_cfg.num_workers,
        shuffle=True,
    )

    # Build model & optimizer
    device = torch.device(model_cfg.device)
    pipeline = PerceptionPipeline(model_cfg, device=device).train()
    optimizer = optim.AdamW(
        pipeline.parameters(),
        lr=model_cfg.learning_rate,
        weight_decay=model_cfg.weight_decay,
    )
    # Learning rate schedule (linear warmup + decay)
    def lr_lambda(step):
        if step < model_cfg.warmup_steps:
            return float(step) / float(max(1, model_cfg.warmup_steps))
        return max(
            0.0,
            float(model_cfg.total_steps - step) / float(max(1, model_cfg.total_steps - model_cfg.warmup_steps))
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    global_step = 0

    # Resume?
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        pipeline.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume}, starting epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
        for batch in epoch_pbar:
            optimizer.zero_grad()
            out = pipeline(batch)
            # Assume a simple L2 loss on trajectory positions
            pred = out["trajectories"]  # (B, T, 4)
            # ground-truth would need to be part of batch—here we skip actual loss
            # loss = compute_loss(pred, batch.gt_trajectory)
            # For demo purposes, use dummy loss:
            loss = pred.norm()
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "step": global_step,
            "model_state": pipeline.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
        }, ckpt_path)

    # Export final metrics CSV
    pipeline.save_metrics(os.path.join(output_dir, "pipeline_metrics.csv"))
    print(f"Training complete. Checkpoints and metrics saved to {output_dir}")


if __name__ == "__main__":
    main()
