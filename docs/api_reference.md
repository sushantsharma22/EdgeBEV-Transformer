# API Reference

## Configs

- `BaseConfig`  
  - `.load_from_yaml(path)`, `.save_to_yaml(path)`, `.to_dict()`  
- `ModelConfig` (extends `BaseConfig`)  
  - Nested sub-configs: `.image_encoder`, `.spatial_transformer`, `.bev_former`, `.fusion_transformer`, `.trajectory_planner`  
- `DeploymentConfig` (extends `BaseConfig`)  
  - `.validate()`, `.compile_backend`, `.quantization`, etc.

## Core

- `BaseModule(config)` – abstract; implements `.build()`, `.forward()`, weight I/O  
- Data structs:  
  - `CameraImage`, `SensorReading`, `MapData`, `FrameData`, `BatchData`, `Trajectory`, `TrajectoryPoint`  
- `MetricsLogger` – `.log(step, phase, metrics)`, `.save_csv(path)`, `.get_latest()`

## Models

- `ImageEncoder(cfg)` – `(B, cams, T, C, H, W) → (B, seq, F)`  
- `SpatialTransformer(cfg)` – `(B, seq, F) → (B, seq, F)`  
- `BEVFormer(cfg)` – `(B, seq, E) → (B, H×W, E)`  
- `FusionTransformer(cfg)` – `(B, seq_bev, E), sensor_dict, map_feats → (B, seq_all, E)`  
- `TrajectoryPlanner(cfg)` – `(B, seq_all, E) → (B, horizon, 4)`

## Pipeline & Engine

- `PerceptionPipeline(config, device)`  
  - `.forward(batch: BatchData) → {trajectories, intermediates, timings_ms}`  
  - `.save_metrics(path)`
- `InferenceEngine(model_cfg, deploy_cfg, weights_path)`  
  - `.run_batch(batch, with_timings)`, `.run_dataloader(dl, ...)`, `.export(...)`

## Utilities

- `src/data/*` — `PerceptionDataset`, `Preprocessor`, `MapHandler`  
- `src/utils/*` — transforms, visualization helpers, quantization/pruning helpers  
