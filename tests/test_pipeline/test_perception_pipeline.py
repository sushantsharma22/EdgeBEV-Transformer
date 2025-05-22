import torch
import numpy as np
import pytest
from config.model_config import ModelConfig
from src.pipeline.perception_pipeline import PerceptionPipeline
from src.core.data_structures import BatchData, CameraImage, SensorReading, MapData, TrajectoryPoint

@pytest.fixture
def dummy_batch(tmp_path):
    # Create a BatchData with B=1, cameras=2, T=2, H=W=8
    B, cams, T, C, H, W = 1, 2, 2, 3, 8, 8
    imgs = []
    for _ in range(B):
        imgs.append([np.zeros((H,W,C), dtype=np.float32) for _ in range(cams*T)])
    sensors = []
    maps = []
    # single sensor reading
    sensors.append(SensorReading(
        timestamp=0.0,
        speed=0.0,
        imu={"acc_x":0,"gyro_z":0},
        steering_angle=0.0,
        gnss={"lat":0,"lon":0,"alt":0}
    ))
    # single map
    maps.append(MapData(timestamp=0.0, region_id="0,0,0,0", waypoints=[], road_graph=None))
    return BatchData(timestamps=[0.0], images=imgs, sensors=sensors, map_data=maps)

def test_pipeline_forward(dummy_batch):
    cfg = ModelConfig()
    pipeline = PerceptionPipeline(cfg).eval()
    out = pipeline(dummy_batch)
    # trajectories
    traj = out["trajectories"]
    assert isinstance(traj, torch.Tensor)
    assert traj.shape[0] == 1 and traj.shape[1] == cfg.trajectory_planner.horizon_steps
    # intermediates present
    for key in ("features","aligned","bev","fused"):
        assert key in out["intermediates"]
    # timings
    assert "encode_ms" in out["timings_ms"]
