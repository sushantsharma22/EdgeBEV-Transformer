# src/data/data_loader.py

import os
import json
from typing import List, Optional, Callable, Dict

import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.core.data_structures import (
    CameraImage,
    SensorReading,
    MapData,
    FrameData,
    BatchData,
)
from src.data.preprocessor import Preprocessor
from src.data.map_handler import MapHandler


class PerceptionDataset(Dataset):
    """
    Dataset for end-to-end perception pipeline.
    Expects a manifest JSON listing samples with:
      - timestamp (float)
      - images: dict of camera_id → image_path
      - sensor: path to sensor JSON/npz
      - region_id: map region identifier
    """

    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        temporal_window: int = 3,
        stride: int = 1,
        preprocess: Optional[Preprocessor] = None,
        map_handler: Optional[MapHandler] = None,
    ):
        self.data_root = data_root
        self.temporal_window = temporal_window
        self.stride = stride
        self.preprocess = preprocess or Preprocessor()
        self.map_handler = map_handler or MapHandler(cache_dir=os.path.join(data_root, "maps"))

        # Load manifest
        with open(manifest_path, "r") as f:
            self.samples: List[Dict] = json.load(f)

        # Sort by timestamp
        self.samples.sort(key=lambda x: x["timestamp"])

    def __len__(self) -> int:
        # Only indices where a full window is available
        return len(self.samples) - (self.temporal_window - 1) * self.stride

    def __getitem__(self, idx: int) -> FrameData:
        # collect temporal_window frames ending at idx + offset
        indices = [
            idx + self.stride * offset
            for offset in range(self.temporal_window)
        ]

        imgs = []
        sensors = None
        map_data = None
        timestamps = []

        for i in indices:
            sample = self.samples[i]
            timestamps.append(sample["timestamp"])

            # Load and preprocess all camera images for this timestamp
            cam_imgs = []
            for cam_id, rel_path in sample["images"].items():
                img_arr = self.preprocess.load_image(os.path.join(self.data_root, rel_path))
                cam_imgs.append(CameraImage(
                    timestamp=sample["timestamp"],
                    camera_id=cam_id,
                    image=img_arr
                ))
            imgs.extend(cam_imgs)

            # Load sensor data once (for the last frame only)
            if i == indices[-1]:
                sensor_dict = self.preprocess.load_sensor(os.path.join(self.data_root, sample["sensor"]))
                sensors = SensorReading(
                    timestamp=sample["timestamp"],
                    speed=sensor_dict["speed"],
                    imu=sensor_dict["imu"],
                    steering_angle=sensor_dict["steering_angle"],
                    gnss=sensor_dict["gnss"],
                )

                # Load map for this region
                map_struct = self.map_handler.get_region(sample["region_id"])
                map_data = MapData(
                    timestamp=sample["timestamp"],
                    region_id=sample["region_id"],
                    waypoints=map_struct.waypoints,
                    road_graph=map_struct.road_graph,
                )

        # Bundle into FrameData
        return FrameData(
            timestamp=timestamps[-1],
            images=imgs,
            sensors=sensors,
            map_data=map_data,
        )


def collate_fn(batch: List[FrameData]) -> BatchData:
    """
    Collate a list of FrameData into BatchData.
    """
    timestamps = [f.timestamp for f in batch]
    # images: List[List[np.ndarray]] of shape (B, cameras*temporal_window)
    images = [[ci.image for ci in f.images] for f in batch]
    sensors = [f.sensors for f in batch]
    map_data = [f.map_data for f in batch]

    return BatchData(
        timestamps=timestamps,
        images=images,
        sensors=sensors,
        map_data=map_data
    )


def get_dataloader(
    data_root: str,
    manifest_path: str,
    batch_size: int,
    temporal_window: int,
    stride: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Utility to create a DataLoader for training or inference.
    """
    preprocess = Preprocessor()
    map_handler = MapHandler(cache_dir=os.path.join(data_root, "maps"))
    dataset = PerceptionDataset(
        data_root=data_root,
        manifest_path=manifest_path,
        temporal_window=temporal_window,
        stride=stride,
        preprocess=preprocess,
        map_handler=map_handler,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
