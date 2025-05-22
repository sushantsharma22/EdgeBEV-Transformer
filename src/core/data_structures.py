# src/core/data_structures.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class CameraImage:
    """
    Representation of a single camera frame.
    """
    timestamp: float                 # seconds since epoch or t0
    camera_id: str                   # e.g., "front_left", "rear_right"
    image: np.ndarray                # H×W×C array (uint8 or float32)


@dataclass
class SensorReading:
    """
    Collection of scalar sensor measurements at a given timestamp.
    """
    timestamp: float
    speed: float                     # vehicle speed (m/s)
    imu: Dict[str, float]            # e.g., {"acc_x":…, "acc_y":…, "gyro_z":…}
    steering_angle: float            # radians or degrees
    gnss: Dict[str, float]           # {"lat":…, "lon":…, "alt":…}


@dataclass
class MapData:
    """
    Geospatial map information from OpenStreetMap.
    """
    timestamp: float
    region_id: str                   # unique identifier for map tile/region
    waypoints: List[Dict[str, float]]  # list of {"lat":…, "lon":…}
    road_graph: Any = None           # raw OSM graph or preprocessed adjacency


@dataclass
class FrameData:
    """
    All inputs at a single time-step (t).
    """
    timestamp: float
    images: List[CameraImage]
    sensors: SensorReading
    map_data: Optional[MapData] = None


@dataclass
class BatchData:
    """
    Batched data for training/inference.
    """
    timestamps: List[float]
    # shape: (batch, cameras, H, W, C)
    images: List[List[np.ndarray]]
    # list length = batch; each entry is SensorReading
    sensors: List[SensorReading]
    # optional per-sample map data
    map_data: Optional[List[MapData]] = None


@dataclass
class TrajectoryPoint:
    """
    Single point in a planned trajectory.
    """
    time_offset: float               # seconds from current t
    x: float                         # x-offset in vehicle frame (m)
    y: float                         # y-offset in vehicle frame (m)
    heading: float                   # orientation (rad)


@dataclass
class Trajectory:
    """
    Full planned trajectory output.
    """
    start_timestamp: float
    points: List[TrajectoryPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
