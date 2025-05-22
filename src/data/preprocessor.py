# src/data/preprocessor.py

import os
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


class Preprocessor:
    """
    Handles loading and basic preprocessing for images and sensor data.

    - Images are loaded via PIL, resized (optionally), converted to float32,
      normalized to [0,1], and then standardized by mean/std.
    - Sensor files (.json or .npz) are parsed into a dict with keys:
        'speed', 'imu', 'steering_angle', 'gnss'.
    """

    def __init__(
            self,
            image_size: Optional[Tuple[int, int]] = None,
            normalize: bool = True,
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            image_size: (width, height) to which all images are resized.
                        If None, no resizing is applied.
            normalize:  Whether to apply per‐channel standardization.
            mean:       Per‐channel means for standardization.
            std:        Per‐channel stds for standardization.
        """
        self.image_size = image_size
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def load_image(self, path: str) -> np.ndarray:
        """
        Load an image, convert to RGB, resize, normalize to [0,1],
        and (optionally) standardize.

        Returns:
            image: np.ndarray of shape (H, W, 3), dtype float32
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.image_size is not None:
                img = img.resize(self.image_size, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0

        if self.normalize:
            arr = (arr - self.mean) / self.std

        return arr

    def load_sensor(self, path: str) -> Dict[str, Any]:
        """
        Load sensor data from a .json or .npz file.

        Expects the following keys in the file:
          - speed:          float
          - imu:            dict of floats, e.g. {"acc_x":…, "gyro_z":…}
          - steering_angle: float
          - gnss:           dict of floats, e.g. {"lat":…, "lon":…}

        Returns:
            A dict with exactly those four entries, typed as:
            {
                "speed": float,
                "imu": Dict[str, float],
                "steering_angle": float,
                "gnss": Dict[str, float],
            }
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sensor file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r") as f:
                data = json.load(f)
        elif ext == ".npz":
            npz = np.load(path, allow_pickle=True)
            data = {
                k: (npz[k].item() if isinstance(npz[k], np.ndarray) and npz[k].shape == ()
                    else npz[k].tolist())
                for k in npz.files
            }
        else:
            raise ValueError(f"Unsupported sensor file type: {ext}")

        # Validate required keys
        required = ["speed", "imu", "steering_angle", "gnss"]
        for key in required:
            if key not in data:
                raise KeyError(f"Missing required sensor key '{key}' in {path}")

        return {
            "speed": float(data["speed"]),
            "imu": dict(data["imu"]),
            "steering_angle": float(data["steering_angle"]),
            "gnss": dict(data["gnss"]),
        }
