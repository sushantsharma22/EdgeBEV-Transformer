import os
import json
import numpy as np
import pytest
from torch.utils.data import DataLoader
from src.data.data_loader import PerceptionDataset, collate_fn, get_dataloader
from src.data.preprocessor import Preprocessor
from src.data.map_handler import MapHandler
from src.core.data_structures import FrameData, BatchData, CameraImage, SensorReading, MapData

def make_dummy_manifest(tmp_path, num_samples=5, cameras=2):
    manifest = []
    for i in range(num_samples):
        images = {f"cam{j}": f"img_{i}_{j}.png" for j in range(cameras)}
        sensor = f"sensor_{i}.json"
        manifest.append({
            "timestamp": float(i),
            "images": images,
            "sensor": sensor,
            "region_id": "0.0,0.0,0.1,0.1"
        })
        # create dummy files
        for j in range(cameras):
            img = (np.zeros((8,8,3), dtype=np.uint8) + i*10 + j).tolist()
            img_path = tmp_path / images[f"cam{j}"]
            Image = pytest.importorskip("PIL.Image").fromarray
            Image(np.array(img, dtype=np.uint8)).save(str(img_path))
        # sensor JSON
        sensor_data = {
            "speed": i,
            "imu": {"acc_x":0.0, "gyro_z":0.0},
            "steering_angle": 0.0,
            "gnss": {"lat":0.0, "lon":0.0, "alt":0.0}
        }
        (tmp_path / sensor).write_text(json.dumps(sensor_data))
    return manifest

def test_perception_dataset_and_collate(tmp_path):
    # write manifest
    manifest = make_dummy_manifest(tmp_path, num_samples=4, cameras=2)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    prep = Preprocessor(image_size=(8,8))
    mh = MapHandler(cache_dir=str(tmp_path))
    ds = PerceptionDataset(
        data_root=str(tmp_path),
        manifest_path=str(manifest_path),
        temporal_window=2,
        stride=1,
        preprocess=prep,
        map_handler=mh
    )
    # length should be num_samples - (window-1)*stride = 4 -1*1 =3
    assert len(ds) == 3

    # get one item
    frame = ds[1]
    assert isinstance(frame, FrameData)
    assert len(frame.images) == 2 * 2  # cameras * window
    assert isinstance(frame.sensors, SensorReading)
    assert isinstance(frame.map_data, MapData)

    # test collate_fn
    batch = [ds[i] for i in range(len(ds))]
    batched = collate_fn(batch)
    assert isinstance(batched, BatchData)
    assert len(batched.images) == len(batch)
    # images are nested lists of np.ndarray
    assert all(isinstance(img, np.ndarray) for sub in batched.images for img in sub)

def test_get_dataloader(tmp_path):
    # reuse manifest
    manifest = make_dummy_manifest(tmp_path, num_samples=4, cameras=2)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    dl = get_dataloader(
        data_root=str(tmp_path),
        manifest_path=str(path),
        batch_size=2,
        temporal_window=2,
        stride=1,
        num_workers=0,
        shuffle=False
    )
    assert isinstance(dl, DataLoader)
    for batch in dl:
        # should yield a BatchData
        from src.core.data_structures import BatchData
        assert isinstance(batch, BatchData)
        break
