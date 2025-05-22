# tests/test_data/test_preprocessor.py

import pytest
import numpy as np
import json
from PIL import Image
from tempfile import NamedTemporaryFile
from src.data.preprocessor import Preprocessor


def test_load_image_file_not_found():
    pr = Preprocessor()
    with pytest.raises(FileNotFoundError):
        pr.load_image("nonexistent_image.png")


def test_load_image_and_normalize(tmp_path):
    # Create a dummy image file
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (32, 32), color=(128, 64, 32))
    img.save(str(img_path))

    pr = Preprocessor(image_size=(16, 16), normalize=True)
    arr = pr.load_image(str(img_path))
    # Expect shape (16,16,3) and dtype float32
    assert arr.shape == (16, 16, 3)
    assert arr.dtype == np.float32
    # Values should be roughly standardized (mean ~0)
    assert np.allclose(arr.mean(), 0, atol=1)


def test_load_sensor_json(tmp_path):
    # Create dummy sensor JSON
    data = {
        "speed": 5.5,
        "imu": {"acc_x":0.1, "gyro_z":0.01},
        "steering_angle": 0.2,
        "gnss": {"lat":1.0, "lon":2.0, "alt":3.0}
    }
    file_path = tmp_path / "sensor.json"
    file_path.write_text(json.dumps(data))

    pr = Preprocessor()
    sensor = pr.load_sensor(str(file_path))
    assert sensor["speed"] == pytest.approx(5.5)
    assert isinstance(sensor["imu"], dict)
    assert sensor["gnss"]["lat"] == pytest.approx(1.0)