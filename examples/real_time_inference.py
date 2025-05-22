"""
Simulated real-time loop: continuously reads latest frame(s) and runs the pipeline.
"""
import time
import cv2
import numpy as np
from config.model_config import ModelConfig
from src.pipeline.inference_engine import InferenceEngine
from src.core.data_structures import BatchData, CameraImage, SensorReading, MapData

# Initialize OpenCV video streams (assume 2 cameras here)
caps = [cv2.VideoCapture(i) for i in (0,1)]
cfg = ModelConfig.load_from_yaml("config/model_config.yaml")
deploy_cfg = cfg  # use same for simplicity
engine = InferenceEngine(cfg, deploy_cfg)

try:
    while True:
        frames = []
        ts = time.time()
        # Read each camera
        for cap in caps:
            ret, img = cap.read()
            if not ret:
                continue
            # convert BGR→RGB and normalize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            frames.append(img.astype(np.float32))
        # Dummy sensor & map
        sensors = [SensorReading(timestamp=ts, speed=0.0, imu={}, steering_angle=0.0, gnss={})]
        maps = [MapData(timestamp=ts, region_id="", waypoints=[], road_graph=None)]
        batch = BatchData(timestamps=[ts], images=[frames], sensors=sensors, map_data=maps)
        out = engine.run_batch(batch)
        traj = out["trajectories"].cpu().numpy()[0]
        print(f"Time {ts:.3f}: planned {traj.shape[0]} steps")
        time.sleep(0.05)
finally:
    for cap in caps:
        cap.release()
