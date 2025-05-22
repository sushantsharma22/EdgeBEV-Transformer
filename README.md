# EdgeBEV-Transformer
End-to-end, multi-modal transformer perception pipeline for real-time trajectory planning on edge devices

# Transformer Perception Pipeline

End-to-end transformer-based perception stack that goes from raw multi-camera images + sensors + map data → bird’s-eye-view features → fused representation → planned trajectory. Designed for edge deployment with no external dependencies at runtime.

## Features

- Multi-camera, multi-temporal image encoding  
- Spatial transformer for pattern alignment  
- BEV projection via cross-attention  
- Multi-modal fusion (vision + IMU + speed + GNSS + steering + map)  
- Transformer-based trajectory planner  
- Built-in metrics logging (CSV) and per-stage timing  
- Edge-friendly export (TorchScript, ONNX) + quantization/pruning helpers 
