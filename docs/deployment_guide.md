# Deployment Guide

## 1. Export Model

Choose backend in `deployment_config.yaml`:

- **TorchScript**:  
  ```bash
  tpp-deploy --config config/deployment_config.yaml \
             --weights path/to/checkpoint.pt 
  tpp-deploy --config config/deployment_config.yaml \
           --weights path/to/checkpoint.pt \
           --manifest data/manifest.json
