# Architecture Overview

The pipeline is organized into modular stages:

1. **ImageEncoder**  
   - Multi‐camera, multi‐temporal ConvNet backbone → sequence of features  
   - Optional sinusoidal positional encoding  

2. **SpatialTransformer**  
   - Standard TransformerEncoder to align patterns across spatial–temporal tokens  

3. **BEVFormer**  
   - Learnable BEV queries + cross‐attention (TransformerDecoder)  
   - Projects features into bird’s‐eye‐view grid  

4. **FusionTransformer**  
   - Projects sensor & map data into the same embedding space  
   - Concatenates vision + modalities → TransformerEncoder  

5. **TrajectoryPlanner**  
   - Learnable trajectory queries + TransformerDecoder  
   - Predicts (x,y) offsets, heading, velocity per step  

All modules subclass `BaseModule` for consistent config, I/O, and integration. The end‐to‐end `PerceptionPipeline` orchestrates data flow, timing, and metrics logging.
