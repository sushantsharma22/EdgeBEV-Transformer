# src/pipeline/inference_engine.py

import os
import time
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader

from config.model_config import ModelConfig
from config.deployment_config import DeploymentConfig
from src.pipeline.perception_pipeline import PerceptionPipeline


class InferenceEngine:
    """
    Wraps the PerceptionPipeline for streamlined model loading,
    batch inference, and optional model export (TorchScript / ONNX).
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        deploy_cfg: DeploymentConfig,
        weights_path: Optional[str] = None,
    ):
        """
        Args:
            model_cfg: Instance of ModelConfig.
            deploy_cfg: Instance of DeploymentConfig.
            weights_path: Optional path to a saved state_dict to load.
        """
        self.model_cfg = model_cfg
        self.deploy_cfg = deploy_cfg

        # Device
        self.device = torch.device(deploy_cfg.device)
        torch.set_num_threads(deploy_cfg.num_inference_threads)

        # Build pipeline
        self.pipeline = PerceptionPipeline(model_cfg, device=self.device)

        # Load weights if provided
        if weights_path:
            state = torch.load(weights_path, map_location=self.device)
            self.pipeline.load_state_dict(state)

        # Switch to eval mode
        self.pipeline.eval()

        # Optionally compile to TorchScript
        self.compiled = None
        if deploy_cfg.compile_backend == "torchscript":
            self.compiled = torch.jit.script(self.pipeline)
            self.pipeline = self.compiled

    def run_batch(
        self,
        batch: Union[dict, torch.Tensor],
        with_timings: bool = False,
    ) -> dict:
        """
        Run inference on a single batch.

        Args:
            batch: A BatchData object or pre-processed torch inputs.
            with_timings: If True, include per-stage timings in the output.

        Returns:
            dict containing:
              - 'trajectories': Tensor of shape (B, T, 4)
              - 'intermediates': dict of module outputs
              - 'timings_ms': dict of per-stage ms timings (if with_timings)
        """
        # If using TorchScript, the forward signature may differ
        out = self.pipeline(batch)
        if not with_timings:
            out.pop("timings_ms", None)
        return out

    def run_dataloader(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        with_timings: bool = False,
    ) -> list:
        """
        Iterate over a DataLoader and collect outputs for each batch.

        Args:
            dataloader: torch.utils.data.DataLoader yielding BatchData.
            max_batches: Stop after this many batches (for benchmarking).
            with_timings: If True, collect and return per-batch timings.

        Returns:
            List of output dicts (see run_batch).
        """
        results = []
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            results.append(self.run_batch(batch, with_timings=with_timings))
        return results

    def export(
        self,
        export_dir: Optional[str] = None,
        sample_batch: Optional[dict] = None,
    ) -> str:
        """
        Export the model to ONNX or TorchScript for deployment.

        Args:
            export_dir: Directory to write the exported model.
            sample_batch: A representative BatchData or tensor dict to trace ONNX.

        Returns:
            Path to the exported model file.
        """
        export_dir = export_dir or self.deploy_cfg.export_dir
        os.makedirs(export_dir, exist_ok=True)

        if self.deploy_cfg.compile_backend == "torchscript":
            out_path = os.path.join(export_dir, "pipeline.pt")
            if isinstance(self.pipeline, torch.jit.ScriptModule):
                self.pipeline.save(out_path)
            else:
                # Fallback: script & save
                scripted = torch.jit.script(self.pipeline)
                scripted.save(out_path)
            return out_path

        elif self.deploy_cfg.compile_backend == "onnx":
            if sample_batch is None:
                raise ValueError("Must provide sample_batch for ONNX export")
            out_path = os.path.join(export_dir, "pipeline.onnx")
            # Prepare inputs: pipeline.forward expects a BatchData
            torch.onnx.export(
                self.pipeline,
                args=(sample_batch,),
                f=out_path,
                export_params=True,
                opset_version=self.deploy_cfg.onnx_opset,
                input_names=["batch"],
                output_names=["trajectories"],
                dynamic_axes={
                    "batch": {0: "batch_size"},
                    "trajectories": {0: "batch_size", 1: "horizon_steps"},
                },
            )
            return out_path

        else:
            raise NotImplementedError(
                f"Export for backend '{self.deploy_cfg.compile_backend}' is not supported."
            )
