# src/utils/optimization.py

import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import (
    quantize_dynamic,
    prepare_qat,
    convert,
    get_default_qconfig,
    default_dynamic_qconfig,
)
from typing import Optional, Dict


def apply_dynamic_quantization(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    modules_to_quantize: Optional[Dict[str, type]] = None,
) -> nn.Module:
    """
    Apply post‐training dynamic quantization to reduce model size & latency.
    By default, quantizes nn.Linear & nn.LSTM layers to int8.

    Args:
        model: PyTorch nn.Module to quantize.
        dtype: Quantized dtype (e.g. torch.qint8).
        modules_to_quantize: Optional dict mapping module names to types.
                             Defaults to {'': [nn.Linear, nn.LSTM]}.
    Returns:
        A quantized nn.Module.
    """
    if modules_to_quantize is None:
        modules_to_quantize = {
            "": [nn.Linear, nn.LSTM]
        }
    return quantize_dynamic(
        model,
        {mod for mods in modules_to_quantize.values() for mod in mods},
        dtype=dtype,
    )


def apply_static_quantization(
    model: nn.Module,
    example_inputs: tuple,
    backend: str = "fbgemm",
    qconfig_name: str = "default",
    output_dir: Optional[str] = None,
) -> nn.Module:
    """
    Apply post‐training static (calibrated) quantization.
    This inserts observers, runs a calibration pass, then converts.

    Args:
        model: PyTorch nn.Module in eval() mode.
        example_inputs: Tuple of inputs to run for calibration.
        backend: Quantization backend ('fbgemm' or 'qnnpack').
        qconfig_name: 'default' or 'qat' for QAT-style fake quant.
        output_dir: Optional path to save intermediate models.
    Returns:
        A statically quantized nn.Module.
    """
    assert not model.training, "Model must be in eval mode for static quantization."
    torch.backends.quantized.engine = backend

    # Choose qconfig
    if qconfig_name == "default":
        qconfig = get_default_qconfig(backend)
    elif qconfig_name == "qat":
        qconfig = get_default_qconfig(backend)
        model.train()
        prepare_qat(model, inplace=True)
    else:
        raise ValueError(f"Unsupported qconfig_name '{qconfig_name}'")

    model.qconfig = qconfig
    prepare = torch.quantization.prepare if qconfig_name == "default" else prepare_qat
    prepare(model, inplace=True)

    # Calibration run
    with torch.no_grad():
        model(*example_inputs)

    # Convert to quantized model
    model.eval()
    quantized = convert(model, inplace=True)

    # Optionally save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(quantized.state_dict(), os.path.join(output_dir, "static_quantized.pt"))

    return quantized


def prune_model_global(
    model: nn.Module,
    amount: float = 0.2,
    pruning_method: prune.BasePruningMethod = prune.L1Unstructured,
):
    """
    Apply global unstructured pruning across all Linear layers.

    Args:
        model: nn.Module to prune.
        amount: Fraction of connections to prune globally.
        pruning_method: A pruning method class from torch.nn.utils.prune.
    """
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=pruning_method,
        amount=amount,
    )
    # Make pruning permanent
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")


def prune_model_layerwise(
    model: nn.Module,
    amount_dict: Dict[str, float],
    pruning_method: prune.BasePruningMethod = prune.L1Unstructured,
):
    """
    Apply unstructured pruning per-layer with specified amounts.

    Args:
        model: nn.Module to prune.
        amount_dict: Dict mapping module names to prune amounts.
                     e.g. {'image_encoder.backbone.0': 0.3}
        pruning_method: Pruning method class from torch.nn.utils.prune.
    """
    for name, module in model.named_modules():
        if name in amount_dict and hasattr(module, "weight"):
            prune.l1_unstructured(
                module,
                name="weight",
                amount=amount_dict[name],
            )
            prune.remove(module, "weight")


def fuse_modules_for_quantization(model: nn.Module, inplace: bool = True):
    """
    Fuse common sequences of modules to improve quantization performance.
    E.g., Conv+BN+ReLU → ConvBnReLU.

    This function should be customized per-model architecture.
    """
    # Example for torchvision ResNet-like modules:
    for module_name, module in model.named_children():
        # Recursively apply
        fuse_modules_for_quantization(module, inplace=inplace)
        # Fuse Conv+BN and Conv+BN+ReLU if found
        if hasattr(module, "conv1") and hasattr(module, "bn1"):
            torch.quantization.fuse_modules(
                module,
                ["conv1", "bn1", "relu"],
                inplace=inplace,
            )


def optimize_for_edge(
    model: nn.Module,
    backend: str = "dynamic",
    example_inputs: Optional[tuple] = None,
    prune_amount: float = 0.0,
    quant_backend: str = "fbgemm",
    output_dir: Optional[str] = None,
) -> nn.Module:
    """
    High-level helper to apply a chain of optimizations for edge deployment.

    Args:
        model: The nn.Module to optimize.
        backend: 'dynamic' or 'static' quantization.
        example_inputs: Required for static quantization.
        prune_amount: If >0, applies global pruning first.
        quant_backend: Backend for static quantization.
        output_dir: Directory to save intermediate artifacts.
    Returns:
        Optimized nn.Module.
    """
    optimized = model

    # 1) Prune
    if prune_amount > 0:
        prune_model_global(optimized, amount=prune_amount)

    # 2) Quantize
    if backend == "dynamic":
        optimized = apply_dynamic_quantization(optimized)
    elif backend == "static":
        assert example_inputs is not None, "example_inputs needed for static quantization"
        fuse_modules_for_quantization(optimized)
        optimized = apply_static_quantization(
            optimized,
            example_inputs=example_inputs,
            backend=quant_backend,
            qconfig_name="default",
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Unknown quantization backend '{backend}'")

    return optimized
