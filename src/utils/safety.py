"""Safety utilities for tensor validation, NaN detection, and sanity checks.

This module provides functions to detect anomalies during training:
- NaN/Inf detection in tensors
- Gradient anomaly detection
- Shape validation
- Device consistency checks
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Union

import torch
from torch import nn

from utils.logging import get_logger

logger = get_logger(__name__)

# Environment variable to enable strict checking (raises exceptions)
_STRICT_CHECKS = os.getenv("CFT_STRICT_TENSOR_CHECKS", "0") == "1"


class TensorAnomalyError(RuntimeError):
    """Raised when a tensor contains NaN or Inf values in strict mode."""
    pass


class ShapeMismatchError(ValueError):
    """Raised when tensor shapes don't match expected dimensions."""
    pass


def check_tensor_finite(
    tensor: torch.Tensor,
    name: str = "tensor",
    *,
    strict: Optional[bool] = None,
    context: Optional[Dict[str, object]] = None,
) -> bool:
    """Check if a tensor contains only finite values (no NaN or Inf).
    
    Args:
        tensor: The tensor to check.
        name: Name for logging/error messages.
        strict: If True, raise TensorAnomalyError on anomaly. 
                Defaults to CFT_STRICT_TENSOR_CHECKS env var.
        context: Additional context dict for error messages (e.g., batch_idx, epoch).
    
    Returns:
        True if tensor is finite, False otherwise.
        
    Raises:
        TensorAnomalyError: If strict=True and tensor contains NaN/Inf.
    """
    if not torch.is_tensor(tensor):
        return True
    
    if strict is None:
        strict = _STRICT_CHECKS
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        ctx_str = ""
        if context:
            ctx_str = " | context: " + ", ".join(f"{k}={v}" for k, v in context.items())
        
        anomaly_type = []
        if has_nan:
            anomaly_type.append("NaN")
        if has_inf:
            anomaly_type.append("Inf")
        
        msg = (
            f"Tensor '{name}' contains {' and '.join(anomaly_type)} values. "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"device={tensor.device}{ctx_str}"
        )
        
        if strict:
            raise TensorAnomalyError(msg)
        else:
            logger.warning(msg)
        
        return False
    
    return True


def check_loss_finite(
    losses: Dict[str, Union[torch.Tensor, float]],
    *,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    strict: Optional[bool] = None,
) -> bool:
    """Check all losses in a dict for NaN/Inf values.
    
    Args:
        losses: Dict mapping loss names to tensor or float values.
        step: Current training step (for context).
        epoch: Current epoch (for context).
        strict: If True, raise on first anomaly.
        
    Returns:
        True if all losses are finite.
    """
    context = {}
    if step is not None:
        context["step"] = step
    if epoch is not None:
        context["epoch"] = epoch
    
    all_finite = True
    for name, value in losses.items():
        if torch.is_tensor(value):
            if not check_tensor_finite(value, name=name, strict=strict, context=context):
                all_finite = False
        elif isinstance(value, float):
            import math
            if math.isnan(value) or math.isinf(value):
                ctx_str = ""
                if context:
                    ctx_str = " | context: " + ", ".join(f"{k}={v}" for k, v in context.items())
                msg = f"Loss '{name}' is {value}{ctx_str}"
                
                if strict or _STRICT_CHECKS:
                    raise TensorAnomalyError(msg)
                else:
                    logger.warning(msg)
                all_finite = False
    
    return all_finite


def check_gradients_finite(
    model: nn.Module,
    name: str = "model",
    *,
    strict: Optional[bool] = None,
    context: Optional[Dict[str, object]] = None,
) -> bool:
    """Check all parameter gradients in a model for NaN/Inf.
    
    Args:
        model: The model to check.
        name: Name for logging.
        strict: If True, raise on anomaly.
        context: Additional context for error messages.
        
    Returns:
        True if all gradients are finite (or None).
    """
    all_finite = True
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            if not check_tensor_finite(
                param.grad,
                name=f"{name}.{param_name}.grad",
                strict=strict,
                context=context,
            ):
                all_finite = False
                if strict or _STRICT_CHECKS:
                    break  # Already raised
    
    return all_finite


def validate_cam_shape(
    cam: torch.Tensor,
    batch_size: int,
    name: str = "cam",
) -> None:
    """Validate that a CAM tensor has the expected shape.
    
    Args:
        cam: The CAM tensor, expected shape [B, H, W].
        batch_size: Expected batch size.
        name: Name for error messages.
        
    Raises:
        ShapeMismatchError: If shape doesn't match.
    """
    if cam.ndim != 3:
        raise ShapeMismatchError(
            f"{name} must be 3D [B, H, W], got shape {tuple(cam.shape)}"
        )
    
    if cam.shape[0] != batch_size:
        raise ShapeMismatchError(
            f"{name} batch size mismatch: expected {batch_size}, "
            f"got {cam.shape[0]} (shape={tuple(cam.shape)})"
        )


def validate_mask_shape(
    mask: torch.Tensor,
    batch_size: int,
    name: str = "mask",
) -> None:
    """Validate that a mask tensor has a valid shape.
    
    Args:
        mask: The mask tensor, expected [B, H, W] or [B, 1, H, W].
        batch_size: Expected batch size.
        name: Name for error messages.
        
    Raises:
        ShapeMismatchError: If shape is invalid.
    """
    if mask.ndim not in (3, 4):
        raise ShapeMismatchError(
            f"{name} must be 3D [B, H, W] or 4D [B, 1, H, W], "
            f"got shape {tuple(mask.shape)}"
        )
    
    if mask.shape[0] != batch_size:
        raise ShapeMismatchError(
            f"{name} batch size mismatch: expected {batch_size}, "
            f"got {mask.shape[0]} (shape={tuple(mask.shape)})"
        )


def validate_device_consistency(
    tensors: Dict[str, torch.Tensor],
    expected_device: Optional[torch.device] = None,
) -> torch.device:
    """Validate that all tensors are on the same device.
    
    Args:
        tensors: Dict mapping names to tensors.
        expected_device: If provided, validate against this device.
        
    Returns:
        The common device.
        
    Raises:
        ValueError: If tensors are on different devices.
    """
    if not tensors:
        return expected_device or torch.device("cpu")
    
    devices = {}
    for name, t in tensors.items():
        if torch.is_tensor(t):
            devices[name] = t.device
    
    unique_devices = set(devices.values())
    if len(unique_devices) > 1:
        device_info = ", ".join(f"{k}: {v}" for k, v in devices.items())
        raise ValueError(
            f"Tensors are on different devices: {device_info}"
        )
    
    actual_device = next(iter(unique_devices)) if unique_devices else torch.device("cpu")
    
    if expected_device is not None and actual_device != expected_device:
        raise ValueError(
            f"Expected device {expected_device}, but tensors are on {actual_device}"
        )
    
    return actual_device


def log_tensor_stats(
    tensor: torch.Tensor,
    name: str,
    *,
    level: str = "debug",
) -> None:
    """Log basic statistics about a tensor for debugging.
    
    Args:
        tensor: The tensor to summarize.
        name: Name for logging.
        level: Log level ("debug", "info", "warning").
    """
    with torch.no_grad():
        stats = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
            "mean": float(tensor.float().mean().item()),
            "std": float(tensor.float().std().item()) if tensor.numel() > 1 else 0.0,
        }
    
    msg = f"Tensor '{name}': " + ", ".join(f"{k}={v}" for k, v in stats.items())
    
    # ALLOWED per STRICTNESS.md: dynamic logger method dispatch based on level string
    log_fn = getattr(logger, level, logger.debug)
    log_fn(msg)
