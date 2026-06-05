"""Compatibility fallback for SAM-Audio's optional xformers import.

SAM-Audio's perception model imports ``AttentionBias`` and ``fmha`` from
``xformers.ops`` at module import time. The model path used by this app defaults
to PyTorch SDPA attention, so xformers is not required for normal inference.
Linux aarch64 systems such as DGX Spark often cannot install a prebuilt xformers
wheel, so this module provides the minimum import-compatible surface only when
``xformers.ops`` is not installed.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import torch
from torch.nn import functional as F


class AttentionBias:
    """Placeholder matching the xformers.ops AttentionBias type."""


class _FMHA:
    @staticmethod
    def memory_efficient_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias=None,
    ) -> torch.Tensor:
        """SDPA-backed fallback for xformers' B/S/H/D attention layout."""
        if attn_bias is not None and not isinstance(attn_bias, torch.Tensor):
            raise RuntimeError("xformers AttentionBias requires the real xformers package.")

        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias)
        return output.transpose(1, 2).contiguous()


fmha = _FMHA()


def ensure_xformers_ops() -> None:
    """Register a minimal xformers.ops shim when xformers is unavailable."""
    xformers_spec = importlib.util.find_spec("xformers")
    ops_spec = importlib.util.find_spec("xformers.ops") if xformers_spec else None
    if ops_spec is not None:
        return

    xformers_module = sys.modules.get("xformers") or types.ModuleType("xformers")
    ops_module = types.ModuleType("xformers.ops")
    ops_module.AttentionBias = AttentionBias
    ops_module.fmha = fmha
    xformers_module.ops = ops_module
    sys.modules.setdefault("xformers", xformers_module)
    sys.modules["xformers.ops"] = ops_module
