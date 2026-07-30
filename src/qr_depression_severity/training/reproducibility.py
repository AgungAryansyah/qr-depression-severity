"""Deterministic execution controls."""

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def validate_precision(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if not torch.cuda.is_available():
        raise RuntimeError(f"{precision} precision requires CUDA")
    if precision == "fp16":
        return torch.float16
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    raise RuntimeError("bf16 precision is unavailable on this CUDA device")
