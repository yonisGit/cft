from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np
import torch


def generate_random_seeds(n: int = 5, *, rng_seed: Optional[int] = None) -> List[int]:
    """Generate n random seeds for experiment reproducibility.
    
    Args:
        n: Number of seeds to generate.
        rng_seed: Optional seed for the RNG used to generate seeds.
                  If None, uses system entropy for true randomness.
    
    Returns:
        List of n random integers in range [0, 2^31-1].
    """
    if rng_seed is not None:
        rng = random.Random(rng_seed)
    else:
        rng = random.Random()  # Uses system entropy
    
    # Generate seeds in a range suitable for PyTorch/NumPy
    max_seed = 2**31 - 1
    return [rng.randint(0, max_seed) for _ in range(n)]


def seed_everything(seed: int, *, deterministic: bool, device: Optional[torch.device] = None) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if device is not None and device.type == "cuda":
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=False)
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        torch.set_float32_matmul_precision("high")
