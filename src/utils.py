"""Utility functions for the NewsRec project.

This module houses helper utilities such as seed setting to
ensure deterministic behaviour across runs.
"""

import os
import random
from typing import Optional

import numpy as np

def seed_everything(seed: int = 42) -> None:
    """Set seed for Python, NumPy and environment to ensure reproducibility.

    Args:
        seed: The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        # If PyTorch is installed, set its seed as well.
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
    except ImportError:
        # Torch is optional; ignore if not available.
        pass
