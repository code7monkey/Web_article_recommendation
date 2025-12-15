"""Placeholder for custom loss functions.

This module is intentionally left blank to allow users to implement
their own loss functions in the future. In this simple example,
recommendations are generated heuristically rather than through
gradient‑based optimisation, so no loss is currently used.
"""

from __future__ import annotations

import numpy as np

def dummy_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """A dummy loss that computes mean squared error.

    Args:
        predictions: Predicted scores.
        targets: Ground truth scores.
    Returns:
        Mean squared error between predictions and targets.
    """
    return float(np.mean((predictions - targets) ** 2))
