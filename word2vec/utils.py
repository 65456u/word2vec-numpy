from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """
    Numerically stable sigmoid that works for scalars and numpy arrays.
    """
    x_array = np.asarray(x)
    positive = x_array >= 0
    negative = ~positive

    result = np.empty_like(x_array, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-x_array[positive]))

    exp_x = np.exp(x_array[negative])
    result[negative] = exp_x / (1.0 + exp_x)

    if np.isscalar(x):
        return float(result.item())
    return result
