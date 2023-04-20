"""
Function for efficient array slice along axis.
Sugestion found in the thread:
        https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def array_slice(
    a: np.ndarray,
    axis: int,
    start: Optional[int],
    end: Optional[int],
    step: int = 1,
) -> np.ndarray:
    """
    Slice array along axis.

    Input:
        a (np.ndarra): array to slice
        axis (int): axis to slice along
        start (int): start index
        end (int): end index
        step (int): step size

        Output:
            np.ndarray: sliced array
    """
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]
