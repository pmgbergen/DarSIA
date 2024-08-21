"""Introduce concept of formats: Scalar, Vector, Tensor through enumeration."""

# Introduce concept of formats: Scalar, Vector, Tensor through enumeration
from enum import Enum


class Format(Enum):
    """Enumeration of formats for data."""

    SCALAR = 0
    VECTOR = 1
    TENSOR = 2
