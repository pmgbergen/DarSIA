"""Frozen phase inference with explicit models, geometry and rule parameters.

The numerical entry point is loaded lazily so importing this package does not
load image data, model files, a review application or a campaign driver.
"""

from importlib import import_module

__all__ = ["infer_labels"]


def __getattr__(name):
    if name == "infer_labels":
        value = import_module(".infer", __name__).infer_labels
        globals()[name] = value
        return value
    raise AttributeError(name)
