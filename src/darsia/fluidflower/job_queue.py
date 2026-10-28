"""Compatibility alias for :mod:`darsia.fluidflower.pipeline.job_queue`."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("darsia.fluidflower.pipeline.job_queue")
