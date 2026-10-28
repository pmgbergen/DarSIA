"""Compatibility alias for :mod:`darsia.fluidflower.mass.concentration_joint`."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("darsia.fluidflower.mass.concentration_joint")
