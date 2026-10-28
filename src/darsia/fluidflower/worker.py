"""Compatibility module and CLI for the whole-run worker."""
from importlib import import_module
import sys

_implementation = import_module("darsia.fluidflower.pipeline.worker")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
elif __name__ != "__mp_main__":
    sys.modules[__name__] = _implementation
