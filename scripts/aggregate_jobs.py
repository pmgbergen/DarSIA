"""Compatibility CLI for :mod:`darsia.fluidflower.pipeline.aggregate`."""
from importlib import import_module
from pathlib import Path
import sys

# Use this checkout even when the interpreter has another editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_implementation = import_module("darsia.fluidflower.pipeline.aggregate")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
elif __name__ != "__mp_main__":
    sys.modules[__name__] = _implementation
