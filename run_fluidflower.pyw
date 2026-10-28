"""Windows GUI bootstrap for uv: use real pythonw with this installed venv.

uv's Windows virtual-environment pythonw shim can forward to python.exe. The
real GUI interpreter avoids a console; site.addsitedir processes only the
central environment's editable-install metadata, with no legacy source path.
"""
from pathlib import Path
import os
import site
import sys
import traceback

environment = Path(__file__).resolve().parent / ".venv-fluidflower"
site.addsitedir(str(environment / "Lib" / "site-packages"))
os.environ["VIRTUAL_ENV"] = str(environment)
os.environ["DARSIA_FLUIDFLOWER_PYTHON"] = str(environment / "Scripts" / "python.exe")
os.environ["PATH"] = str(environment / "Scripts") + os.pathsep + os.environ.get("PATH", "")
application = sys.argv.pop(1) if len(sys.argv) > 1 else "studio"
try:
    if application == "studio":
        from darsia.fluidflower.phase_model_app import main
    elif application == "experiments":
        from darsia.fluidflower.app import main
    elif application == "native":
        from darsia.fluidflower.native_darsia_app import main
    else:
        raise ValueError(f"Unknown FluidFlower application: {application}")
    raise SystemExit(main())
except SystemExit:
    raise
except BaseException:
    if sys.stderr is not None:
        traceback.print_exc()
    raise
