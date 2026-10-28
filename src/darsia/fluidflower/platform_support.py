"""What this computer can start: the shared job layer runs on Windows only.

The job layer needs UNC paths and Windows process and file calls. On another
system (for example a Mac used for Setup) the pages that would prepare or
start a job show :data:`JOBS_ON_WINDOWS` instead.
"""

from __future__ import annotations

import os

JOBS_ON_WINDOWS = (
    "Jobs run on the Windows computers. Open this series there to prepare and start "
    "this work; what you saved here is already in the series folder."
)


def jobs_supported() -> bool:
    """Return True on Windows, the only system that prepares and runs jobs."""
    return os.name == "nt"
