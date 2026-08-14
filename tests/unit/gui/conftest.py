"""Pytest configuration for GUI tests.

Sets up headless Qt rendering (offscreen platform) for test environments without
a display. This is safe to use in CI/CD and development environments alike, since
it only applies the platform if not already set.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
