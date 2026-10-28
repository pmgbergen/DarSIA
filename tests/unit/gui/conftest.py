"""Pytest configuration for GUI tests.

Sets up headless Qt rendering (offscreen platform) for test environments without
a display. This is safe to use in CI/CD and development environments alike, since
it only applies the platform if not already set.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def quick_background_reads(monkeypatch):
    """Answer the window's background readings at once in tests.

    The window waits a moment for a reading and shows a slow one later. The
    tests' files are local, so the reading is waited for in full and the
    window shows its answer as a synchronous read would. Tests of the later
    answer set a short wait themselves.
    """
    from darsia.fluidflower import background_read

    monkeypatch.setattr(background_read, "QUICK_ANSWER_S", 60.0)
