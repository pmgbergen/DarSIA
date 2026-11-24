"""Integration tests run headless: no windows from matplotlib or Qt.

The environment variables also reach the example scripts run as subprocesses
(`test_examples.py`), whose `plt.show()` would otherwise block on a desktop.
"""

import os

import matplotlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")
