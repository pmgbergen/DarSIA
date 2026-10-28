"""Opt-in local lower gas boundary; leaves published v1 support reproducible.

Reachability from a deep injector is not evidence that gas has occupied a
shallower finger at another horizontal position. Within observed columns,
freeze the lowest gas elevation per injection and prevent paths from crossing
below it. Outside observed columns, retain v1's directional migration rule:
do not fabricate a measured floor. This is a classification hypothesis, not a
transport solver, and the inherited histories still require human validation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import phase_plume
from .project import sha256_file

VERSION = "per-injection-curved-gas-support-v2-local-floor"


def load_profile(root):
    path = Path(root) / "phase_plume_support.json"
    if not path.is_file():
        return None
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("format") != VERSION:
        return None
    if not profile.get("runs"):
        raise ValueError("Invalid local-floor plume-support profile")
    phase_plume.PlumeSettings(**profile.get("settings", {}))
    return profile


def runtime_identity(profile):
    digest = hashlib.sha256(phase_plume.runtime_identity(profile).encode())
    digest.update(sha256_file(Path(__file__)).encode())
    return digest.hexdigest()


class LocalFloorSequence(phase_plume.PlumeSequence):
    def local_envelope(self, history):
        tolerance = self.near(history, self.settings.registration_tolerance_m) & self.valid
        rows = np.arange(len(self.y))[:, None]
        floor = np.max(np.where(tolerance, rows, -1), axis=0)
        # Unknown columns retain the directional rule, not a global flat floor.
        return ((floor[None, :] < 0) | (rows <= floor[None, :])) & self.valid

    def reach(self, corridor, history):
        envelope = self.local_envelope(history)
        return super().reach(corridor & envelope, history) & envelope
