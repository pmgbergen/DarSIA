"""Setup records keep shared paths on disk and this computer's paths in memory.

A temporary folder stands in for a share opened at /Volumes/<share>. Each
record family below stores a photograph or folder path: written on a computer
with a machine path map it must hold the shared spelling, read there it must
give the local path, and without a map it must be byte-for-byte as before.
"""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from darsia.fluidflower import storage_paths
from darsia.fluidflower.layer_painting import PaintingSession
from darsia.fluidflower.series_geometry import (
    RigFrame, native_series_link, save_rig_frame, saved_rig_frames, set_reference_run,
    stored_native_series_link,
)
from darsia.fluidflower.series_image_selection import _stored
from darsia.fluidflower.setup_crop_area import CropArea

SHARE = r"\\nas-host\Lab"
CORNERS = {"top_left": (10.0, 20.0), "top_right": (10.0, 900.0),
           "bottom_right": (600.0, 900.0), "bottom_left": (600.0, 20.0)}


class _Project:
    """The metadata part of a geometry package, kept in memory."""

    def __init__(self, link):
        self.manifest = {"tools": {"native_series": deepcopy(link)}}
        self.writes = []

    def update_metadata(self, section, values, *, updated_by="", reason=""):
        self.writes.append(deepcopy(dict(values)))
        self.manifest.setdefault(section, {}).update(deepcopy(dict(values)))


@pytest.fixture
def share(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    local = tmp_path / "Volumes" / "Lab"
    local.mkdir(parents=True)
    storage_paths.set_machine_path_map([(SHARE, str(local))])
    yield local
    storage_paths.set_machine_path_map(())


def test_painting_sessions_store_the_shared_photograph(share):
    photo = share / "Raw" / "AC1" / "DSC1.ARW"
    session = PaintingSession.default("ac1", photo, (6000, 4000))
    payload = session.to_dict()
    assert payload["image"]["path"] == SHARE + r"\Raw\AC1\DSC1.ARW"
    assert PaintingSession.from_dict(json.loads(json.dumps(payload))).image_path == str(photo)


def test_rig_frames_and_crop_areas_store_the_shared_photograph(share):
    photo = str(share / "Raw" / "AC1" / "DSC1.ARW")
    frame = RigFrame(corners=dict(CORNERS), width_m=0.9, height_m=0.5, run_id="ac1", image=photo)
    assert frame.to_dict()["image"] == SHARE + r"\Raw\AC1\DSC1.ARW"
    assert RigFrame.from_dict(frame.to_dict()).image == photo
    area = CropArea(corners=dict(CORNERS), run_id="ac1", image=photo, image_size=(6000, 4000))
    assert area.to_dict()["image"] == SHARE + r"\Raw\AC1\DSC1.ARW"
    assert CropArea.from_dict(area.to_dict()).image == photo


def test_the_series_link_is_read_here_and_written_as_stored(share):
    stored_root = SHARE.replace("\\", "/") + "/FF/DarSIA"
    project = _Project({"series_root": stored_root, "reference_run": "ac1"})
    assert native_series_link(project)["series_root"] == str(share / "FF" / "DarSIA")
    assert stored_native_series_link(project)["series_root"] == stored_root
    assert set_reference_run(project, "ac2")
    photo = str(share / "Raw" / "AC2" / "DSC2.ARW")
    save_rig_frame(project, RigFrame(corners=dict(CORNERS), width_m=0.9, height_m=0.5, run_id="ac2", image=photo))
    written = project.manifest["tools"]["native_series"]
    assert written["series_root"] == stored_root and written["reference_run"] == "ac2"
    assert written["rig_frames"]["ac2"]["image"] == SHARE + r"\Raw\AC2\DSC2.ARW"
    assert saved_rig_frames(project)["ac2"].image == photo


def test_image_selections_store_outside_paths_by_their_shared_spelling(share):
    root = share / "FF" / "DarSIA"
    assert _stored(str(root / "image_selection" / "a.json"), root) == "image_selection/a.json"
    assert _stored(str(share / "Raw" / "DSC1.ARW"), root) == SHARE + r"\Raw\DSC1.ARW"
    assert _stored("relative/x.json", root) == "relative/x.json"


def test_without_a_map_records_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    storage_paths.set_machine_path_map(())
    spelled = "X:/Raw/AC1/DSC1.ARW" if os.name == "nt" else "/srv/raw/AC1/DSC1.ARW"
    session = PaintingSession.default("ac1", spelled, None)
    assert session.to_dict()["image"]["path"] == spelled
    assert PaintingSession.from_dict(session.to_dict()).image_path == spelled
    frame = RigFrame(corners=dict(CORNERS), run_id="ac1", image=spelled)
    assert frame.to_dict()["image"] == spelled and RigFrame.from_dict(frame.to_dict()).image == spelled
    area = CropArea(corners=dict(CORNERS), run_id="ac1", image=spelled)
    assert area.to_dict()["image"] == spelled and CropArea.from_dict(area.to_dict()).image == spelled
    link = {"series_root": SHARE + r"\FF\DarSIA", "reference_run": "ac1"}
    assert native_series_link(_Project(link)) == link
    outside = str(tmp_path / "Raw" / "DSC1.ARW")
    assert _stored(outside, tmp_path / "FF") == outside
