"""Geometry export and corrected setup on another machine (local fixtures).

The export's rig-frame photograph is opened through the per-user storage
aliases, and the corrected-setup project check follows the configured
geometry package instead of a hard-wired ``<series>/geometry_facies``.
"""

import json
import shutil
import tomllib

import pytest
import toml

from darsia.fluidflower import corrected_setup as subject
from darsia.fluidflower import setup_geometry_apply as apply
from darsia.fluidflower.series_geometry import geometry_package_root
from .test_corrected_setup_canonical_layout import layout  # noqa: F401
from .test_corrected_setup_output_pointer import (  # noqa: F401
    Q121_private_output_binding_TESTONLY,
    _Q121_checked_TESTONLY,
    pointer,
)
from .test_setup_geometry_apply import _registration, _series


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / "user" / "series_registry.toml"
    path.parent.mkdir()
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(path))

    def write(aliases):
        lines = ["schema_version = 1", "[storage.path_aliases]"]
        lines += [
            f"{json.dumps(str(a))} = {json.dumps(str(b))}" for a, b in aliases.items()
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write({})
    return write


def _export_from_another_machine(tmp_path, export, images):
    """Move the photographs and keep the export's original absolute path."""
    elsewhere = tmp_path / "student" / "photographs"
    elsewhere.parent.mkdir()
    shutil.copytree(images, elsewhere)
    value = json.loads(export.read_bytes())
    value["rig_frame"]["image"] = str(images / "ac1.png")
    export.write_text(json.dumps(value), encoding="utf-8")
    shutil.rmtree(images)
    return elsewhere


def _point_runs_at(root, folder):
    for run in ("ac1", "ac2"):
        for kind in ("runs", "merged"):
            path = root / f"config/{kind}/{run}.toml"
            data = tomllib.loads(path.read_text())
            data["data"]["folders"] = [str(folder)]
            path.write_text(toml.dumps(data), encoding="utf-8")


def test_rig_frame_photo_of_the_export_is_opened_through_a_storage_alias(
    tmp_path, registry
):
    root, export, images, _ = _series(tmp_path)
    moved = _export_from_another_machine(tmp_path, export, images)
    _point_runs_at(root, moved)
    registry({images: moved})
    calls = []

    def register(*args):
        calls.append(args)
        return _registration(*args)

    plan = apply.prepare_crop_plan(
        root, export, expected_revision_id="revision-1", register=register
    )
    assert plan.runs[0]["registration"]["kind"] == "same_raw_photograph"
    assert [call[:2] for call in calls] == [(moved / "ac1.png", moved / "ac2.png")]
    assert {photo.path for photo in plan.photographs} == {
        moved / "ac1.png",
        moved / "ac2.png",
    }


def test_without_an_alias_the_moved_export_photo_is_still_missing(tmp_path, registry):
    root, export, images, _ = _series(tmp_path)
    moved = _export_from_another_machine(tmp_path, export, images)
    _point_runs_at(root, moved)
    with pytest.raises(FileNotFoundError):
        apply.prepare_crop_plan(
            root, export, expected_revision_id="revision-1", register=_registration
        )


def _configure_package(series, value):
    manifest_path = series / "series.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    manifest["schema_version"] = 1  # A real (versioned) series manifest.
    manifest.setdefault("paths", {})["geometry_facies"] = value
    manifest_path.write_text(toml.dumps(manifest), encoding="utf-8")


def test_corrected_setup_accepts_the_configured_geometry_package(pointer):  # noqa: F811
    data = pointer["layout"]
    series = data["series"]
    package = series / "packages" / "geometry"
    _configure_package(series, "packages/geometry")
    assert geometry_package_root(series) == package
    with pytest.raises(ValueError) as error:
        subject._configuration_bundle(
            series,
            "ac17",
            pointer["sources"]["ac17"],
            project_root=package,
            revision="a" * 32,
        )
    # Past the project check: the old binding's rig.path names the default package.
    assert "actual series project" not in str(error.value)
    assert "rig.path" in str(error.value)


def test_corrected_setup_refuses_the_default_folder_when_another_is_configured(
    pointer,  # noqa: F811
):
    data = pointer["layout"]
    series = data["series"]
    _configure_package(series, "packages/geometry")
    with pytest.raises(ValueError, match="actual series project and revision"):
        subject._configuration_bundle(
            series,
            "ac17",
            pointer["sources"]["ac17"],
            project_root=data["project"].root,
            revision="a" * 32,
        )


def test_corrected_setup_default_package_is_unchanged(pointer):  # noqa: F811
    data = pointer["layout"]
    series = data["series"]
    assert geometry_package_root(series) == series / "geometry_facies"
    bundle = subject._configuration_bundle(
        series,
        "ac17",
        pointer["sources"]["ac17"],
        project_root=data["project"].root,
        revision="a" * 32,
    )
    assert bundle["dual_configuration_binding"]["run_id"] == "ac17"


def test_private_corrected_binding_also_follows_the_configured_package(
    Q121_private_output_binding_TESTONLY,  # noqa: F811
):
    data = Q121_private_output_binding_TESTONLY
    assert _Q121_checked_TESTONLY(data) == data["value"]
    _configure_package(data["series"], "packages/geometry")
    with pytest.raises(ValueError, match="exact local project/run/revision"):
        _Q121_checked_TESTONLY(data)
