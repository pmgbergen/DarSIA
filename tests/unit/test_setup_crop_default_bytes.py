"""The default Setup crop (same as the reference quad) keeps today's bytes.

The golden values below were recorded with the code before a separate crop
area existed (commit 9cb67fd8). With the default choice, the native Setup
export, the run crop configurations that "Apply rig corners to Setup" writes,
the Build crop correction and the rig frame read back from a run file must
stay identical to them. Only the temporary folder (with its path separators),
line endings and the creation time are normalised; the compressed label
archive embeds its write time, so its file hash is checked through the
decoded labels instead.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import pytest
import toml

import darsia
from darsia.fluidflower import setup_geometry_apply as apply
from darsia.fluidflower.camera_registration import CameraRegistration
from darsia.fluidflower.geometry_facies import GEOMETRY_REVISION_FORMAT
from darsia.fluidflower.series_geometry import (
    RigFrame,
    SeriesRunInputs,
    create_series_geometry_project,
    export_native_setup,
    rig_frame_from_inputs,
    save_injection_points,
    save_rig_frame,
    series_run_inputs,
)
from darsia.presets.workflows.config.corrections import CurvatureCorrectionConfig

CORNERS = {
    "top_left": (3.25, 4.25),
    "bottom_left": (44.25, 6.25),
    "bottom_right": (43.25, 73.25),
    "top_right": (4.25, 72.25),
}

# Recorded with commit 9cb67fd8, before the separate crop area existed.
GOLDEN = {
    "export": {
        "export_json": "85e2ba3e31ccd1d3d94a48266f13a01356615f030cb52c9fb63715fc01f6011d",
        "native_setup_toml": "864673ccee4a9487bbcc604e3b6327649e30c3278dbfe73949f2ab1554fc9757",
        "colored_labels": "1a826247a0f717a18890b8128dafbc1d44beb6f135745120ff6f017ad1be1329",
        "native_labels": "a7f5464b7614c43796262cc436a0e6e2a0371c0fb0f2a25538b32b213f06e42a/uint8",
        "native_labels_metadata": '{"dimensions": [0.49, 0.9], "indexing": "ij", '
        '"origin": [0.0, 0.49], "space_dim": 2}',
        "facies_properties": "7d2c68f434a62e0772d83a12c4bc9be79890c41c6fd825e5233c5c9f903e1a01",
    },
    "apply": {
        "changes": [
            [
                "series/config/runs/ac1.toml",
                "ac1",
                "cb0dae12eba09b87d124e93090f56ebc47c8377b641c43ba6917a1a389b0a40e",
            ],
            [
                "series/config/merged/ac1.toml",
                "ac1",
                "dc34e10cbc507f456473e49ad7a74bdcc10733e9fc4978ddc30c6e32507a90d6",
            ],
            [
                "series/config/runs/ac2.toml",
                "ac2",
                "246916b77346eeec2b3f8b42d0b51bc465077da3706f51aac26e1cced904fd36",
            ],
            [
                "series/config/merged/ac2.toml",
                "ac2",
                "a336d3fc72764992b9d27dd0830ca19a2cf6b648a6d10bcb0b278803204ff427",
            ],
        ],
        "runs": "9d377bd20516a69bca7abf05a49e58964025d43512176bd485724d5abadd22cc",
    },
    "build_crop": {
        "settings": '{"crop": {"bottom_left": [44, 6], "bottom_right": [43, 73], '
        '"height": 0.49, "in_meters": true, "top_left": [3, 4], "top_right": [4, 72], '
        '"width": 0.9}}',
        "engine_crop": '{"keys": ["pts_src", "width", "height", "in_meters"], '
        '"pts_src": ["VoxelArray", [[3, 4], [44, 6], [43, 73], [4, 72]]], '
        '"values": {"height": 0.49, "in_meters": true, "width": 0.9}}',
        "metadata": '{"dimensions": [0.49, 0.9], "origin": [0.0, 0.49], "shape": [43, 80]}',
        "pixels": "6a8447e29d249192b9d5e61134827e05bb4126bf1319e51180504786d92dc455",
    },
    "frame_from_run_crop": {
        "frame": '{"corners": {"bottom_left": [44.0, 6.0], "bottom_right": [43.0, 73.0], '
        '"top_left": [3.0, 4.0], "top_right": [4.0, 72.0]}, "height_m": 0.49, '
        '"image": "ac1.png", "image_size": null, "run_id": "ac1", '
        '"source": "series configuration", "width_m": 0.9}'
    },
}


def _sha(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalised(text: str, folder: Path) -> str:
    """Replace the temporary folder, its path separators and line endings.

    The writers spell the folder raw, JSON- or TOML-escaped; the rest of each
    path is compared with "/" so that the values hold on every platform.
    """
    raw = str(folder)
    for form in (raw.replace("\\", "\\\\"), raw, raw.replace("\\", "/")):
        text = text.replace(form, "<TMP>")
    text = re.sub(
        r"<TMP>[^\"'\s,\]]*",
        lambda match: match.group(0).replace("\\\\", "/").replace("\\", "/"),
        text,
    )
    return text.replace("\r\n", "\n")


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml.dumps(data), encoding="utf-8")


def _series(folder: Path):
    root = folder / "series"
    images = folder / "photographs"
    images.mkdir(parents=True)
    for name, colour in (("ac1", (50, 60, 70)), ("ac2", (52, 61, 69))):
        Image.new("RGB", (80, 50), colour).save(images / f"{name}.png")
    _write(
        root / "series.toml",
        {
            "schema_version": 1,
            "series_id": "local",
            "paths": {
                "common_config": "config/common.toml",
                "run_configs": "config/runs",
                "merged_configs": "config/merged",
            },
            "runs": {
                "ids": ["ac1", "ac2"],
                "config_files": {
                    "ac1": "config/runs/ac1.toml",
                    "ac2": "config/runs/ac2.toml",
                },
            },
        },
    )
    common = {
        "rig": {"dim": 2, "width": 0.9, "height": 0.49},
        "corrections": {
            "active": ["type", "resize"],
            "curvature": {"active": ["init"], "init": {"horizontal_bulge": 0.2}},
        },
    }
    _write(root / "config/common.toml", common)
    for run in ("ac1", "ac2"):
        run_data = {
            "data": {
                "folders": [str(images)],
                "baseline": f"{run}.png",
                "results": str(root / "results" / run),
            }
        }
        _write(root / f"config/runs/{run}.toml", run_data)
        merged = deepcopy(common)
        merged.update(run_data)
        _write(root / f"config/merged/{run}.toml", merged)
    return root, images


def _geometry_export(folder: Path, *, prepare_project=None):
    """A saved raw-frame revision with a marked rig frame, exported for Setup."""
    root, images = _series(folder)
    reference = next(row for row in series_run_inputs(root) if row.run_id == "ac1")
    project = create_series_geometry_project(
        series_id="local", series_name="Local", series_root=root, reference=reference
    )
    labels = np.ones((50, 80), dtype=np.int32)
    labels[25:, :] = 2
    labels[:, 60:] = 3
    labels_path = folder / "labels.npz"
    np.savez_compressed(labels_path, array=labels)
    properties = folder / "facies_properties.csv"
    properties.write_bytes(b"id,name\n1,Upper sand\n2,Lower sand\n")
    current = {
        "format": GEOMETRY_REVISION_FORMAT,
        "revision_id": "revision-1",
        "recorded_utc": "2026-10-27T12:00:00+00:00",
        "recorded_by": "reviewer",
        "configuration": {
            "labels_path": str(labels_path),
            "regions": [
                {"label_id": 1, "facies_id": 1},
                {"label_id": 2, "facies_id": 2},
                {"label_id": 3, "facies_id": 2},
            ],
            "grouping_mode": "shared_facies",
            "width_m": 0.9,
            "height_m": 0.49,
        },
        "artifacts": {"facies_properties_path": str(properties)},
    }
    current_path = project.root / "overrides" / "geometry_facies" / "current.json"
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(json.dumps(current), encoding="utf-8")
    save_rig_frame(
        project,
        RigFrame(
            corners=dict(CORNERS),
            width_m=0.9,
            height_m=0.49,
            run_id="ac1",
            image=str(images / "ac1.png"),
            image_size=(80, 50),
        ),
    )
    save_injection_points(
        project,
        [
            {
                "location_id": "ip1",
                "name": "Port",
                "pixel": {"run_id": "ac1", "row": 30.0, "col": 40.0},
            }
        ],
    )
    if prepare_project is not None:
        prepare_project(project, images)
    return root, images, project, export_native_setup(project, reference=reference)


def _registration(source, target, corners):
    return CameraRegistration(
        np.array([[1.0, 0.0, -3.5], [0.0, 1.0, 2.5], [0.0, 0.0, 1.0]]),
        "affine",
        30,
        30,
        0.3,
        0.5,
        (80, 50),
        (80, 50),
        {"top": 10, "left": 10, "right": 10},
    )


def _export_values(folder: Path, export) -> dict:
    text = (export.destination / "export.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert text == json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    # Today's digest of the export inputs, written out as its formula.
    expected_digest = hashlib.sha256(
        json.dumps(
            {
                "reference_run": manifest["reference_run"],
                "frame": manifest["rig_frame"],
                "points": manifest["injection_points"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["inputs_digest"] == expected_digest
    manifest["inputs_digest"] = "<DIGEST>"
    manifest["created_utc"] = "<TIME>"
    for row in manifest["files"].values():
        row["sha256"] = "<SHA>"
    with np.load(export.native_labels, allow_pickle=True) as archive:
        native_labels = archive["array"]
        native_metadata = archive["metadata"].item()
    setup_text = export.setup_toml.read_text(encoding="utf-8")
    return {
        "export_json": _sha(
            _normalised(json.dumps(manifest, indent=2, ensure_ascii=False), folder)
        ),
        "native_setup_toml": _sha(_normalised(setup_text, folder)),
        "colored_labels": _sha(np.asarray(Image.open(export.colored_image)).tobytes()),
        "native_labels": _sha(native_labels.tobytes()) + f"/{native_labels.dtype}",
        "native_labels_metadata": json.dumps(native_metadata, sort_keys=True),
        "facies_properties": _sha(export.facies_properties.read_bytes()),
    }


def _plan_values(folder: Path, root: Path, export) -> dict:
    plan = apply.prepare_crop_plan(
        root,
        export.destination / "export.json",
        expected_revision_id="revision-1",
        register=_registration,
    )
    return {
        "changes": [
            [
                change.path.relative_to(folder).as_posix(),
                change.run_id,
                _sha(_normalised(change.after.decode("utf-8"), folder)),
            ]
            for change in plan.changes
        ],
        "runs": _sha(_normalised(json.dumps(list(plan.runs), sort_keys=True), folder)),
    }


def _crop_correction_values() -> dict:
    section = {
        "active": ["crop"],
        "crop": {
            "top_left": [3, 4],
            "bottom_left": [44, 6],
            "bottom_right": [43, 73],
            "top_right": [4, 72],
            "width": 0.9,
            "height": 0.49,
            "in_meters": True,
        },
    }
    settings = CurvatureCorrectionConfig().load(section).to_dict()
    correction = darsia.CurvatureCorrection(config=settings)
    image = darsia.ScalarImage(
        np.indices((50, 80)).sum(axis=0).astype(np.float32) / 128.0,
        dimensions=[0.5, 0.8],
    )
    corrected = correction(image)
    crop = correction.config["crop"]
    return {
        "settings": json.dumps(settings, sort_keys=True),
        "engine_crop": json.dumps(
            {
                "keys": list(crop),
                "pts_src": [type(crop["pts_src"]).__name__, crop["pts_src"].tolist()],
                "values": {key: crop[key] for key in crop if key != "pts_src"},
            },
            sort_keys=True,
        ),
        "metadata": json.dumps(
            {
                "dimensions": [float(value) for value in corrected.dimensions],
                "origin": [float(value) for value in corrected.origin],
                "shape": list(corrected.img.shape),
            }
        ),
        "pixels": _sha(np.ascontiguousarray(corrected.img).tobytes()),
    }


def _frame_from_run_crop_values() -> dict:
    row = SeriesRunInputs(
        run_id="ac1",
        config_path=Path("ac1.toml"),
        baseline_image=Path("ac1.png"),
        width_m=0.9,
        height_m=0.49,
        crop={
            "top_left": [3, 4],
            "bottom_left": [44, 6],
            "bottom_right": [43, 73],
            "top_right": [4, 72],
            "width": 0.9,
            "height": 0.49,
            "in_meters": True,
        },
    )
    return {"frame": json.dumps(rig_frame_from_inputs(row).to_dict(), sort_keys=True)}


def default_outputs(folder: Path) -> dict:
    """Every compared default output of one fresh fixture."""
    root, _images, _project, export = _geometry_export(folder)
    values = {"export": _export_values(folder, export)}
    values["apply"] = _plan_values(folder, root, export)
    values["build_crop"] = _crop_correction_values()
    values["frame_from_run_crop"] = _frame_from_run_crop_values()
    return values


@pytest.fixture
def no_aliases(tmp_path, monkeypatch):
    path = tmp_path / "user" / "series_registry.toml"
    path.parent.mkdir()
    path.write_text("schema_version = 1\n", encoding="utf-8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(path))


def test_default_crop_outputs_are_byte_identical_to_the_recorded_code(
    tmp_path, no_aliases
):
    folder = tmp_path / "fixture"
    folder.mkdir()
    assert default_outputs(folder) == GOLDEN
