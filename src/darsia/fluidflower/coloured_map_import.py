"""UI import of a native coloured map on the raw reference photograph grid.

This is a file-input adapter to the original palette-labeling workflow. It is
not an automatic sand detector and never runs the painted-layer classifier.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import uuid

import numpy as np
from PIL import Image
import toml

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.series_geometry import RigFrame, SeriesRunInputs, photo_size
from darsia.presets.workflows.setup.setup_labeling import segment_colored_image
from darsia.utils.segmentation import label_image


@dataclass(frozen=True)
class ImportedColouredMap:
    session: lp.PaintingSession
    proposal: lp.LayerProposal
    receipt: Path
    work_directory: Path
    native_labels: Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)


def _editable_polygons(classes: np.ndarray, layer_ids: list[int]) -> list[lp.Polygon]:
    """Represent regions without simplification; verify with native rasterizer."""
    import cv2

    polygons = []
    for layer_id in layer_ids:
        mask = np.ascontiguousarray(classes == layer_id, dtype=np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            points = [(float(y), float(x)) for x, y in contour.reshape(-1, 2)]
            polygon = lp.Polygon(layer_id, points)
            # Reuse the editor's actual representation limit, not a new area
            # tolerance. A degenerate component cannot become a draggable ring.
            if len(points) < 3 or polygon.area() < 1.0:
                raise ValueError(
                    "The imported map contains a component that the existing "
                    "polygon editor cannot represent exactly. The map was not changed."
                )
            polygons.append(polygon)
    rebuilt = lp.rasterize_polygons(polygons, classes.shape, (1.0, 1.0))
    if not np.array_equal(rebuilt, classes):
        raise ValueError(
            "The imported regions cannot be represented exactly by the existing "
            "editable polygons. No contour approximation or resize was applied."
        )
    return polygons


def imported_layer_state(
    labels: np.ndarray,
    reference: SeriesRunInputs,
    *,
    raw_size: tuple[int, int],
    frame: RigFrame | None,
) -> tuple[lp.PaintingSession, lp.LayerProposal]:
    """Prepare existing raw-grid editor objects without changing a project."""
    array = np.asarray(labels)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            "The original palette workflow did not produce a 2-D integer label map."
        )
    if tuple(array.shape) != (raw_size[1], raw_size[0]):
        raise ValueError(
            "The imported labels do not have the raw reference photograph's exact dimensions."
        )
    if frame is not None:
        if frame.run_id and frame.run_id != reference.run_id:
            raise ValueError("The rig frame belongs to another reference run.")
        if frame.image_size is not None and tuple(frame.image_size) != raw_size:
            raise ValueError("The rig frame belongs to another raw photograph size.")
        if frame.image and reference.baseline_image is not None:
            try:
                same = os.path.samefile(frame.image, reference.baseline_image)
            except OSError:
                same = os.path.normcase(
                    os.path.abspath(frame.image)
                ) == os.path.normcase(os.path.abspath(reference.baseline_image))
            if not same:
                raise ValueError(
                    "The rig frame was marked on another photograph; apply/register the corners to this raw baseline before importing its map."
                )
    inside = lp.frame_mask(frame, array.shape, (1.0, 1.0))
    native_ids = sorted(int(value) for value in np.unique(array[inside]))
    if not native_ids or len(native_ids) > 255:
        raise ValueError(
            "The existing layer editor needs between 1 and 255 imported layers."
        )
    classes = np.zeros(array.shape, dtype=np.uint8)
    session = lp.PaintingSession(
        run_id=reference.run_id,
        image_path=str(reference.baseline_image or ""),
        image_size=raw_size,
        image_stamp=lp.image_stamp(reference.baseline_image),
    )
    for layer_id, native_id in enumerate(native_ids, start=1):
        name = f"Imported layer {native_id}"
        session.layers.append(
            lp.PaintedLayer(
                layer_id,
                name,
                lp.LAYER_COLORS[(layer_id - 1) % len(lp.LAYER_COLORS)],
                name,
            )
        )
        # An imported colour does not establish physical facies properties.
        # The existing facies editor requires those before a revision is saved.
        session.facies[name] = lp.FaciesEntry()
        classes[inside & (array == native_id)] = layer_id
    session.polygons = _editable_polygons(
        classes, [layer.layer_id for layer in session.layers]
    )
    # Exact imported outlines are the initial editable geometry. The existing
    # Propose boundaries command remains the separate, unchanged refinement.
    session.snap_mm = 0.0
    summary = lp._region_counts(classes)
    proposal = lp.LayerProposal(
        classes=classes,
        confidence=np.full(classes.shape, 255, dtype=np.uint8),
        image_size=raw_size,
        digest=lp.proposal_digest(session, frame),
        created_utc=datetime.now(timezone.utc).isoformat(),
        frame=lp._frame_key(frame),
        layers=[
            {
                "layer_id": layer.layer_id,
                "name": layer.name,
                "pixels": summary[layer.layer_id][0],
                "regions": summary[layer.layer_id][1],
                "painted": True,
                "polygons": sum(p.layer_id == layer.layer_id for p in session.polygons),
            }
            for layer in session.layers
        ],
        polygon_count=len(session.polygons),
    )
    return session, proposal


def import_coloured_map(
    source_map: str | Path,
    reference: SeriesRunInputs,
    config: Mapping[str, Any],
    *,
    input_grid: str,
    work_root: str | Path,
    frame: RigFrame | None,
    segment: Callable[..., Any] | None = None,
) -> ImportedColouredMap:
    """Run the unchanged original workflow in a fresh, isolated import folder.

    The file dialog must explicitly explain that the input map belongs to the
    *raw reference photograph*. Corrected-grid maps are rejected: curvature's
    cached output-to-input coordinate samples are not a general inverse label
    correction API. Equal dimensions alone must not be used to guess a grid.
    Nothing is saved into the series or geometry revision by this function.
    """
    if input_grid != "raw":
        raise ValueError(
            "Import a coloured map on the raw reference photograph's grid. "
            "Corrected-grid maps are not resized or inverse-corrected by this import."
        )
    if reference.baseline_image is None:
        raise ValueError("Choose a reference run with a raw baseline photograph first.")
    baseline = Path(reference.baseline_image)
    raw_size = photo_size(baseline)
    baseline_sha = _sha(baseline.read_bytes())
    original = Path(source_map)
    source_bytes = original.read_bytes()
    with Image.open(io.BytesIO(source_bytes)) as image:
        if image.size != raw_size:
            raise ValueError(
                f"The coloured map is {image.width} x {image.height} px; the raw "
                f"reference photograph is {raw_size[0]} x {raw_size[1]} px. "
                "Import uses the raw grid exactly, without resizing."
            )
        if image.getexif().get(274, 1) != 1:
            raise ValueError(
                "The coloured map has an EXIF orientation transform; use an image saved on the raw pixel grid."
            )
    rig = deepcopy(dict(config.get("rig") or {}))
    if not all(key in rig for key in ("width", "height", "dim")):
        raise ValueError(
            "The selected run needs its existing [rig] width, height and dimension."
        )
    labels_config = deepcopy(dict(config.get("labeling") or {}))
    folder = Path(work_root) / f"coloured-map-import-{uuid.uuid4().hex}"
    folder.mkdir(parents=True, exist_ok=False)
    frozen_map = folder / f"input{original.suffix}"
    _write_new(frozen_map, source_bytes)
    labels_path = folder / "labels.npz"
    rig["path"] = str(folder / "unused-rig-path")
    labels_config.update(colored_image=str(frozen_map), labels=str(labels_path))
    # Only the sections required by the original labeling workflow are passed;
    # no unrelated run workflows or user output paths can be triggered.
    generated = toml.dumps({"rig": rig, "labeling": labels_config}).encode("utf-8")
    config_path = folder / "labeling.toml"
    _write_new(config_path, generated)
    workflow = segment_colored_image if segment is None else segment
    workflow(config_path, show=False)
    with np.load(labels_path, allow_pickle=False) as archive:
        if "array" not in archive.files:
            raise ValueError(
                "The original workflow's labels archive has no native 'array' field."
            )
        labels = np.asarray(archive["array"])
    session, proposal = imported_layer_state(
        labels, reference, raw_size=raw_size, frame=frame
    )
    if original.read_bytes() != source_bytes:
        raise RuntimeError(
            f"The coloured map changed while it was being imported: {original}"
        )
    if photo_size(baseline) != raw_size or _sha(baseline.read_bytes()) != baseline_sha:
        raise RuntimeError("The raw reference photograph changed during import.")
    workflow_file = Path(inspect.getsourcefile(segment_colored_image) or "")
    labeling_file = Path(inspect.getsourcefile(label_image) or "")
    receipt = {
        "format": "darsia-ui-coloured-map-import-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_map": str(original),
        "source_map_sha256": _sha(source_bytes),
        "frozen_input": str(frozen_map),
        "frozen_input_sha256": _sha(frozen_map.read_bytes()),
        "reference_run": reference.run_id,
        "raw_baseline": str(baseline),
        "raw_baseline_sha256": baseline_sha,
        "raw_size": list(raw_size),
        "input_grid": "raw",
        "resized": False,
        "inverse_correction_applied": False,
        "workflow": "darsia.presets.workflows.setup.setup_labeling.segment_colored_image",
        "workflow_source_sha256": _sha(workflow_file.read_bytes()),
        "palette_labeling_source_sha256": _sha(labeling_file.read_bytes()),
        "ui_import_adapter_source_sha256": _sha(Path(__file__).read_bytes()),
        "generated_config_sha256": _sha(generated),
        "native_labels": str(labels_path),
        "native_labels_sha256": _sha(labels_path.read_bytes()),
        "interpretation": "Imported original palette segmentation; no ExtraTrees fit was run.",
        "extra_trees_fit_run": False,
        "legacy_storage_method_tag": lp.PROPOSAL_METHOD,
        "storage_method_tag_is_retained_format_metadata": True,
        "editor_state": "raw-grid editable polygons, verified by native rasterizer",
        "initial_editor_session_sha256": _sha(
            json.dumps(session.to_dict(), ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ),
        "initial_editor_classes_sha256": _sha(proposal.classes.tobytes()),
        "initial_editor_classes_shape": list(proposal.classes.shape),
        "initial_editor_classes_dtype": str(proposal.classes.dtype),
        "facies_properties_inferred": False,
    }
    illustration = labels_path.with_suffix(".jpg")
    if illustration.is_file():
        receipt["native_illustration_sha256"] = _sha(illustration.read_bytes())
    receipt_path = folder / "import.json"
    _write_new(
        receipt_path,
        (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return ImportedColouredMap(session, proposal, receipt_path, folder, labels_path)
