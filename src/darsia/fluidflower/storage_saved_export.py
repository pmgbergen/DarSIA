"""Explicit saved-storage exports; no producer, implicit collect, or movie encoder.

Only complete native-phase-n1 products are admitted by the existing reader.
A phase2/N1-only completion needs its own qualified saved-output contract.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections.abc import Mapping
import io
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from . import storage_indexed_products as indexed
from . import storage_original_photo as originals
from . import storage_saved_products as saved
from . import storage_saved_rendering as rendering
from . import storage_saved_tables as tables
from .pipeline import products_publication as publication


def _write_json(path, value):
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, allow_nan=False,
                                 sort_keys=True, indent=2) + "\n").encode("utf8"))
        stream.flush()
        os.fsync(stream.fileno())
    return rendering._asset(path)


def _options(table, figure_field, film_field, concentration_limit_g_l, fps):
    if type(table) is not bool:
        raise ValueError("Table selection must be explicit true or false")
    requested = table or figure_field is not None or film_field is not None
    fields = [field for field in (figure_field, film_field) if field is not None]
    if concentration_limit_g_l is not None and "concentration" not in fields:
        raise ValueError("A concentration limit requires an explicit concentration export")
    for field in fields:
        rendering._limit(field, concentration_limit_g_l if field == "concentration" else None)
    if film_field is not None:
        if (isinstance(fps, bool) or not isinstance(fps, (int, float))
                or not math.isfinite(fps) or fps <= 0):
            raise ValueError("Film frames need an explicit finite positive fps")
    elif fps is not None:
        raise ValueError("fps applies only to an explicit film-frame sequence")
    return requested


def _photograph(bound, stored, *, resolve_asset, geometry_producer_sha256):
    """Validate a producer's saved pre-light PNG; never follow its source links."""
    loaded = []
    document = saved._json(saved._read_bytes(bound, resolve_asset, loaded))
    if (document.get("format") != "darsia-stored-registered-original-photo-v1"
            or document.get("complete") is not True
            or document.get("run") != stored["run"]
            or document.get("case_id") != stored["case_id"]
            or document.get("precision") != stored["precision"]
            or document.get("source") != stored["source"]
            or document.get("camera_source", {}).get("sha256") != stored["source"]["source_sha256"]
            or document.get("geometry_producer_sha256") != geometry_producer_sha256
            or document.get("colour_light_or_v5_changes") is not False
            or document.get("camera_RAW_or_correction_chain_executed") is not False
            or document.get("current_phase_products_changed") is not False
            or document.get("display_dtype") != "uint8"
            or document.get("alpha_meaning") != originals._ALPHA
            or document.get("alpha_inside_ROI") != 255 or document.get("alpha_outside_ROI") != 0):
        raise ValueError("Saved original photograph identity/pre-light contract differs")
    data = saved._read_bytes(document["photograph"], resolve_asset, loaded)
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.mode != "RGBA":
            raise ValueError("Saved original photograph must remain literal RGBA PNG")
        rgba = np.asarray(image).copy()
    shape = list(stored["grid"]["native_shape"])
    box = document.get("stored_photograph_ROI_xywh")
    if (document.get("target_canvas_hw") != shape or list(rgba.shape) != [*shape, 4]
            or not isinstance(box, list) or len(box) != 4
            or any(type(value) is not int for value in box)):
        raise ValueError("Saved original photograph canvas/ROI differs")
    x, y, width, height = box
    if (min(x, y) < 0 or min(width, height) < 1
            or x + width > shape[1] or y + height > shape[0]
            or document.get("current_native_origin_xy") != [-x, -y]):
        raise ValueError("Saved original photograph placement differs")
    coverage = np.zeros(tuple(shape), dtype=bool)
    coverage[y:y + height, x:x + width] = True
    roi = rgba[y:y + height, x:x + width, :3]
    if (not np.array_equal(rgba[..., 3], coverage.astype(np.uint8) * 255)
            or np.any(rgba[~coverage, :3])
            or originals._pixels(rgba) != document.get("RGBA_pixel_identity")
            or originals._pixels(roi) != document.get("known_ROI_pixel_identity")
            or originals._pixels(roi) != document.get("copied_ROI_pixel_identity")
            or document.get("full_canvas_photograph_coverage") != bool(coverage.all())):
        raise ValueError("Saved original photograph pixels/coverage differ")
    rgba.setflags(write=False)
    return rgba, dict(receipt=deepcopy(bound), photograph=deepcopy(document["photograph"]),
        loaded_assets=loaded, full_canvas_photograph_coverage=bool(coverage.all()),
        alpha_meaning=document["alpha_meaning"], geometry_producer_sha256=geometry_producer_sha256)


def _film_frame(destination, stored, rgba, photo, *, field, limit):
    """Reuse the qualified metric-field figure; replace only its missing left pane."""
    figure, image = rendering._figure(stored, field, limit)
    try:
        if image is None:
            raise ValueError("A pending saved field cannot become a film-ready frame")
        left = figure.axes[0]
        left.clear()
        left.set_facecolor("white")
        left.imshow(rgba, origin="upper", interpolation="nearest")
        left.set_title("Original registered photograph - before light correction")
        left.set_axis_off()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=120, facecolor="white")
            stream.flush()
            os.fsync(stream.fileno())
        with Image.open(destination) as encoded:
            width, height = encoded.size
        document = dict(format="darsia-saved-original-left-frame-v1", complete=True,
            run=stored["run"], case_id=stored["case_id"], precision=stored["precision"],
            field=field, fixed_limits=[0., limit], photograph=photo,
            original_photo_RGB_or_alpha_changed=False, display_resize_only=True,
            light_correction_or_registration_executed=False,
            coarse_field_grid=stored["grid"], field_array_transposed_or_regridded=False,
            storage_receipt=stored["storage_receipt"], loaded_storage_assets=stored["loaded_assets"],
            source=stored["source"], width_px=width, height_px=height,
            frame=rendering._asset(destination))
        if stored.get("analysis_scope") is not None:
            document.update(analysis_scope=stored["analysis_scope"],
                quantified_stored_mass_totals_g=stored["quantified_mass_totals_g"], unknown_mass_g=None)
        receipt = _write_json(destination.with_name(destination.name + ".receipt.json"), document)
        return dict(frame=document["frame"], receipt=receipt, case_id=stored["case_id"],
                    width_px=width, height_px=height)
    finally:
        rendering.release_stored_figure(figure)


def _scalar_view(stored):
    """Only the literal small fields consumed by the unchanged CSV writer."""
    document = stored["storage_document"]
    value = dict(run=stored["run"], case_id=stored["case_id"], precision=stored["precision"],
        source=deepcopy(stored["source"]), mass_totals_g=deepcopy(stored["mass_totals_g"]),
        native_scalar_reference={key: stored["native_scalar_reference"][key]
                                 for _component, key in tables._COMPONENTS},
        storage_document={key: deepcopy(document[key])
                          for key in ("configuration", "code_commit", "calibration_sha256")},
        before_rules=deepcopy(stored["before_rules"]),
        storage_receipt=deepcopy(stored["storage_receipt"]),
        loaded_assets=deepcopy(stored["loaded_assets"]))
    if stored.get("analysis_scope") is not None:
        value.update(analysis_scope=deepcopy(stored["analysis_scope"]),
            quantified_mass_totals_g=deepcopy(stored["quantified_mass_totals_g"]), unknown_mass_g=None)
        value["native_scalar_reference"].update({"quantified_" + key:
            stored["native_scalar_reference"]["quantified_" + key] for _, key in tables._COMPONENTS})
    return value


def export_indexed_storage(index, destination, *, current_machine_id, collections=(),
                           source_paths=None, table=False, figure_field=None, film_field=None,
                           concentration_limit_g_l=None, fps=None, photographs=None,
                           geometry_producer_sha256=None):
    """Export new files from one explicit immutable products-index closure.

    ``photographs`` maps case IDs to indexed original-photo receipt descriptors.
    The original producer must already have published/indexed the PNG and receipt.
    Nothing here makes those products, searches for them, or collects them.
    Without export selections this is an IO-free no-op. Encoded movies are pending.
    """
    if not _options(table, figure_field, film_field, concentration_limit_g_l, fps):
        return dict(state="not_requested", outputs=[], default_native_unchanged=True)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Saved exports require one fresh output directory")
    if film_field is not None and re.fullmatch(r"[0-9a-f]{64}", str(geometry_producer_sha256)) is None:
        raise ValueError("Film frames require the explicit qualified geometry producer SHA")
    if photographs is not None and not isinstance(photographs, Mapping):
        raise ValueError("Original photographs must be an explicit case-to-receipt mapping")
    collections = tuple(collections)
    resolve = indexed.indexed_asset_resolver(index, current_machine_id=current_machine_id,
        collections=collections, source_paths=source_paths)
    completion = saved._json(publication.read_published_metadata(index))
    run = saved.read_stored_run(completion, resolve_asset=resolve,
        load_before=film_field == "before_rules_native_pixels", stream=True)
    photos = dict(photographs or {})
    if not photos.keys() <= set(run["ordered_case_ids"]):
        raise ValueError("Original photographs must name only this run's exact case IDs")
    blocked, frames, scalar_views = {}, [], []
    destination.mkdir(parents=True, exist_ok=False)
    result = dict(format="darsia-saved-storage-export-v1", complete=False,
        state="streaming", run=run["run"], precision=run["precision"],
        products_index=deepcopy(index), ordered_case_ids=run["ordered_case_ids"],
        table=None, figures=[], film_frames=None,
        no_numerical_producer_or_implicit_collect=True, phase2_N1_contract_admitted=False)
    limit = (rendering._limit(film_field,
        concentration_limit_g_l if film_field == "concentration" else None)
        if film_field is not None else None)
    number = 0
    for stored in run["frames"]:
        case = stored["case_id"]
        if table:
            scalar_views.append(_scalar_view(stored))
        if film_field is not None:
            if film_field.startswith("before_rules_") and stored["before_rules"]["state"] == "pending":
                blocked[case] = dict(state="not_film_ready", reason="saved_before_rules_pending")
            elif case not in photos:
                blocked[case] = dict(state="not_film_ready", reason="indexed_registered_pre_light_photo_not_bound")
            else:
                try:
                    rgba, photo = _photograph(photos[case], stored, resolve_asset=resolve,
                        geometry_producer_sha256=geometry_producer_sha256)
                except saved.StoredProductCollectionRequired as error:
                    blocked[case] = dict(state="not_film_ready", reason="explicit_photo_collect_required",
                                         descriptor=deepcopy(error.descriptor))
                except FileNotFoundError:
                    blocked[case] = dict(state="not_film_ready", reason="bound_photo_unavailable")
                else:
                    frame = _film_frame(destination / "frames" / ("frame_%06d.png" % number),
                        stored, rgba, photo, field=film_field, limit=limit)
                    if frames and (frame["width_px"], frame["height_px"]) != (frames[0]["width_px"], frames[0]["height_px"]):
                        raise ValueError("Film frame geometry changed within one sequence")
                    frames.append(frame)
                    del rgba, photo
        # Neither the generator nor this caller keeps the previous coarse arrays
        # or native BEFORE image alive while the next acquisition is read.
        del stored
        if figure_field is not None:
            # The unchanged preview writer rereads this already admitted record.
            # Release the iterator's arrays first: no overlapping coarse copies.
            result["figures"].append(rendering.render_stored_frame(
                destination / "figures" / ("figure_%06d.png" % number),
                run["original_completion"]["records"][number], resolve_asset=resolve,
                field=figure_field,
                concentration_limit_g_l=concentration_limit_g_l if figure_field == "concentration" else None))
        number += 1
    if table:
        csv_path, csv_receipt = tables._destination(destination / "mass.csv")
        result["table"] = tables._write(csv_path, csv_receipt, scalar_views,
            manifest_assets=run["loaded_manifest_assets"])
    if film_field is not None:
        sequence = dict(format="darsia-saved-original-left-frame-sequence-v1",
            state="not_film_ready" if blocked else "frames_ready_encoding_pending", fps=float(fps),
            timing="one frame per acquisition at explicit fps; acquisition intervals not inferred",
            field=film_field, fixed_limits=[0., limit], ordered_case_ids=run["ordered_case_ids"],
            frames=frames, blocked=blocked, original_left_frame_sequence_ready=not blocked,
            movie_encoded=False, movie_encoding="pending: no qualified explicit-fps encoder connected")
        result["film_frames"] = dict(sequence=sequence,
            receipt=_write_json(destination / "film_sequence.json", sequence))
    result.update(complete=not blocked,
                  state="not_film_ready" if blocked else "requested_exports_complete")
    receipt = _write_json(destination / "export.json", result)
    return dict(result, receipt=receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--index-size-bytes", required=True, type=int)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bindings", help="Explicit JSON collections/source_paths/photographs; never a search root")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--figure-field", choices=sorted(rendering.FIELDS))
    parser.add_argument("--film-field", choices=sorted(rendering.FIELDS))
    parser.add_argument("--concentration-limit-g-l", type=float)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--geometry-producer-sha256")
    args = parser.parse_args(argv)
    bindings = {}
    if args.bindings:
        with Path(args.bindings).open("rb") as stream:
            raw = stream.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("Explicit export bindings exceed the metadata limit")
        bindings = saved._json(raw)
        if not isinstance(bindings, dict) or not bindings.keys() <= {"collections", "source_paths", "photographs"}:
            raise ValueError("Unknown explicit export binding")
    result = export_indexed_storage(dict(path=args.index, sha256=args.index_sha256,
        size_bytes=args.index_size_bytes), args.output, current_machine_id=args.machine,
        table=args.table, figure_field=args.figure_field, film_field=args.film_field,
        concentration_limit_g_l=args.concentration_limit_g_l, fps=args.fps,
        geometry_producer_sha256=args.geometry_producer_sha256, **bindings)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
