"""Explicit saved-only export for genuine N1 mass-only coarse products.

This separate entrypoint never calls the native-phase reader or a numerical
producer. PNG sequences retain the explicit-fps, encoding-pending contract.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from . import storage_saved_export as export
from . import storage_saved_products as saved
from . import storage_saved_rendering as rendering
from . import storage_saved_tables as tables
from . import storage_n1_saved_products as n1
from .storage_indexed_products import indexed_asset_resolver
from .pipeline import products_publication as publication


def _preview(destination, stored, field, limit):
    """Use the unchanged display composition and dispose its buffers per frame."""
    destination = Path(destination)
    receipt = destination.with_name(destination.name + ".receipt.json")
    if destination.exists() or receipt.exists():
        raise FileExistsError("N1 saved previews are immutable")
    figure, image = rendering._figure(stored, field, limit)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=120)
            import os
            stream.flush()
            os.fsync(stream.fileno())
        pending = field.startswith("before_rules_") and stored["before_rules"]["state"] == "pending"
        value = dict(format="darsia-saved-n1-coarse-render-receipt-v1", complete=True,
            producer_kind="n1-mass-only", run=stored["run"], case_id=stored["case_id"],
            precision=stored["precision"], field=field, field_state="pending" if pending else "complete",
            fixed_limits=[0., limit], grid=stored["grid"],
            axes="native pixels, upper origin" if field == "before_rules_native_pixels" else "metric cells, y increasing",
            array_transposed=False, array_regridded=False, per_frame_normalization=False,
            photograph=dict(state="missing", reason="preview_only; original-left film is a separate explicit selection"),
            storage_receipt=stored["storage_receipt"], loaded_assets=stored["loaded_assets"],
            stored_mass_totals_g=stored["mass_totals_g"], preview=rendering._asset(destination))
        if stored.get("analysis_scope") is not None:
            value.update(analysis_scope=stored["analysis_scope"],
                quantified_stored_mass_totals_g=stored["quantified_mass_totals_g"], unknown_mass_g=None)
        return dict(preview=value["preview"], receipt=export._write_json(receipt, value), field_state=value["field_state"])
    finally:
        rendering.release_stored_figure(figure)


def export_indexed_n1_storage(index, destination, *, current_machine_id, collections=(), source_paths=None,
                              table=False, figure_field=None, film_field=None,
                              concentration_limit_g_l=None, fps=None, geometry_producer_sha256=None):
    """Export one streamed frame at a time through exact index/collect bindings."""
    if not export._options(table, figure_field, film_field, concentration_limit_g_l, fps):
        return dict(state="not_requested", outputs=[], default_native_unchanged=True)
    if film_field is not None and re.fullmatch(r"[0-9a-f]{64}", str(geometry_producer_sha256)) is None:
        raise ValueError("N1 film frames require the explicit qualified geometry producer SHA")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("N1 saved exports require one fresh output directory")
    resolve = indexed_asset_resolver(index, current_machine_id=current_machine_id,
                                     collections=tuple(collections), source_paths=source_paths)
    completion = saved._json(publication.read_published_metadata(index))
    run = n1.read_n1_stored_run(completion, resolve_asset=resolve,
        load_before="before_rules_native_pixels" in {film_field, figure_field}, stream=True)
    destination.mkdir(parents=True, exist_ok=False)
    result = dict(format="darsia-saved-n1-storage-export-v1", complete=False, state="streaming",
        producer_kind="n1-mass-only", run=run["run"], precision=run["precision"], products_index=deepcopy(index),
        ordered_case_ids=run["ordered_case_ids"], table=None, figures=[], film_frames=None,
        no_numerical_producer_or_implicit_collect=True, phase2_N1_contract_admitted=True)
    blocked, frames, scalar_views = {}, [], []
    limit = rendering._limit(film_field, concentration_limit_g_l if film_field == "concentration" else None) if film_field else None
    for number, stored in enumerate(run["frames"]):
        case = stored["case_id"]
        if table:
            scalar_views.append(export._scalar_view(stored))
        if film_field:
            if film_field.startswith("before_rules_") and stored["before_rules"]["state"] == "pending":
                blocked[case] = dict(state="not_film_ready", reason="saved_before_rules_pending")
            elif stored["photograph"]["state"] != "complete":
                blocked[case] = dict(state="not_film_ready", reason=stored["photograph"]["reason"])
            else:
                try:
                    rgba, photo = n1.read_n1_original_photo(stored,
                        resolve_asset=resolve, geometry_producer_sha256=geometry_producer_sha256)
                except saved.StoredProductCollectionRequired as error:
                    blocked[case] = dict(state="not_film_ready", reason="explicit_photo_collect_required",
                                         descriptor=deepcopy(error.descriptor))
                except FileNotFoundError:
                    blocked[case] = dict(state="not_film_ready", reason="bound_photo_unavailable")
                else:
                    item = export._film_frame(destination / "frames" / ("frame_%06d.png" % number),
                        stored, rgba, photo, field=film_field, limit=limit)
                    if frames and (item["width_px"], item["height_px"]) != (frames[0]["width_px"], frames[0]["height_px"]):
                        raise ValueError("N1 film frame geometry changed within one sequence")
                    frames.append(item)
                    del rgba, photo
        if figure_field:
            result["figures"].append(_preview(destination / "figures" / ("figure_%06d.png" % number),
                stored, figure_field, rendering._limit(figure_field,
                    concentration_limit_g_l if figure_field == "concentration" else None)))
        del stored
    if table:
        path, receipt = tables._destination(destination / "mass.csv")
        result["table"] = tables._write(path, receipt, scalar_views, manifest_assets=run["loaded_manifest_assets"])
    if film_field:
        sequence = dict(format="darsia-saved-original-left-frame-sequence-v1",
            producer_kind="n1-mass-only", state="not_film_ready" if blocked else "frames_ready_encoding_pending",
            fps=float(fps), timing="one frame per acquisition at explicit fps; acquisition intervals not inferred",
            field=film_field, fixed_limits=[0., limit], ordered_case_ids=run["ordered_case_ids"], frames=frames,
            blocked=blocked, original_left_frame_sequence_ready=not blocked, movie_encoded=False,
            movie_encoding="pending: no qualified explicit-fps encoder connected")
        result["film_frames"] = dict(sequence=sequence, receipt=export._write_json(destination / "film_sequence.json", sequence))
    result.update(complete=not blocked, state="not_film_ready" if blocked else "requested_exports_complete")
    return dict(result, receipt=export._write_json(destination / "export.json", result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--index-size-bytes", required=True, type=int)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bindings")
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
            raise ValueError("Explicit N1 export bindings exceed the metadata limit")
        bindings = saved._json(raw)
        if not isinstance(bindings, dict) or not bindings.keys() <= {"collections", "source_paths"}:
            raise ValueError("Unknown explicit N1 export binding")
    result = export_indexed_n1_storage(dict(path=args.index, sha256=args.index_sha256, size_bytes=args.index_size_bytes),
        args.output, current_machine_id=args.machine, table=args.table, figure_field=args.figure_field,
        film_field=args.film_field, concentration_limit_g_l=args.concentration_limit_g_l, fps=args.fps,
        geometry_producer_sha256=args.geometry_producer_sha256, **bindings)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
