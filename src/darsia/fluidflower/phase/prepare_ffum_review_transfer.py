"""Prepare source-bound FF_UM annotation coordinates without changing photographs.

Run with the pinned historical DarSIA runtime. Only the explicitly staged UM
adapters are loaded from the new implementation. Outputs are local artifacts;
the historical model, review images, rig and source photographs remain read-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run(args):
    import cv2
    import darsia
    import numpy as np
    from threadpoolctl import threadpool_limits

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    threadpool_limits(1)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    shared = _module("darsia.fluidflower.phase.um_bm_review_regions",
                     here / "um_bm_review_regions.py")
    mapping_module = _module("darsia.fluidflower.phase.ffum_review_mapping",
                             here / "ffum_review_mapping.py")
    regions = shared.load_model_regions(args.model, series_id="FF_UM")
    if len(regions) != 2527 or len({r.record["region_id"] for r in regions}) != 2527:
        raise ValueError("The bound source must contain all 2527 distinct UM regions")
    groups = defaultdict(list)
    for region in regions:
        groups[(region.frame.manifest_path, region.frame.frame_index)].append(region)
    extracted = json.loads((args.model / "extraction.json").read_bytes())
    sources = {(row["native_manifest"], row["frame_index"]): row for row in extracted["frames"]}
    plan = json.loads(args.plan.read_bytes())
    resolved_photos = {}
    for root in plan["review_roots"]:
        for row in root["frames"]:
            key = (root["run"], Path(row["photograph"]).name)
            value = (row["photograph"], row["raw"])
            if key in resolved_photos and resolved_photos[key] != value:
                raise ValueError(f"Ambiguous photograph binding: {key}")
            resolved_photos[key] = value
    keys = list(groups)
    if args.pilot_runs:
        keys = [next(key for key in keys if groups[key][0].frame.run == run)
                for run in args.pilot_runs]
    start = time.time()
    receipts = []
    for number, key in enumerate(keys):
        selected = groups[key]
        frame = selected[0].frame
        source = sources[key]
        frame_id = f"{number:03d}_{frame.run}_frame_{frame.frame_index:03d}"
        saved_path = output / f"{frame_id}.json"
        photo_value, raw_value = resolved_photos[(frame.run, frame.source["source_name"])]
        if args.resume and saved_path.is_file():
            saved = json.loads(saved_path.read_bytes())
            if (saved["original_manifest_sha256"] != frame.manifest_sha256
                    or saved["display_origin_jpg_xy"] != list(args.origin)
                    or Path(saved["original_photo"]) != Path(photo_value)
                    or saved["raw"] != (str(Path(raw_value)) if raw_value else None)
                    or saved["physical_context"] != source["domain"]
                    or _sha(source["domain"]["path"]) != source["domain"]["sha256"]
                    or [row["region_id"] for row in saved["regions"]]
                    != [region.record["region_id"] for region in selected]
                    or _sha(saved["original_photo"]) != saved["original_photo_sha256"]
                    or any(_sha(row["path"]) != row["sha256"] for row in saved["correction_sources"])):
                raise ValueError("Saved UM geometry differs from its bound source")
            receipts.append(dict(run=frame.run, frame_index=frame.frame_index,
                                 path=str(saved_path), regions=len(selected)))
            print(json.dumps(dict(stage="resumed", **receipts[-1])), flush=True)
            continue
        photo = Path(photo_value)
        raw = Path(raw_value) if raw_value else None
        photo_sha = _sha(photo)
        recorded_photo_sha = frame.source.get("source_sha256")
        if recorded_photo_sha is not None and photo_sha != recorded_photo_sha:
            raise ValueError(f"Historical photograph changed: {photo}")
        rig = shared.shared_drive_path(frame.manifest["correction_contract"]["legacy_rig"])
        correction_paths = [rig/f"correction_{i}.npz" for i in (1, 2, 3)]
        resize = darsia.ResizeCorrection(); resize.load(correction_paths[0])
        drift = darsia.DriftCorrection(); drift.load(correction_paths[1])
        curvature = darsia.CurvatureCorrection(); curvature.load(correction_paths[2])
        image = darsia.imread(photo).img
        resized = resize.correct_array(image)
        cv2.setRNGSeed(0)
        _, matrix = shared.capture_legacy_drift(drift, resized)
        shape = tuple(map(int, curvature.cache["shape"]))
        if shape != frame.shape_hw:
            raise ValueError("Saved curvature shape differs from the bound historical content")
        mapper = mapping_module.UmReviewMapping(curvature.cache["grid"], shape, matrix,
                    (image.shape[1], image.shape[0]), (resized.shape[1], resized.shape[0]),
                    tuple(args.origin))
        cell_check = mapper.validate_all_cells()
        domain_ref = source["domain"]
        if _sha(domain_ref["path"]) != domain_ref["sha256"]:
            raise ValueError("Historical physical-coordinate context changed")
        if domain_ref.get("kind") == "registered-context":
            with np.load(domain_ref["path"], allow_pickle=False) as context:
                spacing = [abs(float(context["y_m"][1]-context["y_m"][0])),
                           abs(float(context["x_m"][1]-context["x_m"][0]))]
        else:
            if Path(domain_ref["path"]).name != "facies.npz":
                raise ValueError("Unknown historical UM physical-domain representation")
            # Original UM native_domain uses the serialized rig dimensions.
            # Its classifications are neither recomputed nor modified here.
            with np.load(domain_ref["path"], allow_pickle=True) as context:
                if tuple(context["array"].shape) != shape:
                    raise ValueError("Historical rig domain differs from review content")
                spacing = (np.asarray(context["metadata"].item()["dimensions"]) / shape).tolist()
        rows, outlines, samples = [], {}, []
        for region in selected:
            outline = mapping_module.annotation_content_outline(region.record["payload"],
                canvas_wh=frame.manifest["canvas_wh"],
                content_box_xywh=frame.manifest["content_box_xywh"], step=4, clip=True)
            valid = ((outline[:, 0] >= 0) & (outline[:, 0] <= shape[1]-1)
                     & (outline[:, 1] >= 0) & (outline[:, 1] <= shape[0]-1))
            q = outline[valid]
            if len(q) == 0:
                raise ValueError(f"No source content on region outline: {region.record['region_id']}")
            moved = mapper.content_to_display(q)
            recovered = mapper.display_to_content(moved)
            error = np.linalg.norm(recovered-q, axis=1)
            if not np.isfinite(error).all() or error.max() > .01:
                raise ValueError("UM annotation coordinate roundtrip failed")
            identity = region.record["region_id"]
            outlines[identity] = moved.astype(np.float32)
            sample = q[np.linspace(0, len(q)-1, min(len(q), 32)).astype(int)]
            samples.append(sample)
            rows.append(dict(original_record=region.record, original_extraction=region.extraction_record,
                region_id=identity, transformed_outline_key=identity,
                bbox_display_xyxy=[*moved.min(0).tolist(), *moved.max(0).tolist()],
                roundtrip_max_content_px=float(error.max()),
                outside_historical_content_outline_points=int((~valid).sum())))
        metric = mapper.physical_jacobian(np.concatenate(samples), content_spacing_yx_m=spacing)
        singular = np.linalg.svd(metric, compute_uv=False)
        np.savez_compressed(output/f"{frame_id}_outlines.npz", **outlines)
        # Saved curvature is immutable and shared by frames that use this rig.
        grid_sha = _sha(correction_paths[2])
        grid_path = output/f"curvature_{grid_sha}.npz"
        if not grid_path.exists():
            np.savez(grid_path, shape=shape, grid=curvature.cache["grid"])
        receipt = dict(run=frame.run, frame_index=frame.frame_index, source_name=frame.source["source_name"],
            original_manifest=frame.manifest_path, original_manifest_sha256=frame.manifest_sha256,
            original_photo=str(photo), original_photo_sha256=photo_sha,
            photo_binding_status=("verified legacy source SHA" if recorded_photo_sha is not None
                                  else "captured original source SHA; legacy manifest has no source checksum"),
            raw=str(raw) if raw is not None else None,
            raw_binding_status="bound" if raw is not None else "unavailable; RAW preparation must fail",
            correction_sources=[dict(path=str(p), sha256=_sha(p)) for p in correction_paths],
            curvature_cache=str(grid_path), drift_matrix=matrix.tolist(), drift_rng_seed=0,
            photo_wh=[image.shape[1], image.shape[0]], resized_wh=[resized.shape[1], resized.shape[0]],
            display_origin_jpg_xy=args.origin, common_geometric_anchor="green unchanged; R-only relative JPG profile",
            physical_context=domain_ref, content_spacing_yx_m=spacing, cell_check=cell_check,
            projected_1mm_pixels_singular_values=dict(min=singular.min(0).tolist(),
                p05=np.percentile(singular,5,axis=0).tolist(), median=np.median(singular,axis=0).tolist(),
                p95=np.percentile(singular,95,axis=0).tolist(), max=singular.max(0).tolist()),
            regions=rows)
        for name in receipt["projected_1mm_pixels_singular_values"]:
            receipt["projected_1mm_pixels_singular_values"][name] = (
                np.asarray(receipt["projected_1mm_pixels_singular_values"][name])*.001).tolist()
        _write(output/f"{frame_id}.json", receipt)
        receipts.append(dict(run=frame.run, frame_index=frame.frame_index, path=str(output/f"{frame_id}.json"),
                             regions=len(rows)))
        print(json.dumps(dict(stage="mapped", **receipts[-1], seconds=round(time.time()-start,1),
            scale=receipt["projected_1mm_pixels_singular_values"], cell_check=cell_check)), flush=True)
    _write(output/"transfer_index.json", dict(source_model=str(args.model), source_regions=len(regions),
        source_frames=len(groups), prepared_frames=receipts, pilot=bool(args.pilot_runs),
        photographs_modified=False, old_rig_geometry_applied_to_new_photographs=False,
        feature_metric_selected=False, elapsed_seconds=time.time()-start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-runs", nargs="*")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--origin", type=int, nargs=2, default=(-19, -11))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
