"""Prepare and execute an isolated replay of the saved AC28 lighting method.

This is deliberately separate from production jobs. Original files and the
series choice are read only; a successful replay writes a new output folder.
"""

import ast
from copy import deepcopy
import datetime as dt
import json
from pathlib import Path
import tomllib
import uuid

import cv2
import numpy as np

from .normalisation_recipe import _json, _one, assemble_saved_method, descriptor, require_exact_array
from .native.io import resolve_asset_path, sha256_file


FITTER_SHA = "819d783521e91d5e3cae6118b2b1932ec724a9cadff7855221076b735057a2d0"
RESIDUAL_SHA = "f2f11c66394baaf585223e2295cb3173df6c00283fec23ca0c30feecbdc56380"
AREA_SHA = "8553de98f0c6a999b1a3add780892e815cd17bc513728f8b698ff3cbd8b4f097"


def _missing_source(source):
    return ValueError(
        "Original reproduction source is missing. Restore this exact file and prepare again.\n"
        + "Path: " + source["path"] + "\nExpected SHA-256: " + source["sha256"])


def _source_hash(source):
    try:
        return sha256_file(source["path"])
    except FileNotFoundError as error:
        raise _missing_source(source) from error


def _original_parameters(fit, source_spec):
    """Bind literal parameter data to the exact original source implementations."""
    if (fit["fitter_sha256"] != FITTER_SHA or fit["local_residual_module_sha256"] != RESIDUAL_SHA
            or source_spec["source_renderer_sha256"] != AREA_SHA):
        raise ValueError("This replay requires the exact recorded fitter, residual and RAW sampler.")
    return dict(
        installation=dict(chart_offset_xy=[16.0, 0.0], virtual_reference_name="ac28_virtual_old_donor_consensus",
                          maximum_asset_iterations=12, previous_weight=.5, next_weight=.5),
        residual=dict(colour_sigma=3, division_floor=1e-6, minimum_centring_pixels=100,
                      residual_colour_limit=.08, fraction_tolerance=1e-6, minimum_support_pixels=1000,
                      sigma_reference_cells=8.0, minimum_smooth_support=.05),
        area=dict(homography_min_determinant=1e-9, maximum_rows=512, maximum_columns=800,
                  cell_scale=8, tile_cell_rows=16, signal_floor=1e-6,
                  contract="ffac-source-bound-ea-direct-8x8-area-grid-v1",
                  aggregation="8x8 arithmetic area mean of full-native EA camera-linear RGB, then natural log",
                  colour_basis="RAW black-800; no WB, matrix, gamma, autoexposure or clipping"))


def prepare_replay(series_root, run, frame_index):
    """Read and bind the saved recipe; never decode, fit, or alter source files."""
    root = Path(series_root)
    manifest = root / "series.toml"
    raw = manifest.read_bytes()
    document = tomllib.loads(raw.decode("utf-8-sig"))
    if not ((document.get("phase_normalisation") or {}).get("field_sources") or {}).get("v5_raw_photometry", {}).get("source"):
        raise ValueError("This series has no replayable saved RAW correction source.")
    if run not in document["runs"]["ids"]:
        raise ValueError("Choose a run belonging to this series.")
    native = document["native"]
    configuration_source = descriptor(root / native["configuration"], native["sha256"])
    configuration = _json(configuration_source)
    aliases = configuration["source"]["asset_aliases"]
    source = document["phase_normalisation"]["field_sources"]["v5_raw_photometry"]["source"]
    plan = assemble_saved_method(source, run, frame_index=frame_index,
        resolve_path=lambda value: resolve_asset_path(value, aliases))
    fit_source = plan["verified_metadata"][2]
    fit = _json(fit_source)
    fit_directory = Path(fit_source["path"]).parent.parent
    outputs = fit_directory.parent
    legacy_root = outputs.parent
    sources = list(plan["verified_metadata"]) + [configuration_source]

    def path(value):
        candidate = resolve_asset_path(value, aliases)
        return candidate if candidate.is_absolute() else legacy_root / candidate

    def bound(value, digest=None):
        candidate = path(value)
        item = descriptor(candidate, digest or sha256_file(candidate))
        sources.append(item)
        return item

    def read(value, digest=None):
        item = bound(value, digest)
        return _json(item), item

    fitter = bound(fit_directory / "fit_multirun_observed_sand_v4r5.py", FITTER_SHA)
    if sha256_file(fitter["path"]) != FITTER_SHA:
        raise ValueError("The recorded normalisation fitter has changed.")
    spec, _ = read(fit_directory / "source_only_extension_spec_v6_v4r5_interior_sand.json", fit["spec_sha256"])
    parent, _ = read(fit_directory / spec["parent_specs"][0])
    convergence, _ = read(outputs / "ffac-v5-portable-preview-20260929/ea_native_v5_fixedpoint_spec.json")
    source_rows, _ = read(outputs / "ffac-raw-jpeg-composite-audit-20260929/expanded_ea43_source_grid_result_v2.json", fit["source_manifest_sha256"])
    source_spec, _ = read(source_rows["spec_path"], source_rows["spec_sha256"])
    parameters = _original_parameters(fit, source_spec)
    if (parent["frozen_cohorts"]["old_donors"] != fit["old_donors"]
            or parent["frozen_cohorts"]["new_donors"] != fit["new_donors"]
            or convergence["maximum_asset_iterations"] != parameters["installation"]["maximum_asset_iterations"]):
        raise ValueError("The recorded donor lists or iteration limit have changed.")
    masks, _ = read(fit["photo_sand_manifest_path"], fit["photo_sand_manifest_sha256"])
    support, _ = read(fit_directory / "v4r5_interior_source_support_43_v1.json", fit["source_interior_support_receipt_sha256"])
    binding, _ = read(source_spec["exact_B_source_binding_path"], source_spec["exact_B_source_binding_sha256"])
    geometry, _ = read(source_spec["safe_geometry_path"], source_spec["safe_geometry_sha256"])
    old_directory = outputs / "ffac-crossrun-geometry-independent-audit-20260929/ea_refgeom_safe_calibration43_v4"
    old, _ = read(old_directory / "manifest.json", fit["old_v4_manifest_sha256"])
    # This literal directory belongs to the exact SHA-bound original fitter.
    tree = ast.parse(Path(fitter["path"]).read_bytes())
    clip_values = [node.value.args[0].value for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) for t in node.targets)
        and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", None) == "Path"
        and len(node.value.args) == 1 and isinstance(node.value.args[0], ast.Constant)]
    if len(clip_values) != 1:
        raise ValueError("The original ColorChecker source directory is not unambiguous.")
    rows = []
    for row in source_rows["rows"]:
        identity = row["run"]
        baseline = _one([item for item in binding["rows"] if item["is_canonical"]], identity)
        mask = _one(masks["rows"], identity)
        chart = _one(old["results"], identity)
        saved_fit = _one(fit["results"], identity)
        if row["source_manifest_sha256"] != mask["source_manifest_sha256"] or row["source_arw_sha256"] != baseline["camera_arw_sha256"]:
            raise ValueError("Baseline and optical support belong to different sources: " + identity)
        rows.append(dict(run=identity, baseline=deepcopy(baseline), source=deepcopy(row),
            homography=deepcopy(geometry["runs"][identity]["H_ref_to_run_native"]),
            mask=bound(mask["mask_npz_path"], mask["mask_npz_sha256"]),
            mask_receipt=bound(mask["receipt_path"], mask["receipt_sha256"]),
            chart=bound(chart["baseline_path"], chart["baseline_sha256"]),
            chart_sidecar=bound(path(chart["baseline_path"]).with_suffix(".json"), chart["sidecar_sha256"]),
            clipped=bound(Path(clip_values[0]) / (identity + ".npz")),
            saved_fit=bound(saved_fit["baseline_path"], saved_fit["baseline_sha256"]),
            saved_sidecar=bound(path(saved_fit["baseline_path"]).with_suffix(".json"), saved_fit["sidecar_sha256"]),
            expected_grid=bound(row["expanded_npz_path"], row["expanded_npz_sha256"]),
            source_certificate=bound(row["source_certificate_path"], row["source_certificate_sha256"])))
    photo_aggregate, photo_bound = read(plan["producer_identity"]["photo_source_aggregate_path"], plan["producer_identity"]["photo_source_aggregate_sha256"])
    processor_rows = photo_aggregate.get("rows", photo_aggregate.get("revisions", []))
    processor_matches = [row for row in processor_rows if row.get("run") == run and row.get("source_manifest_sha256") == rows[[r["run"] for r in rows].index(run)]["source"]["source_manifest_sha256"]]
    if len(processor_matches) != 1:
        raise ValueError("The saved frame source is missing or ambiguous in the original photo aggregate.")
    bound(processor_matches[0]["source_manifest_path"], processor_matches[0]["source_manifest_sha256"])
    for key in ("expected_reference", "expected_reference_sidecar", "reference_assets", "expected_log_rgb", "expected_valid_mask"):
        plan[key] = bound(plan[key]["path"], plan[key]["sha256"])
    selected_source = _json(plan["verified_metadata"][-1])["source"]
    for key in ("camera_arw", "camera_jpeg", "original_review_jpeg"):
        item = bound(selected_source[key], selected_source[key + "_sha256"])
        if not Path(item["path"]).is_file():
            raise _missing_source(item)
    old_fields = {item["run"]: bound(item["field_path"], item["field_sha256"]) for item in fit["old_B_local_residual_fields"]}
    matrix = bound(old["chart_matrix_path"], old["chart_matrix_sha256"])
    return dict(format="saved-lighting-replay-request-v1", series_root=str(root), manifest_bytes=raw,
        plan=plan, configuration=configuration, parameters=parameters, convergence=convergence["convergence_at_end_of_iteration"],
        fit=fit, masks=masks, support=support, source_spec=source_spec, rows=rows, matrix=matrix,
        old_fields=old_fields, processor_row=processor_matches[0], photo_aggregate=photo_bound, sources=sources)


def _load_array(bound, key=None):
    if _source_hash(bound) != bound["sha256"]:
        raise ValueError("The saved array source changed: " + bound["path"])
    if key is None:
        return np.load(bound["path"], allow_pickle=False)
    with np.load(bound["path"], allow_pickle=False) as archive:
        return archive[key]


def execute_replay(request, output_parent, *, progress=None, check_cancelled=None):
    """Fit and apply from RAW, requiring exact saved bits before publishing output."""
    from .native import photometry
    from .native.normalisation_installation import fit_installation
    from .native.normalisation_raw import registered_edge_aware_raw_area_grid
    from .native.normalisation_apply import apply_historical_frame
    from .native.normalisation_reproduction import require_exact_fit
    from .native.source import SourceProcessor

    def report(message):
        if check_cancelled is not None:
            check_cancelled()
        if progress is not None:
            progress(message)

    if (Path(request["series_root"]) / "series.toml").read_bytes() != request["manifest_bytes"]:
        raise ValueError("The series changed; prepare the replay again.")
    for source in request["sources"]:
        if check_cancelled is not None:
            check_cancelled()
        if _source_hash(source) != source["sha256"]:
            raise ValueError("A replay source changed; prepare again: " + source["path"])
    output_parent = Path(output_parent)
    if not output_parent.is_dir():
        raise ValueError("Choose an existing output folder.")
    config, parameters = request["configuration"], request["parameters"]
    plan, fit = request["plan"], request["fit"]
    grid = photometry.ReferenceGrid(tuple(fit["grid_shape_hw"]), tuple(fit["grid_origin_cell_xy"]),
        tuple(config["photometry"]["grid_shape_hw"]), parameters=config["photometry"])
    B, sand, xy, charts, clipped = {}, {}, {}, {}, {}
    for index, row in enumerate(request["rows"], start=1):
        identity, baseline, source = row["run"], row["baseline"], row["source"]
        report(f"Reproducing RAW baseline {index}/{len(request['rows'])}: " + identity.upper())
        expected = dict(arw_sha256=baseline["camera_arw_sha256"], camera_jpeg_sha256=baseline["camera_jpeg_sha256"],
                        review_native_jpeg_sha256=baseline["original_review_jpeg_sha256"])
        resolve = lambda value: resolve_asset_path(value, config["source"]["asset_aliases"])
        B[identity], _ = registered_edge_aware_raw_area_grid(resolve(baseline["camera_arw"]), resolve(baseline["camera_jpeg"]),
            resolve(baseline["original_review_jpeg"]), expected_source_hashes=expected,
            source_certificate_path=row["source_certificate"]["path"], source_certificate_sha256=row["source_certificate"]["sha256"],
            h_target_to_review_native=np.asarray(row["homography"], np.float64), origin_cell_xy=grid.origin_cell_xy,
            shape_cells_hw=grid.shape_hw, source_id=identity, profile_id=config["source"]["profiles"][identity],
            content_box=config["source"]["content_box_xywh"], parameters=config["registration"], area_parameters=parameters["area"])
        require_exact_array(B[identity], _load_array(row["expected_grid"], source["expanded_npz_key"]))
        good = _load_array(row["mask"], "sand").astype(bool)
        distance = cv2.distanceTransform(good.astype(np.uint8), cv2.DIST_L2, 5)
        sand[identity] = good & (distance >= fit["source_interior_erosion_cells"])
        chart = photometry.load_grid_baseline(Path(row["chart"]["path"]), parameters=config["photometry"])
        xy[identity], charts[identity] = chart.xy, chart.CB
        clipped[identity] = _load_array(row["clipped"], "chart_clip").any(0)
    old, new = fit["old_donors"], fit["new_donors"]
    count_old = np.sum(np.stack([sand[r] for r in old]), axis=0)
    count_new = np.sum(np.stack([sand[r] for r in new]), axis=0)
    common_old = count_old >= fit["source_sand_old_donor_minimum"]
    common_new = common_old & (count_new >= fit["source_sand_new_donor_minimum"])
    if int(common_old.sum()) != request["support"]["common_old_cells"] or int(common_new.sum()) != request["support"]["common_new_cells"]:
        raise ValueError("The original donor support was not reproduced.")
    if fit["spatial_cv"]:
        yy, xx = np.mgrid[:grid.shape_hw[0], :grid.shape_hw[1]]
        withheld = ((7*(yy//4) + 11*(xx//4)) % 5) == 0
        common_old &= ~withheld
        common_new &= ~withheld
    result = fit_installation(baseline_rgb=B, fit_support={r: (sand[r] & common_old).astype(np.int16) for r in B},
        chart_xy=xy, chart_log_rgb=charts, chart_clipped=clipped, grid=grid, old_donors=old, new_donors=new,
        new_light_runs=[r for r in B if int(r[2:]) >= 50], reference_run="ac28", matrix=_load_array(request["matrix"], "M"),
        common_old=common_old, common_new=common_new, parameters=parameters["installation"], convergence=request["convergence"],
        residual_parameters=parameters["residual"], photometry_parameters=config["photometry"],
        check_cancelled=check_cancelled, on_iteration=lambda row: report("Fitting lighting: iteration " + str(row["iteration"])))
    for row in request["rows"]:
        saved, _ = photometry.load_baseline(Path(row["saved_fit"]["path"]), parameters=config["photometry"])
        require_exact_fit(result["fits"][row["run"]], saved)
    saved, _ = photometry.load_baseline(Path(plan["expected_reference"]["path"]), parameters=config["photometry"])
    require_exact_fit(result["reference"], saved)
    for key in ("consensus", "flat"):
        require_exact_array(result[key], _load_array(plan["reference_assets"], key))
    for identity, field in result["old_fields"].items():
        require_exact_array(field, _load_array(request["old_fields"][identity], "field"))
    report("Applying reproduced lighting to " + plan["run"].upper())
    # Validate the complete cohort under the current policy, then replay the
    # historical keep/raw application in this dedicated B1 child process.
    cv2.ipp.setUseIPP(False)
    processor = SourceProcessor(request["processor_row"], calibration_manifest_path=Path(plan["verified_metadata"][2]["path"]),
        photo_aggregate_path=Path(request["photo_aggregate"]["path"]), enclosed_hole_policy=config["source"]["policy_ids"]["keep"],
        display_mode=config["source"]["policy_ids"]["raw_display"], isolated_invalid_policy=config["source"]["policy_ids"]["keep"], parameters=config)
    processor.fit = result["fits"][plan["run"]]
    processor.reference = result["reference"]
    processor.old_field = result["old_fields"].get(plan["run"])
    frame = _json(plan["verified_metadata"][-1])
    source = frame["source"]
    cv2.ipp.setUseIPP(True)
    try:
        corrected = apply_historical_frame(processor, run=plan["run"], frame_id="frame_" + str(plan["frame_index"]),
            camera_jpeg=resolve(source["camera_jpeg"]), camera_arw=resolve(source["camera_arw"]), review_jpeg=resolve(source["original_review_jpeg"]),
            source_manifest_sha256=request["processor_row"]["source_manifest_sha256"],
            content_shape_hw=tuple(config["source"]["native_shape_hw"]), content_box_xywh=tuple(config["source"]["content_box_xywh"]))
    finally:
        cv2.ipp.setUseIPP(False)
    require_exact_array(corrected.log_linear_rgb, _load_array(plan["expected_log_rgb"]))
    require_exact_array(corrected.required_valid_mask, _load_array(plan["expected_valid_mask"]))
    report("Exact saved fit and output reproduced; saving a separate copy")
    name = "normalisation-reproduction-" + plan["run"] + "-" + uuid.uuid4().hex[:12]
    output, pending = output_parent / name, output_parent / ("." + name + ".partial")
    pending.mkdir()
    np.save(pending / "log_linear_rgb.npy", corrected.log_linear_rgb, allow_pickle=False)
    np.save(pending / "valid_mask.npy", corrected.required_valid_mask, allow_pickle=False)
    receipt = dict(format="saved-lighting-reproduction-result-v1", run=plan["run"], frame_index=plan["frame_index"],
        reference_run="ac28", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), exact_raw_grid_parity=True,
        exact_saved_fit_parity=True, exact_saved_output_parity=True, exact_saved_valid_mask_parity=True,
        iteration_history=result["iteration_history"], parameters=parameters, sources=request["sources"])
    (pending / "reproduction.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    if check_cancelled is not None:
        check_cancelled()
    pending.rename(output)
    return str(output)
