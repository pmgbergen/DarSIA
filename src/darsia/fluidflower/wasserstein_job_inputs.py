"""Named original mass-product consumers for a single Wasserstein queue task.

Called by InputCatalog.collect during preparation and its normal RunIOBinding
replay. Only metadata is decoded here; numerical arrays are never loaded.
"""
from __future__ import annotations

from copy import deepcopy
import ntpath
from pathlib import PurePosixPath, PureWindowsPath
import posixpath

from .wasserstein_job_protocol import descriptor, validate_operation

MASS_PRODUCTS = ("mass_total", "mass_aqueous", "mass_free_gas")
PRODUCTS = (*MASS_PRODUCTS, "transport_weight")


def relative_descriptor(index_path, value):
    """Resolve an original index locator without replacing its byte identity."""
    bound = dict(value)
    raw = bound.get("path")
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("A field frame needs its original locator")
    windows = PureWindowsPath(index_path)
    if windows.drive:
        path = PureWindowsPath(raw)
        bound["path"] = ntpath.normpath(str(path if path.is_absolute() else windows.parent / path))
    else:
        path = PurePosixPath(raw)
        bound["path"] = posixpath.normpath(str(path if path.is_absolute() else PurePosixPath(index_path).parent / path))
    # Normalize the original locator lexically on its own platform. InputCatalog
    # admits this resulting path under roots before stat/open; replay maps this
    # same name to the worker's positively contained, byte-verified location.
    # Do not resolve a historical Windows drive through a different host's cwd.
    return descriptor(bound)


def catalogue_wasserstein_inputs(catalog, config):
    """Add the exact saved plan, indices and consumed frame/weight payloads."""
    plan = catalog.doc(config["plan"], "configuration", "wasserstein.plan")
    contract = validate_operation(config, plan)
    manifest_bound = descriptor(contract["mass_result"])
    manifest = catalog.doc(manifest_bound, "configuration", "wasserstein.mass_result")
    if (manifest.get("format") != "darsia-fluidflower-mass-analysis-result-v1"
            or manifest.get("complete") is not True
            or manifest.get("mass_analysis_plan_id") != contract["mass_analysis_plan_id"]
            or manifest.get("mass_task_contract_sha256") != contract["mass_task_contract_sha256"]):
        raise ValueError("Wasserstein mass results differ from the frozen upstream task")
    rows = manifest.get("run_results")
    if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
            or len({row.get("run_id") for row in rows}) != len(rows)):
        raise ValueError("The mass result needs distinct explicit run records")
    run_map = {row.get("run_id"): row for row in rows}
    references = contract["mass_result"].get("product_artifacts", [])
    reference_map = {(row.get("run_id"), row.get("product", row.get("component"))): row for row in references}
    if len(reference_map) != len(references):
        raise ValueError("Duplicate frozen mass-product references")
    selected = config["comparison"]
    index_rows = {}
    source_times = None
    for run in contract["run_ids"]:
        row = run_map.get(run, {})
        times = row.get("target_times_hours")
        if row.get("complete") is not True or not isinstance(times, list) or not times:
            raise ValueError("A selected run has no complete mass time grid: " + run)
        if source_times is not None and times != source_times:
            raise ValueError("Wasserstein runs must retain their shared time grid")
        source_times = times
        if selected["time_index"] >= len(times) or times[selected["time_index"]] != selected["time_hours"]:
            raise ValueError("The comparison time differs from its actual mass input")
        for product in PRODUCTS:
            use = "wasserstein." + run + "." + product
            bound = descriptor(reference_map.get((run, product)))
            original = row.get("products", {}).get(product)
            if not isinstance(original, dict):
                raise ValueError("Missing original mass product: " + use)
            # Original manifests may omit a size; the frozen plan records it.
            normalized = relative_descriptor(manifest_bound["path"],
                dict(original, size_bytes=original.get("size_bytes", bound["size_bytes"])))
            if normalized != bound:
                raise ValueError("Mass manifest and saved index binding differ: " + use)
            index = catalog.doc(bound, "configuration", use + ".index")
            frames = index.get("frames")
            if (index.get("format") != "darsia-fluidflower-field-index-v1"
                    or index.get("run_id") != run or index.get("product") != product
                    or not isinstance(frames, list) or not frames
                    or any(not isinstance(frame, dict) for frame in frames)):
                raise ValueError("Invalid original field index: " + use)
            expected_times = [frame.get("time_hours") for frame in frames]
            if (product == "transport_weight" and len(frames) not in (1, len(times))
                    or (product != "transport_weight" or len(frames) != 1) and expected_times != times):
                raise ValueError("Original mass/weight index has a different time grid: " + use)
            index_rows[(run, product)] = dict(bound=bound, document=deepcopy(index))
            # The original materializer consumes the first weight for every
            # selected run, requiring one symmetric cost field across runs.
            consumed = [0] if product == "transport_weight" else (
                [selected["time_index"]] if run in (selected["run_1"], selected["run_2"])
                and product == selected["mass_component"] else [])
            for number in consumed:
                catalog.add(relative_descriptor(bound["path"], frames[number]),
                            "rig_model" if product == "transport_weight" else "image",
                            use + ".frame." + str(number))
    return dict(plan=plan, contract=contract, manifest=manifest, indices=index_rows)


def collect_wasserstein_catalog(catalog, config_descriptor, config):
    """Preparation helper with the same rows InputCatalog.collect will replay."""
    from .pipeline.job_inputs import CATALOG_FORMAT, validate_catalog
    reference = catalog.add(config_descriptor, "configuration", "run.configuration")
    catalogue_wasserstein_inputs(catalog, config)
    return validate_catalog(dict(format=CATALOG_FORMAT, complete=True, run=config["run"],
        configuration=reference, inputs=[catalog.entries[key] for key in sorted(catalog.entries)],
        payload_hashes_verified=False, numerical_calls_started=False,
        runtime_support="darsia-role-io-runtime-v1"))
