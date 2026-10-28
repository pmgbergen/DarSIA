"""Read original acquisition metadata for a sealed production selection.

This joins an admitted N1 run to its original phase-one configuration so the
Jobs UI can compare the current saved image selection. It neither verifies a
phase-one completion nor grants execution eligibility. The original producer's
catalogue and completion guards remain responsible for those checks.
"""

from __future__ import annotations


def selection_sources(value, config, *, series_manifest):
    """Return ``(phase1_configuration, original_input_catalog)`` for N1 only.

    ``value`` and ``config`` are the unchanged plan and run returned by the
    original execution-plan reader. Only the bound phase-one configuration is
    opened here; RAW, arrays, completion products and workers are not touched.
    The returned catalog retains the plan/run aliases for subsequent geometry
    metadata reads and acquisition-locator comparisons.
    """
    if not isinstance(value, dict) or value.get("pipeline") != "n1":
        raise ValueError("Production selection sources require an explicit n1 plan")
    if not isinstance(config, dict) or config.get("format") != "darsia-n1-run-v1":
        raise ValueError("Production selection sources require a sealed N1 run")

    from .pipeline.alias_job import _manifest
    from .pipeline.job_inputs import InputCatalog
    from .pipeline.staged_data import _bound, _same_json, validate_phase1_run

    job = value.get("job")
    if (not isinstance(job, dict) or job.get("precision") not in {"float32", "float64"}
            or not _same_json(config.get("precision"), job["precision"])):
        raise ValueError("Production run precision differs from its execution plan")
    declarations = value.get("runs")
    if (not isinstance(config.get("run"), str) or not config["run"]
            or not isinstance(declarations, list)
            or any(not isinstance(row, dict) for row in declarations)
            or sum(_same_json(row.get("run"), config["run"]) for row in declarations) != 1):
        raise ValueError("Production run must occur exactly once in its execution plan")
    frames = config.get("frames")
    if not isinstance(frames, list) or not frames or any(not isinstance(row, dict) for row in frames):
        raise ValueError("Production selection requires its complete ordered frames")
    binding = config.get("phase1")
    if not isinstance(binding, dict):
        raise ValueError("Production selection requires its original phase-one binding")
    bound = _bound(binding.get("configuration"), where="production phase-one configuration")

    _path, _document, roots = _manifest(series_manifest)
    catalog = InputCatalog(
        roots, [*value.get("data_aliases", ()), *config.get("data_aliases", ())]
    )
    source = validate_phase1_run(catalog.doc(
        bound, "configuration", "UI.selection.phase1.configuration"
    ))
    if (not _same_json(source["run"], config["run"])
            or not _same_json(source["precision"], config["precision"])):
        raise ValueError("Production source run or precision differs from phase one")

    # The sealer carries these aliases into phase two. Do not make an omitted
    # or conflicting source locator work by silently adding another alias set.
    for alias in source.get("data_aliases", ()):
        if not _same_json(catalog.physical(alias["logical"]), alias["physical"]):
            raise ValueError("Production must carry its original phase-one aliases")

    originals = source["frames"]
    order = [frame.get("case_id") for frame in frames]
    if (len(originals) != len(frames)
            or not _same_json(source["ordered_case_ids"], order)
            or not _same_json([frame["case_id"] for frame in originals], order)):
        raise ValueError("Production case order differs from its original phase-one selection")
    for current, original in zip(frames, originals):
        for field in ("case_id", "frame_index", "elapsed_h"):
            if field not in current or not _same_json(current[field], original[field]):
                raise ValueError("Production frame " + field + " differs from its phase-one source")
        if ("source_binding" not in current
                or not _same_json(current["source_binding"], original["source"]["source_binding"])):
            raise ValueError("Production source binding differs from its phase-one acquisition")
    return source, catalog
