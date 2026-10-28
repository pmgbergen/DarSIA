"""Table diagnostics from original replay DATA without publication authority."""

from pathlib import Path

import numpy as np


def _context_products(context, painting_binding, source_analysis, effective_labels):
    """Match actual materialized source/products before consuming original masks."""
    from .painted_adoption_table import _local, _same
    from .painted_analysis_exclusions import prepared_exclusion
    from .painted_target_integration import effective_product
    from .photo_edge_continuation import read_contracts, verify_source

    preparation, folder = context["preparation"], Path(context["folder"])
    if not _same(preparation["painting_facies"], painting_binding):
        raise ValueError("The outside context belongs to another approved painting")
    rows = [row for row in preparation["runs"] if row.get("status") == "prepared"]
    if len({row["run_id"] for row in rows}) != len(rows) or {row["run_id"] for row in rows} != set(effective_labels):
        raise ValueError("The outside context and replay have different native run identities")
    source_map = _local(folder, preparation["source_map"])
    with np.load(source_map["path"], allow_pickle=False) as archive:
        actual_source = np.array(archive["array"], copy=True)
    if actual_source.dtype != source_analysis.dtype or not np.array_equal(actual_source, source_analysis):
        raise ValueError("The outside context differs from the actual replayed source array")
    record = preparation["continuation"]
    sidecar = _local(folder, dict(path=record["file"], sha256=record["sha256"], size_bytes=record["size_bytes"]))
    contracts = read_contracts(sidecar["path"])["runs"]
    verify_source(painting_binding, actual_source, contracts, preparation=preparation, folder=folder)
    native_ids, guarded = set(), [source_map, sidecar]
    for row in rows:
        arrays, bound = effective_product(preparation, folder, row)
        labels = effective_labels[row["run_id"]]
        if arrays["array"].dtype != labels.dtype or not np.array_equal(arrays["array"], labels):
            raise ValueError("The outside context differs from the actual replayed effective array")
        exclusion = prepared_exclusion(preparation, folder, row["run_id"])
        values = labels if exclusion is None else labels[~exclusion]
        native_ids.update(int(value) for value in np.unique(values))
        guarded.append(bound)
    return preparation, folder, native_ids, guarded


def transfer_table_data(current_revision, *, painting_binding, source_analysis, effective_labels,
                        source_fixture_drawn_ids=(), native_fixture_drawn_ids=(), outside_context=None):
    """Join authenticated replay arrays to their actual saved session and parent CSV.

    The caller supplies source, effective native arrays and fixture IDs already
    checked by the original replay. An actual materializer context may supply
    original gap masks and outside proof. Otherwise zero remains unmapped.
    """
    from .adoption_facies_table import build_adoption_facies_table, MAX_PARENT_BYTES
    from .painted_adoption_table import (
        _checked, _read, _same, _source, _with_frame_row,
        table_region_data, verify_outside_proof, verify_version,
    )

    session, approved, frozen_frame = verify_version(painting_binding)
    source = _source(current_revision)
    parent = _checked(source["revision"], metadata=True)
    if not _same(source["revision"], painting_binding["frame_basis"]["workspace_revision"]):
        raise ValueError("The table and explicit painting frame belong to different workspace revisions")
    if (not isinstance(parent, dict) or parent.get("revision_id") != source.get("revision_id")
            or Path(str((parent.get("artifacts") or {}).get("facies_properties_path"))).resolve()
            != Path(source["csv"]["path"]).resolve()):
        raise ValueError("The table source is not its actual parent Store CSV")
    committed = (parent.get("artifacts") or {}).get("source_mode_files", {}).get("facies_properties")
    if not isinstance(committed, dict) or not _same(_checked(committed), _checked(source["csv"])):
        raise ValueError("The parent table differs from its committed Store artifact identity")

    approved_ids = set(map(int, np.unique(source_analysis)))
    approved_ids.update(int(value) for value in source_fixture_drawn_ids if value)
    native_ids = set()
    for labels in effective_labels.values():
        native_ids.update(int(value) for value in np.unique(labels))
    context_guards = []
    if outside_context is not None:
        preparation, folder, native_ids, context_guards = _context_products(
            outside_context, painting_binding, source_analysis, effective_labels)
    if not native_ids:
        raise ValueError("No native run map is available from the original replay")
    drawn_ids = {int(value) for value in native_fixture_drawn_ids if value}
    if ((native_ids | drawn_ids) - {0}) - approved_ids:
        raise ValueError("A replayed run contains a layer absent from its approved source map")

    _checked(source["csv"])
    parent_bytes = _read(source["csv"]["path"], MAX_PARENT_BYTES)
    table = build_adoption_facies_table(session, parent_bytes,
        expected_sha256=source["csv"]["sha256"], used_layer_ids=sorted(approved_ids - {0}))
    unmapped = [0] if 0 in native_ids else []
    table_ids = native_ids - {0}
    if 0 in native_ids and outside_context is not None:
        context_guards.extend(verify_outside_proof(preparation, folder, approved, frozen_frame))
        table = _with_frame_row(table, parent_bytes)
        table_ids, unmapped = native_ids, []
    spec, regions = table_region_data(table, table_ids, drawn_ids)
    guarded = [source["revision"], source["csv"], painting_binding["version"],
               *painting_binding["files"].values(), *context_guards]
    for bound in guarded:
        _checked(bound)
    return dict(table=table, spec=spec, regions=regions, permeability_unit=session.permeability_unit,
                parent_source=source, guarded_sources=guarded,
                unmapped_label_ids=unmapped)
