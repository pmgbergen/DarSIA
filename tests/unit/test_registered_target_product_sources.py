"""Original V2 replay and effective Store-copy revalidation stay distinct.

These are genuine tiny synthetic Prepare/Store fixtures. Negative cases
deliberately reseal only test descriptors to get past byte admission and reach
the original scientific replay or the effective-archive copy comparison. They
do not forge a campaign approval or replace any production verifier.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import photo_edge_continuation as continuation
from .test_geometry_registered_export_apply import registered_setup  # noqa: F401
from .test_painted_adoption_table import imported, synthetic_payload_admission  # noqa: F401
from .test_painted_analysis_exclusions import (
    _make_gap_case, _paint_component, _adopt_target_fixture,
)


def _archive(path):
    with np.load(path, allow_pickle=True) as data:
        return {name: deepcopy(data[name]) for name in data.files}


def _write_archive(path, values):
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **values)
    return sources.descriptor(path)


def _reseal_guarded_paths(binding, *changed):
    """Update explicit negative-test descriptors, never a production reader."""
    replacements = {Path(row["path"]).resolve(): row for row in changed}
    for row in binding["guarded_sources"]:
        replacement = replacements.get(Path(row["path"]).resolve())
        if replacement is not None:
            row.update(deepcopy(replacement))


def _reseal_preparation(binding, preparation):
    """Keep a mutated test preparation's existing location and explicit hash."""
    path = Path(binding["registration"]["measurement"]["path"])
    path.write_text(json.dumps(preparation, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    bound = sources.descriptor(path)
    binding["registration"]["measurement"] = deepcopy(bound)
    _reseal_guarded_paths(binding, bound)


@pytest.mark.parametrize("tamper", ["original-base-mask", "effective-painted-mask", "store-copy-metadata"])
def test_registered_target_revalidation_keeps_original_replay_and_exact_effective_store_copy(
        registered_setup, tmp_path, monkeypatch, tamper):
    case = _make_gap_case(registered_setup, tmp_path, monkeypatch)
    edited, _, correction = _paint_component(case)
    prepared, revision = _adopt_target_fixture(case, [correction], "registered-target-source-proof")
    folder = Path(prepared["folder"])
    rows = {row["run_id"]: row for row in prepared["runs"] if row.get("status") == "prepared"}
    edited_run = edited["run_id"]
    unedited_run = next(run for run in rows if run != edited_run)
    assert edited_run in prepared["target_corrections"]["runs"]
    assert unedited_run not in prepared["target_corrections"]["runs"]
    bindings = {run: sources.registered_run_sources(case.data.project.root, revision, run)
                for run in (edited_run, unedited_run)}
    base_paths = {run: (folder / row["labels"]["path"]).resolve() for run, row in rows.items()}
    target_path = (folder / prepared["target_corrections"]["runs"][edited_run]["artifact"]["path"]).resolve()
    store_paths = {Path(bound["files"]["labels"]["path"]).resolve() for bound in bindings.values()}
    original_verifier = continuation.verify_product
    observed = []

    def original_replay(path, *args, **kwargs):
        observed.append(Path(path).resolve())
        assert observed[-1] in set(base_paths.values()), "Effective/Store product entered the original carrier verifier"
        return original_verifier(path, *args, **kwargs)

    # Exercise the live revalidation gate separately from construction/cache
    # admission, including a selected unedited run in the global target cohort.
    with pytest.MonkeyPatch.context() as recording:
        recording.setattr(continuation, "verify_product", original_replay)
        for run, bound in bindings.items():
            before = len(observed)
            sources.verify_registered_sources(bound)
            assert base_paths[run] in observed[before:]
    assert target_path not in observed and not store_paths.intersection(observed)

    binding = deepcopy(bindings[edited_run])
    measurement = Path(binding["registration"]["measurement"]["path"])
    preparation = json.loads(measurement.read_bytes())
    entry = next(row for row in preparation["runs"] if row["run_id"] == edited_run)
    if tamper == "store-copy-metadata":
        store = Path(binding["files"]["labels"]["path"])
        effective_bytes = target_path.read_bytes()
        values = _archive(store)
        labels_before = values["array"].copy()
        metadata = deepcopy(values["metadata"].item())
        metadata["synthetic_store_copy_tamper"] = "same labels, different original archive"
        values["metadata"] = metadata
        binding["files"]["labels"] = _write_archive(store, values)
        _reseal_guarded_paths(binding, binding["files"]["labels"])
        np.testing.assert_array_equal(_archive(store)["array"], labels_before)
        assert target_path.read_bytes() == effective_bytes
        # Its new test descriptor matches the mutated Store file: refusal must
        # arise from exact effective-source copy identity, not its old SHA.
        with pytest.raises(ValueError, match="Store V2 labels differ"):
            sources.verify_registered_sources(binding)
        return

    if tamper == "original-base-mask":
        path = base_paths[edited_run]
        values = _archive(path)
        values["filled"].flat[0] = not bool(values["filled"].flat[0])
        changed = _write_archive(path, values)
        entry["labels"].update({key: changed[key] for key in ("sha256", "size_bytes")})
        expected_error = "V2 continuation mask differs"
    else:
        path = target_path
        values = _archive(path)
        values["painted_mask"].flat[0] = not bool(values["painted_mask"].flat[0])
        changed = _write_archive(path, values)
        artifact = preparation["target_corrections"]["runs"][edited_run]["artifact"]
        artifact.update({key: changed[key] for key in ("sha256", "size_bytes")})
        binding["registration"]["labels"] = deepcopy(changed)
        expected_error = "effective target product differs"
    _reseal_guarded_paths(binding, changed)
    _reseal_preparation(binding, preparation)
    with pytest.raises(ValueError, match=expected_error):
        sources.verify_registered_sources(binding)
