"""Finite original local reuse/Store tests; no campaign/source qualification."""
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import photo_edge_continuation as continuation
from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower.local_geometry_check import (
    CheckedGeometryCandidate, check_geometry_candidate,
)
from darsia.fluidflower.local_map_adoption import (
    PROOF_FORMAT, prepare_local_adoption, check_local_adoption,
    save_checked_local_adoption, verify_local_preparation, verify_local_selected,
)
from .test_painted_analysis_exclusions import (
    ACTION, gap_case, registered_setup, imported, synthetic_payload_admission,
    _approve, _resolved,
)  # noqa: F401 -- genuine original finite fixture dependency chain
from .test_painted_transfer_replay import _context, _retained, _unchanged


def _prepare(case, destination):
    approval = _approve(case.review)
    result = prepare_local_adoption(case.review["review_binding"],
        source_context=_context(case), destination=destination,
        analysis_exclusions=approval)
    return result, approval


def _candidate(case, prepared):
    declaration = gaps.make_facies_source_acquisition(prepared, "indicator", **ACTION)
    return check_local_adoption(case.data.revision, prepared["folder"],
        facies_source_acquisition=declaration)


def test_reviewed_map_reuse_publishes_real_new_preparation_without_replaying_carrier(gap_case, monkeypatch):
    before = _retained(gap_case)
    # The original fixture has already produced the actual saved map/review.
    # Failing on a second replay proves reuse, rather than replacing a guard
    # with a successful mock.
    monkeypatch.setattr(continuation, "verify_product",
        lambda *a, **k: pytest.fail("Unchanged reviewed maps were recomputed"))
    prepared, _ = _prepare(gap_case, gap_case.root / "local_reuse")
    assert prepared["painting_geometry"]["format"] == PROOF_FORMAT
    assert not prepared["adopted"]
    actual = adoption.read_preparation(prepared["folder"])
    assert actual == prepared
    assert (Path(prepared["folder"]) / "index.json").is_file()
    for fresh, old in zip(prepared["runs"], gap_case.review["runs"]):
        assert (Path(prepared["folder"]) / fresh["labels"]["path"]).read_bytes() == Path(old["base_product"]["path"]).read_bytes()
    _unchanged(before)


@pytest.mark.parametrize('native_gap', [False, True])
def test_new_chart_products_receive_current_findings_while_original_review_remains_frozen(
        registered_setup, tmp_path, monkeypatch, native_gap):
    from types import SimpleNamespace
    from darsia.fluidflower import painting_chart_transfer as chart
    from darsia.fluidflower import painting_chart_transfer_reader as reader
    from darsia.fluidflower.layer_painting_fixtures import read_underlay
    from darsia.fluidflower.painted_target_integration import effective_product
    from darsia.fluidflower.rig_point_alignment import effective_model_matrix
    from .test_painted_adoption_table import prepared_painting
    from darsia.fluidflower import painted_adoption_table as table

    if native_gap:
        # Genuine original native transport leaves an inside gap in run two,
        # plus real outside-frame zeros. The new approved whole
        # chart covers the former; the latter still needs its exact proof.
        with pytest.raises(gaps.GapReviewRequired) as stopped:
            prepared_painting(registered_setup, tmp_path, fixture=True, continuation=True,
                outside=True, outside_targets=True)
        review = stopped.value.review
        from darsia.fluidflower.painted_adoption_table import verify_version
        drawing, *_ = verify_version(review['source']['painting_facies'])
        assert sum(row['gap']['pixels'] for row in review['runs']) > 0
    else:
        original, _version, drawing, *_ = prepared_painting(
            registered_setup, tmp_path, fixture=True, continuation=True)
        review = gaps.build_gap_review(original, original['folder'])
    retained = _retained(SimpleNamespace(review=review))
    source = read_underlay(review['preparation_context']['source_fixture_underlay'])
    material = source.analysis_layers.copy()
    material[source.fixture_mask] = source.drawn_layers[source.fixture_mask]
    centres = np.asarray([[x, -y] for y in range(6) for x in range(4)], np.float64)
    rows = {row['run_id']: row for row in review['runs']}

    class Session:
        proposal_format = chart.DETECTED_FORMAT

        def __init__(self, *args, **kwargs):
            pass

        def proposal(self, run, *, correction_mm=None, proposal_format=chart.DETECTED_FORMAT):
            row = rows[run]
            fields = gaps._arrays(row['product'])
            model = gaps._checked(row['target_model'], metadata=True)
            native = np.asarray(model['baseline_native_grid']['H_baseline_camera_u_to_rig_corrected_native'])
            metric = np.linalg.inv(effective_model_matrix(model, review['geometry_index'])) @ np.linalg.inv(native)
            return chart.propose_chart_transfer(run, fields['array'], fields['inside'], fields['array'] == 8,
                material, source.fixture_mask, np.linalg.inv(row['native_to_painting_h']), metric,
                centres, centres, dict(review=review['review_binding'], painting_facies=review['source']['painting_facies'],
                    target_model=row['target_model']), chart_id=8,
                known_ids=[layer.layer_id for layer in drawing.layers], correction_mm=correction_mm,
                proposal_format=proposal_format)

    monkeypatch.setattr(reader, 'ChartTransferSession', Session)
    approvals = {run: chart.confirm_chart_transfer(Session().proposal(run), actor='synthetic human',
        reason='Explicit new whole-chart fixture review') for run in rows}
    prepared = prepare_local_adoption(review['review_binding'], source_context=review['preparation_context'],
        destination=tmp_path / 'current_chart_reuse', chart_transfers=approvals)
    fresh = gaps.read_gap_review(prepared['gap_review']['path'])
    assert fresh['review_binding'] != review['review_binding']
    assert fresh['chart_source_review'] == review['review_binding']
    assert prepared['painting_geometry']['review'] == review['review_binding']
    for entry, row in zip(prepared['runs'], fresh['runs']):
        fields, bound = effective_product(prepared, prepared['folder'], entry)
        assert row['product'] == bound and row['product'] != row['base_product']
        assert row['gap']['pixels'] == int(fields['gaps'].sum())
        assert row['gap']['identity'] == gaps._identity(fields['gaps'])
    verify_local_preparation(prepared, prepared['folder'])
    if native_gap:
        assert sum(row['gap']['pixels'] for row in fresh['runs']) == 0
        assert prepared['outside_frame_proof']['format'] == table.OUTSIDE_FORMAT_V2
        _, approved, frame = table.verify_version(prepared['painting_facies'])
        table.verify_outside_proof(prepared, prepared['folder'], approved, frame)
        run = next(entry for entry in prepared['runs'] if entry['run_id'] == 'two')
        effective, _ = effective_product(prepared, prepared['folder'], run)
        assert not np.any((effective['array'] == 0) & effective['inside'])
        # Removing the explicitly approved chart product must restore the real
        # original inside gap and the original outside-frame refusal.
        original_context = table._continuation_outside_context(prepared, prepared['folder'], prepared['painting_facies'])
        without_chart = deepcopy(prepared)
        without_chart.pop('chart_transfers')
        without_chart.pop('chart_review')
        original_context['preparation'] = without_chart
        with pytest.raises(ValueError, match='zero labels are INSIDE its frozen own frame'):
            table._continuation_outside_mask(prepared['folder'], run, original_context)
    _unchanged(retained)


def test_one_actual_local_check_saves_through_original_store_without_second_validation(gap_case, monkeypatch):
    before = _retained(gap_case)
    prepared, approval = _prepare(gap_case, gap_case.root / "checked_local")
    original = geometry.validate_geometry_config
    calls = []
    def real_check(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(geometry, "validate_geometry_config", real_check)
    candidate, summary, checked = _candidate(gap_case, prepared)
    report = checked.report_for(candidate)
    assert not [x for x in report["issues"] if x["severity"] == "error"]
    assert summary["runs"] == [row["run_id"] for row in prepared["runs"]]
    resolutions = _resolved(candidate, report, approval)
    saved = save_checked_local_adoption(geometry.GeometryRevisionStore(gap_case.data.project),
        candidate, checked, actor="synthetic human", reason="Reuse the reviewed local maps",
        review_resolutions=resolutions)
    assert calls == [1]
    assert saved["configuration"]["provenance"]["local_map_reuse"]["format"] == PROOF_FORMAT
    assert saved["review_resolutions"]["materialized"]["records"] == resolutions["records"]
    assert Path(saved["artifacts"]["depth_path"]).read_bytes() == Path(candidate["depth_path"]).read_bytes()
    for run, fields in saved["artifacts"]["run_geometry"].items():
        source = next(row for row in prepared["runs"] if row["run_id"] == run)
        assert Path(fields["labels_path"]).read_bytes() == (Path(prepared["folder"]) / source["labels"]["path"]).read_bytes()
    _unchanged(before)


@pytest.mark.parametrize("artifact", ["source_map", "model", "product", "continuation"])
def test_local_reuse_refuses_changed_original_inputs(gap_case, artifact):
    context = _context(gap_case)
    if artifact == "source_map":
        bound = gap_case.review["source_map"]
    elif artifact == "model":
        bound = gap_case.review["runs"][0]["target_model"]
    elif artifact == "product":
        bound = gap_case.review["runs"][0]["base_product"]
    else:
        bound = context["continuation"]
    path = Path(bound["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        prepare_local_adoption(gap_case.review["review_binding"],
            source_context=context, destination=gap_case.root / "changed_input")


@pytest.mark.parametrize("fault", ["run_order", "source", "model", "copied_map"])
def test_published_local_reuse_refuses_detached_source_or_copied_products(gap_case, fault):
    prepared, _ = _prepare(gap_case, gap_case.root / "bound_copy")
    changed = deepcopy(prepared)
    if fault == "run_order":
        changed["runs"].reverse()
    elif fault == "source":
        changed["map_version"]["sha256"] = "0" * 64
    elif fault == "model":
        changed["runs"][0]["target_model"]["sha256"] = "0" * 64
    else:
        path = Path(prepared["folder"]) / prepared["runs"][0]["labels"]["path"]
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        verify_local_preparation(changed, prepared["folder"])


def test_selected_reuse_does_not_read_an_unrelated_run_product(gap_case, monkeypatch):
    prepared, _ = _prepare(gap_case, gap_case.root / "selected")
    chosen, other = prepared["runs"][:2]
    old_other = next(row for row in gap_case.review["runs"] if row["run_id"] == other["run_id"])
    forbidden = {str((Path(prepared["folder"]) / other["labels"]["path"]).resolve()),
                 str(Path(old_other["base_product"]["path"]).resolve())}
    from darsia.fluidflower import painted_adoption_table as table
    actual = table._checked
    def selected_only(bound, **kwargs):
        assert str(Path(bound["path"]).resolve()) not in forbidden
        return actual(bound, **kwargs)
    monkeypatch.setattr(table, "_checked", selected_only)
    verify_local_selected(prepared, prepared["folder"], chosen)


@pytest.mark.parametrize("fault", ["candidate", "map_bytes", "depth_bytes"])
def test_checked_candidate_refuses_changed_results_before_store_publication(gap_case, fault):
    prepared, _ = _prepare(gap_case, gap_case.root / "stale_check")
    candidate, _, checked = _candidate(gap_case, prepared)
    if fault == "candidate":
        candidate["width_m"] *= 2
    else:
        name = candidate["depth_path"] if fault == "depth_bytes" else candidate["labels_path"]
        path = Path(name)
        stat = path.stat()
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    before = geometry.GeometryRevisionStore(gap_case.data.project).current_path.read_bytes()
    with pytest.raises(ValueError):
        save_checked_local_adoption(geometry.GeometryRevisionStore(gap_case.data.project),
            candidate, checked, actor="synthetic human", reason="This changed input must refuse")
    assert geometry.GeometryRevisionStore(gap_case.data.project).current_path.read_bytes() == before


def test_plain_report_or_boolean_cannot_be_an_actual_local_check(gap_case):
    prepared, _ = _prepare(gap_case, gap_case.root / "real_check_type")
    candidate, _, checked = _candidate(gap_case, prepared)
    with pytest.raises(TypeError):
        CheckedGeometryCandidate(None, candidate, checked.report_for(candidate), {}, {})
    for wrong in (True, checked.report_for(candidate)):
        with pytest.raises(TypeError):
            geometry.GeometryRevisionStore(gap_case.data.project).save_revision(candidate,
                actor="synthetic human", reason="A report flag is not an actual Check",
                checked_candidate=wrong, preserve_failed_artifacts=True)


def test_depth_relocation_cannot_reuse_arithmetic_for_different_bytes(gap_case):
    prepared, _ = _prepare(gap_case, gap_case.root / "depth_copy")
    candidate, _, checked = _candidate(gap_case, prepared)
    source = geometry._file_reference(candidate["depth_path"], include_full_sha256=True)
    wrong = gap_case.root / "wrong_depth_copy.npz"
    with wrong.open("xb") as stream:
        np.savez_compressed(stream, array=np.ones((2, 2)))
    changed = deepcopy(candidate)
    changed.update(depth_path=str(wrong.resolve()), depth_artifact_path=str(wrong.resolve()),
                   depth_source_reference=deepcopy(source))
    with pytest.raises(ValueError):
        checked.materialized_report(changed, source, source)
