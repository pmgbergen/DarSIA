"""Setup shares a complete reviewed-map proof, retaining real selected checks."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import local_map_adoption as local
from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import painted_adoption_table as table
from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower import painted_target_integration as targets
from darsia.fluidflower import painting_chart_transfer as chart
from darsia.fluidflower import painting_chart_transfer_reader as reader
from darsia.fluidflower import photo_edge_continuation as continuation
from darsia.fluidflower import setup_source_reads as setup_reads
from darsia.fluidflower.layer_painting_fixtures import read_underlay
from darsia.fluidflower.rig_point_alignment import effective_model_matrix

from .test_painted_adoption_table import prepared_painting
from .test_painted_analysis_exclusions import (  # noqa: F401
    ACTION, _resolved, imported, registered_setup, synthetic_payload_admission,
)


@pytest.fixture
def chart_preparation(registered_setup, tmp_path, monkeypatch):
    """Create two real approved effective charts through the original APIs."""
    original, _, drawing, *_ = prepared_painting(
        registered_setup, tmp_path, fixture=True, continuation=True
    )
    review = gaps.build_gap_review(original, original["folder"])
    source = read_underlay(review["preparation_context"]["source_fixture_underlay"])
    material = source.analysis_layers.copy()
    material[source.fixture_mask] = source.drawn_layers[source.fixture_mask]
    centres = np.asarray([[x, -y] for y in range(6) for x in range(4)], np.float64)
    rows = {row["run_id"]: row for row in review["runs"]}

    class Session:
        proposal_format = chart.DETECTED_FORMAT

        def __init__(self, *args, **kwargs):
            pass

        def proposal(self, run, *, correction_mm=None, proposal_format=chart.DETECTED_FORMAT):
            row = rows[run]
            fields = gaps._arrays(row["product"])
            model = gaps._checked(row["target_model"], metadata=True)
            native = np.asarray(model["baseline_native_grid"]["H_baseline_camera_u_to_rig_corrected_native"])
            metric = np.linalg.inv(effective_model_matrix(model, review["geometry_index"])) @ np.linalg.inv(native)
            return chart.propose_chart_transfer(
                run, fields["array"], fields["inside"], fields["array"] == 8,
                material, source.fixture_mask, np.linalg.inv(row["native_to_painting_h"]), metric,
                centres, centres, dict(review=review["review_binding"], painting_facies=review["source"]["painting_facies"],
                    target_model=row["target_model"]), chart_id=8,
                known_ids=[layer.layer_id for layer in drawing.layers], correction_mm=correction_mm,
                proposal_format=proposal_format,
            )

    monkeypatch.setattr(reader, "ChartTransferSession", Session)
    approvals = {run: chart.confirm_chart_transfer(
        Session().proposal(run), actor="synthetic human", reason="Explicit fixture review"
    ) for run in rows}
    return local.prepare_local_adoption(
        review["review_binding"], source_context=review["preparation_context"],
        destination=tmp_path / "local_charts", chart_transfers=approvals,
    )


def observe_original_proof(monkeypatch):
    original_proof, original_product = local._verify_local_preparation, targets.effective_product
    proofs, products = [], []
    inside_proof = False

    def proof(preparation, folder):
        nonlocal inside_proof
        proofs.append(True)
        inside_proof = True
        try:
            return original_proof(preparation, folder)
        finally:
            inside_proof = False

    def product(preparation, folder, row):
        if inside_proof:
            products.append(row["run_id"])
        return original_product(preparation, folder, row)

    monkeypatch.setattr(local, "_verify_local_preparation", proof)
    monkeypatch.setattr(targets, "effective_product", product)
    return proofs, products


def read_selected(preparation, entry):
    """Keep the same selected-map check and common proof as a Store read."""
    local.verify_local_selected(preparation, preparation["folder"], entry)
    return continuation.verify_native_proof(preparation, preparation["folder"])


def test_two_selected_reads_replay_local_charts_once_only_within_setup(
        chart_preparation, monkeypatch):
    preparation = chart_preparation
    proofs, products = observe_original_proof(monkeypatch)
    with setup_reads.reuse_setup_source_reads():
        first = read_selected(preparation, preparation["runs"][0])
        expected = deepcopy(first)
        first[0]["sha256"] = "0" * 64
        assert read_selected(preparation, preparation["runs"][1]) == expected
        assert len(proofs) == 1
        assert products == ["one", "two"]
        with table._FILES.fresh():
            assert read_selected(preparation, preparation["runs"][0]) == expected
        assert len(proofs) == 2 and products == ["one", "two"] * 2
        # A forced pass is not retained as an ordinary reusable success.
        assert read_selected(preparation, preparation["runs"][1]) == expected
        assert len(proofs) == 3
        with setup_reads.reuse_setup_source_reads():
            assert read_selected(preparation, preparation["runs"][0]) == expected
        assert read_selected(preparation, preparation["runs"][1]) == expected
        assert len(proofs) == 4
    with setup_reads.reuse_setup_source_reads():
        assert read_selected(preparation, preparation["runs"][0]) == expected
    assert len(proofs) == 5
    for entry in preparation["runs"]:
        assert read_selected(preparation, entry) == expected
    assert len(proofs) == 7 and products == ["one", "two"] * 7


def test_direct_local_preparation_calls_share_the_original_body_with_native_reads(
        chart_preparation, monkeypatch):
    preparation = chart_preparation
    proofs, products = observe_original_proof(monkeypatch)
    with setup_reads.reuse_setup_source_reads():
        first = local.verify_local_preparation(preparation, preparation["folder"])
        assert local.verify_local_preparation(preparation, preparation["folder"]) == first
        assert continuation.verify_native_proof(preparation, preparation["folder"]) == first
        assert len(proofs) == 1 and products == ["one", "two"]
    assert local.verify_local_preparation(preparation, preparation["folder"]) == first
    assert len(proofs) == 2 and products == ["one", "two"] * 2


def test_two_genuine_selected_store_reads_share_the_original_local_proof(
        chart_preparation, registered_setup, monkeypatch):
    preparation = chart_preparation
    declaration = gaps.make_facies_source_acquisition(preparation, "indicator", **ACTION)
    candidate, _, checked = local.check_local_adoption(
        registered_setup.revision, preparation["folder"], facies_source_acquisition=declaration
    )
    report = checked.report_for(candidate)
    resolutions = _resolved(candidate, report, None)
    revision = local.save_checked_local_adoption(
        geometry.GeometryRevisionStore(registered_setup.project), candidate, checked,
        actor="synthetic human", reason="Explicit reviewed chart adoption",
        review_resolutions=resolutions,
    )
    proofs, products = observe_original_proof(monkeypatch)
    published = json.loads((Path(preparation["folder"]) / "index.json").read_bytes())
    with setup_reads.reuse_setup_source_reads():
        for entry in preparation["runs"]:
            selected = sources.registered_run_sources(
                registered_setup.project.root, revision, entry["run_id"]
            )
            assert selected["run_id"] == entry["run_id"]
        assert len(proofs) == 1
        assert products == ["one", "two"]
        local.verify_local_preparation(published, preparation["folder"])
        assert len(proofs) == 1


def test_changed_consumed_source_replays_original_local_proof_and_rejects(
        chart_preparation, monkeypatch):
    preparation = chart_preparation
    proofs, products = observe_original_proof(monkeypatch)
    with setup_reads.reuse_setup_source_reads():
        continuation.verify_native_proof(preparation, preparation["folder"])
        path = Path(preparation["folder"]) / preparation["source_map"]["path"]
        path.write_bytes(path.read_bytes() + b"changed source")
        with pytest.raises(ValueError, match="changed"):
            continuation.verify_native_proof(preparation, preparation["folder"])
        with pytest.raises(ValueError, match="changed"):
            continuation.verify_native_proof(preparation, preparation["folder"])
    assert len(proofs) == 3
    assert products == ["one", "two"]


def test_forced_fresh_proof_rehashes_actual_bytes_with_colliding_stamps(
        chart_preparation, monkeypatch):
    preparation = chart_preparation
    proofs, _ = observe_original_proof(monkeypatch)
    with setup_reads.reuse_setup_source_reads() as sources:
        continuation.verify_native_proof(preparation, preparation["folder"])
        path = Path(preparation["folder"]) / preparation["source_map"]["path"]
        for cache in (table._FILES, sources):
            original_stamp, old_stamp = cache.stamp, cache.stamp(path)
            monkeypatch.setattr(cache, "stamp", lambda value, original=original_stamp, old=old_stamp:
                old if Path(value) == path else original(value))
        path.write_bytes(path.read_bytes() + b"changed source")
        with table._FILES.fresh():
            with pytest.raises(ValueError, match="changed"):
                continuation.verify_native_proof(preparation, preparation["folder"])
    assert len(proofs) == 2


def test_preparation_identity_and_metadata_ceiling_bound_operation_reuse(tmp_path, monkeypatch):
    path = tmp_path / "source.json"
    path.write_bytes(b"exact metadata")
    bound = table._bound(path)
    calls = []

    def proof(preparation, folder):
        calls.append(preparation["identity"])
        return [table._checked(bound)]

    monkeypatch.setattr(setup_reads, "_MAXIMUM_PROOFS", 1)
    with setup_reads.reuse_setup_source_reads():
        for identity in (1, 1, 2, 1):
            setup_reads.reuse_local_preparation_proof(dict(identity=identity), tmp_path, reader=proof)
        assert calls == [1, 2, 1]
        entries = setup_reads._PROOFS.get()["entries"]
        assert len(entries) == 1
        assert sum(row["cost"] for row in entries.values()) <= setup_reads._MAXIMUM_PROOF_METADATA_BYTES
    assert not entries
    monkeypatch.setattr(setup_reads, "_MAXIMUM_PROOF_METADATA_BYTES", 1)
    with setup_reads.reuse_setup_source_reads():
        for _ in range(2):
            setup_reads.reuse_local_preparation_proof(dict(identity=3), tmp_path, reader=proof)
        assert not setup_reads._PROOFS.get()["entries"]
    assert calls == [1, 2, 1, 3, 3]
