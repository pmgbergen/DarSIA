"""Changed chart pixels through native Build planning and the P1 mask reader."""
import json
from pathlib import Path
from types import SimpleNamespace

import darsia
import numpy as np
import pytest

from darsia.fluidflower import geometry_facies, painted_native_product
from darsia.fluidflower import painting_chart_products, painting_chart_transfer
from darsia.fluidflower import painting_chart_transfer_reader
from darsia.fluidflower.mass.n1_pipeline import DataReader
from darsia.fluidflower.painted_adoption_table import _bound
from darsia.fluidflower.painted_target_integration import effective_product
from darsia.fluidflower.pipeline.local_build_chart import read_chart
from .test_painting_chart_transfer import inputs


def test_changed_chart_is_native_build_source_and_reaches_p1_physical_mask(tmp_path, monkeypatch):
    def document(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf8")
        return _bound(path)

    run = "ac23"
    folder = tmp_path / "preparation"
    (folder / run).mkdir(parents=True)
    data = inputs()
    data["base_labels"] = data["base_labels"].astype(np.int32)
    model = document(folder / "model.json", {"baseline_native_grid": {"fixture": True}})
    review = document(folder / "review.json", {"fixture": "saved chart review"})
    metadata = dict(dimensions=[.06, .08], space_dim=2, indexing="ij")
    mask = data["inside"]
    base = dict(array=data["base_labels"], inside=mask, support=mask,
        filled=~mask, continued=~mask, effective_coverage=mask, gaps=~mask)
    base_path = folder / run / "labels.npz"
    np.savez_compressed(base_path, **base, metadata=metadata)
    base_bound = dict(_bound(base_path), path=f"{run}/labels.npz")
    entry = dict(run_id=run, status="prepared", labels=base_bound,
                 reference_model=model, target_model=model)
    from darsia.fluidflower.photo_edge_continuation import FORMAT, POLICY
    continuation = document(folder / "continuation.json", dict(
        format=FORMAT, policy=POLICY, runs={run: {}}, adopted=False))
    preparation = dict(format="darsia-fluidflower-painted-map-adoption-preparation-v2",
        painting_geometry=dict(review=review), painting_facies={}, runs=[entry],
        source_map=base_bound, geometry_index=model,
        continuation=dict(file="continuation.json", sha256=continuation["sha256"],
                          size_bytes=continuation["size_bytes"]))
    data["source_binding"] = dict(review=review, painting_facies={}, target_model=model)

    class Session:
        def __init__(self, *args, **kwargs):
            pass
        def proposal(self, run_id, *, correction_mm=None):
            return painting_chart_transfer.propose_chart_transfer(
                run_id, **data, correction_mm=correction_mm)

    monkeypatch.setattr(painting_chart_transfer_reader, "ChartTransferSession", Session)
    approval = painting_chart_transfer.confirm_chart_transfer(Session().proposal(run),
        actor="Reviewer", reason="Checked the saved chart footprint")
    preparation["chart_transfers"] = painting_chart_products.prepare_chart_products(
        preparation, folder, {run: approval})
    preparation_bound = document(folder / "index.json", preparation)
    effective, chart_bound = effective_product(preparation, folder, entry)
    assert not np.array_equal(effective["array"], base["array"])

    project = tmp_path / "project"
    labels_path = project / "results/geometry/revision/runs" / run / "labels.npz"
    labels_path.parent.mkdir(parents=True)
    labels_path.write_bytes(Path(chart_bound["path"]).read_bytes())
    revision = dict(revision_id="revision", configuration=dict(
        provenance=dict(local_map_reuse={}),
        painted_adoption_table=dict(preparation=preparation_bound)))
    document(project / "results/geometry/revision/revision.json", revision)
    monkeypatch.setattr(geometry_facies, "resolve_run_geometry_artifacts",
        lambda *_args: dict(scope="run_override", labels_path=str(labels_path)))
    contract = painted_native_product.adopted_run_native_product(revision, run,
        project_root=project, registered_sources=dict(files=dict(labels=_bound(labels_path))))
    assert contract["source_labels"] == chart_bound
    dependencies = painted_native_product.native_analysis_exclusion_dependencies(contract)
    assert chart_bound in dependencies and review in dependencies

    # The original scalar writer preserves the adopted int32 categorical field.
    field_path = tmp_path / "build_labels.npz"
    darsia.ScalarImage(effective["array"], dimensions=[.06, .08]).save(field_path)
    context = SimpleNamespace(run=run, labels=darsia.imread(field_path).img, provenance={})
    chart_mask, provenance = read_chart(dict(preparation=preparation_bound),
        reader=DataReader(), context=context)
    np.testing.assert_array_equal(chart_mask, context.labels == 17)
    np.testing.assert_array_equal(chart_mask, Session().proposal(run).result["chart_mask"])
    assert provenance["approval"] == approval
    context.labels = context.labels.copy()
    context.labels[chart_mask] = 3
    with pytest.raises(ValueError, match="actual saved Build"):
        read_chart(dict(preparation=preparation_bound), reader=DataReader(), context=context)
