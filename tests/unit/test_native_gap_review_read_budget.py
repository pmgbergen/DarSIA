"""Build planning uses the typed review budget without binary/foreign replay."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import painted_adoption_table as table
from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower import photo_edge_continuation as continuation
from darsia.fluidflower import rig_raw_build


@pytest.mark.parametrize("route", ["analysis-exclusion", "target-correction", "shared-review-cache"])
def test_raw_request_catalogue_reads_large_typed_review_and_preserves_admission_and_identity(
    tmp_path, monkeypatch, route
):
    """Only selected JSON is read; DATA declarations remain literal, not accepted."""
    def binary(name):
        return dict(path=str(tmp_path / name), sha256="1" * 64, size_bytes=1)

    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")), encoding="utf8")
        return table._bound(path)

    def local(value):
        return dict(value, path=str(Path(value["path"]).relative_to(tmp_path)))

    selected_raw, selected_labels = binary("one/selected.ARW"), binary("one/labels.npz")
    model = save("one/model.json", dict(reference_image=selected_raw))
    scientific_contract = dict(geometry_index=binary("geometry_index.json"),
        source_model=dict(descriptor=model, photograph=selected_raw),
        target_model=dict(descriptor=model, photograph=selected_raw))
    sidecar = save("continuation.json", dict(format=continuation.FORMAT, policy=continuation.POLICY,
        runs={"one": scientific_contract}, adopted=False))
    row = dict(run_id="one", gap=dict(pixels=0), findings=[], components=[],
        product=selected_labels, reference_model=model, target_model=model,
        native_photo=binary("one/saved-native-rgb.npz"), contract=scientific_contract)
    row["review_row_sha256"] = gaps._row_digest(row)
    value = dict(format=gaps.REVIEW_FORMAT, meaning=gaps.MEANING, runs=[row], issues=[],
        source=dict(painting_facies=dict(files={"map.npz": binary("saved-map.npz")}),
            geometry_index=binary("geometry_index.json"), padding="x" * (16 * 1024 * 1024)),
        source_map=binary("source-map.npz"),
        display_context=dict(historical_display_only=binary("never-read/historical-series.toml")))
    review = save("gap_review.json", value)
    assert 16 * 1024 * 1024 < review["size_bytes"] < gaps.MAX_REVIEW_BYTES
    entry = dict(run_id="one", labels=local(selected_labels), reference_model=model, target_model=model)
    preparation = dict(runs=[entry], painting_facies=deepcopy(value["source"]["painting_facies"]),
        geometry_index=deepcopy(value["source"]["geometry_index"]), source_map=local(value["source_map"]),
        continuation=dict(file="continuation.json", sha256=sidecar["sha256"], size_bytes=sidecar["size_bytes"]))
    if route in {"analysis-exclusion", "shared-review-cache"}:
        approval = save("gap-approval.json", dict(review=review))
        preparation["analysis_exclusions"] = dict(approval=approval,
            runs={"one": dict(artifact=binary("one/exclusion-mask.npz"))})
    if route in {"target-correction", "shared-review-cache"}:
        version = save("target-version.json", dict(product=binary("one/target-product.npz"), basis=dict(review=review)))
        preparation["target_corrections"] = dict(runs={"one": dict(artifact=local(binary("one/effective.npz")),
            bindings=[dict(version=version, approval=binary("target-approval.json"))])})
    contract = dict(preparation=save("index.json", preparation), run_id="one")
    admitted, read = set(), []

    class Reader:
        def path(self, bound):
            assert "never-read" not in Path(bound["path"]).parts
            admitted.add(bound["path"])
            return Path(bound["path"])

    original_checked = gaps._checked

    def observe_checked(bound, *, metadata=False):
        assert bound["path"] in admitted, "JSON was read before original-root admission"
        assert Path(bound["path"]).suffix == ".json", "Planning hashed a binary descriptor"
        read.append((bound["path"], metadata))
        return original_checked(bound, metadata=metadata)

    monkeypatch.setattr(gaps, "_checked", observe_checked)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("Planning decoded an archive"))
    with pytest.raises(ValueError, match="limit"):
        table._checked(review, metadata=True)  # The generic budget remains unchanged.
    result = rig_raw_build._native_request_dependencies(dict(native_product=contract), Reader())
    assert review in result and model in result and selected_labels in result and selected_raw in result
    assert all(bound["path"] in admitted for bound in result)
    assert (review["path"], False) in read and (review["path"], True) not in read
    assert all("never-read" not in Path(bound["path"]).parts for bound in result)
    before = list(read)
    assert rig_raw_build._native_request_dependencies({}) == []
    assert gaps.native_analysis_exclusion_dependencies(None, Reader()) == [] and read == before
    if route == "shared-review-cache":
        assert read.count((review["path"], False)) == 1, "C and target replay must reuse the same typed review"
        preparation["target_corrections"]["runs"]["one"]["bindings"][0]["version"] = review
        contract["preparation"] = save("index.json", preparation)
        with pytest.raises(ValueError, match="limit"):
            rig_raw_build._native_request_dependencies(dict(native_product=contract), Reader())
        assert (review["path"], True) in read, "A typed cache entry must not bypass a generic metadata read"
    with Path(review["path"]).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="changed"):
        rig_raw_build._native_request_dependencies(dict(native_product=contract), Reader())
