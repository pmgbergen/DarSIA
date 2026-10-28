"""Local synthetic metadata only. Prepared, not executed in candidate assembly."""
from copy import deepcopy
import hashlib
import json

import pytest
import toml

from darsia.fluidflower.saved_v5_source_import import (
    build_source_import, read_source_import, import_label,
)


def bound(path, value):
    raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


@pytest.fixture
def imported(tmp_path):
    receipt = bound(tmp_path / "material.json", {
        "format": "ffac-v5-native-materialization-batch-v1",
        "canonical_runs": 2,
        "canonical_mapping": [{"run": "ac28"}, {"run": "ac52"}],
        "training_eligibility": "studio-validation-only-provisional",
        "accepted_run_native_geometry_revisions": 0,
        "run_geometry_validation_only": True,
    })
    values = {
        "format": "ffac-v5-raw-full-native-training-v1",
        "materialization_receipt_path": receipt["path"],
        "materialization_receipt_sha256": receipt["sha256"],
        "ac28_light_anchor": True,
        "physical_scale_qa_pass": False,
        "bottom_gas_guard_allowed": False,
        "track1_photo_only": True,
        "provisional_studio_training": True,
        "mass_analysis_allowed": False,
        "unit_denominator": 65535.0,
    }
    request = bound(tmp_path / "training_request.json", {
        "model_id": "saved-model", "settings": {"series": ["ffac"], "v5_raw_photometry": values},
    })
    return request, values, build_source_import(request, "FF_AC")


def test_sparse_import_preserves_literal_fields_without_a_ranking_or_reference_claim(imported):
    request, values, record = imported
    assert record["format"] == "series-saved-v5-normalisation-v1"
    assert record["v5_raw_photometry"] == values
    assert record["runs"] == ["ac28", "ac52"]
    assert not set(record) & {"ranking", "majority_class", "selection", "reference_run", "suggested_reference", "ready", "done"}
    assert record["field_sources"]["v5_raw_photometry"] == {
        "source": request, "pointer": "/settings/v5_raw_photometry",
    }
    assert "ac28_light_anchor=true; no reference_run field" in import_label(record)
    assert record["materialization"]["run_geometry_validation_only"] is True
    assert record["materialization"]["accepted_run_native_geometry_revisions"] == 0


def test_source_import_refuses_changed_source_and_mismatched_series(imported):
    request, _values, record = imported
    with pytest.raises(ValueError, match="another series"):
        build_source_import(request, "other")
    altered = deepcopy(record)
    altered["v5_raw_photometry"]["mass_analysis_allowed"] = True
    with pytest.raises(ValueError, match="differs"):
        read_source_import(altered, "FF_AC", ["ac28", "ac52"])
    from pathlib import Path
    Path(request["path"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="changed"):
        read_source_import(record, "FF_AC", ["ac28", "ac52"])


def test_normalisation_page_displays_sparse_import_but_never_unlocks_training(qtbot, tmp_path, imported):
    from darsia.fluidflower.normalisation_step import NormalisationStep
    _request, _values, record = imported
    manifest = tmp_path / "series.toml"
    manifest.write_bytes(toml.dumps({"series_id": "FF_AC", "runs": {"ids": ["ac28", "ac52"]}, "phase_normalisation": record}).encode())
    before = manifest.read_bytes()
    calls = []
    page = NormalisationStep(provider=lambda *_a: calls.append("ranking-provider"))
    qtbot.addWidget(page)
    ready = []
    page.readiness_changed.connect(ready.append)
    page.set_context(tmp_path, "FF_AC")
    qtbot.waitUntil(page.can_close)
    assert page.source_import == record
    assert page.snapshot is None and page.ready is False
    assert ready and not any(ready)
    assert not calls
    assert "Imported" in page.source_label.text()
    assert "ranking" in page.status.text().lower()
    assert "65535.0" in [page.imported_fields.item(i, 1).text() for i in range(page.imported_fields.rowCount())]
    assert page.save_button.isHidden()
    page.reload()
    qtbot.waitUntil(page.can_close)
    assert manifest.read_bytes() == before



def test_runtime_series_alias_is_accepted_without_changing_saved_source_fields(imported):
    _request, _values, record = imported
    assert read_source_import(record, "ffac", ["ac28", "ac52"]) == record
    with pytest.raises(ValueError, match="another series"):
        read_source_import(record, "FF_BM", ["ac28", "ac52"])


def test_historical_source_import_survives_relocation_and_current_exclusions(
    imported, tmp_path, monkeypatch
):
    from pathlib import Path
    from darsia.fluidflower import review_storage
    _request, _values, record = imported
    original = deepcopy(record)
    locations = {}
    for index, source in enumerate(record["sources"]):
        path = tmp_path / f"archived-{index}.json"
        path.write_bytes(Path(source["path"]).read_bytes())
        locations[source["path"]] = path
    monkeypatch.setattr(review_storage, "resolve_review_path", lambda path: locations[str(path)])
    current = record["runs"][:1]
    assert read_source_import(record, record["series_id"], current) == original
    assert record == original
    for current in ([], [record["runs"][0]] * 2, ["unknown-run"]):
        with pytest.raises(ValueError, match="run list"):
            read_source_import(record, record["series_id"], current)
    locations[record["sources"][0]["path"]].write_bytes(b"{}")
    with pytest.raises(ValueError, match="changed"):
        read_source_import(record, record["series_id"], record["runs"][:1])


def test_sparse_import_reload_refuses_changed_run_order_or_nonfinite_sources(imported):
    request, _values, record = imported
    with pytest.raises(ValueError, match="run list"):
        read_source_import(record, "FF_AC", ["ac52", "ac28"])
    from pathlib import Path
    raw = b'{"settings":{"series":["ffac"],"v5_raw_photometry":NaN}}'
    path = Path(request["path"])
    path.write_bytes(raw)
    changed = dict(request, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    with pytest.raises(ValueError, match="Nonfinite"):
        build_source_import(changed, "ffac")
