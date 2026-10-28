"""A larger typed gap-review budget retains the original metadata guards."""
from copy import deepcopy
import json

import pytest

from darsia.fluidflower import painted_adoption_table as table
from darsia.fluidflower import painted_analysis_exclusions as review


def _manifest(*, padding=0):
    """Literal synthetic metadata with one genuine row/component hash closure."""
    identity = "1" * 64
    row = dict(run_id="synthetic", gap=dict(pixels=1), findings=[])
    component = dict(component_id=identity, identity=dict(sha256=identity), pixels=1)
    component["review_row_sha256"] = review._digest(dict(
        context=review._digest(row), component=component))
    row["components"] = [component]
    row["review_row_sha256"] = review._row_digest(row)
    return dict(format=review.REVIEW_FORMAT, meaning=review.MEANING,
                source=dict(synthetic_padding="x" * padding), runs=[row], issues=[])


def _retain(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")), encoding="utf8")
    return dict(deepcopy(value), review_binding=table._bound(path))


@pytest.fixture(scope="module")
def large_review(tmp_path_factory):
    folder = tmp_path_factory.mktemp("large-synthetic-gap-review")
    path = folder / "gap_review.json"
    value = _manifest(padding=16 * 1024 * 1024)
    bound = _retain(path, value)["review_binding"]
    assert 16 * 1024 * 1024 < bound["size_bytes"] < 128 * 1024 * 1024
    return path, bound


def test_typed_gap_review_over_generic_limit_reads_path_and_retained_memory(large_review):
    path, bound = large_review
    loaded = review.read_gap_review(path)
    assert loaded["review_binding"] == bound
    assert len(loaded["source"]["synthetic_padding"]) == 16 * 1024 * 1024
    assert loaded["runs"][0]["components"][0]["pixels"] == 1
    assert review.read_gap_review(loaded) == loaded


def test_typed_gap_review_budget_preserves_limits_identity_json_and_row_guards(
        large_review, tmp_path):
    _, bound = large_review
    with pytest.raises(ValueError, match="limit"):
        table._checked(bound, metadata=True)

    # An actual sparse oversized file has an exact descriptor. Its JSON is
    # never parsed: the typed budget must reject it before content admission.
    oversized = tmp_path / "oversized-review.json"
    with oversized.open("xb") as stream:
        stream.seek(128 * 1024 * 1024)
        stream.write(b" ")
    too_large = _manifest()
    too_large["review_binding"] = table._bound(oversized)
    with pytest.raises(ValueError, match="limit|budget|large"):
        review.read_gap_review(too_large)
    with pytest.raises(ValueError, match="limit|budget|large"):
        review.read_gap_review(oversized)

    retained = _retain(tmp_path / "small-review.json", _manifest())
    for key, changed in (("sha256", "0" * 64),
                         ("size_bytes", retained["review_binding"]["size_bytes"] + 1)):
        altered = deepcopy(retained)
        altered["review_binding"][key] = changed
        with pytest.raises(ValueError, match="changed|binding"):
            review.read_gap_review(altered)

    for index, raw in enumerate((b'{"value":0,"value":1}', b'{"value":NaN}',
                                 b'{"value":Infinity}', b'{"value":-Infinity}')):
        path = tmp_path / f"invalid-json-{index}.json"
        path.write_bytes(raw)
        with pytest.raises(ValueError, match="Duplicate|Nonfinite"):
            review.read_gap_review(path)

    bad_row = _manifest()
    bad_row["runs"][0]["review_row_sha256"] = "0" * 64
    row_path = tmp_path / "changed-row.json"
    _retain(row_path, bad_row)
    with pytest.raises(ValueError, match="row identity"):
        review.read_gap_review(row_path)

    bad_component = _manifest()
    row = bad_component["runs"][0]
    row["components"][0]["review_row_sha256"] = "0" * 64
    row["review_row_sha256"] = review._row_digest(row)
    component_path = tmp_path / "changed-component.json"
    _retain(component_path, bad_component)
    with pytest.raises(ValueError, match="connected gap-area identity"):
        review.read_gap_review(component_path)

    altered = deepcopy(retained)
    altered["source"]["synthetic_padding"] = "changed only in memory"
    with pytest.raises(ValueError, match="in-memory gap review differs"):
        review.read_gap_review(altered)
