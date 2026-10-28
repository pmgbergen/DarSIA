"""Series decision settings: who decided and which choices apply come from series.toml."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from darsia.fluidflower import series_decisions as decisions


def _series(root, text):
    root.mkdir(parents=True, exist_ok=True)
    (root / "series.toml").write_text('schema_version = 1\nseries_id = "finite"\n' + text, encoding="utf-8")
    return root


def test_decision_settings_are_read_from_the_series_manifest(tmp_path):
    root = _series(tmp_path / "series", '[decisions]\ndecider = "Test decider"\n'
                   'gap_review_decision_ref = "series-gap-review"\n'
                   '[decisions.calibration_protocol]\nsha256 = "' + "a" * 64 + '"\nsize_bytes = 3\n')
    value = decisions.read_series_decisions(root)
    assert value["decider"] == "Test decider"
    assert decisions.series_decisions_from_bytes((root / "series.toml").read_bytes()) == value
    assert decisions.decision_setting(value, "calibration_protocol") == dict(sha256="a" * 64, size_bytes=3)
    assert decisions.text_setting(value, "decider") == "Test decider"
    assert decisions.configured_decider(root) == decisions.series_actor(root) == "Test decider"
    assert decisions.gap_review_decision_ref(root) == "series-gap-review"
    value["decider"] = "changed copy"
    assert decisions.read_series_decisions(root)["decider"] == "Test decider"


def test_a_missing_setting_makes_the_feature_unavailable_with_its_key(tmp_path):
    root = _series(tmp_path / "series", "")
    assert decisions.read_series_decisions(root) == {}
    with pytest.raises(decisions.DecisionSettingUnavailable, match="decisions.calibration_protocol"):
        decisions.decision_setting(decisions.read_series_decisions(root), "calibration_protocol")
    assert decisions.configured_decider(root) is None
    assert decisions.configured_decider(tmp_path / "no-series") is None
    assert decisions.gap_review_decision_ref(root) is None
    assert decisions.gap_review_decision_ref(None) is None
    blank = _series(tmp_path / "blank", '[decisions]\ndecider = " "\n')
    with pytest.raises(decisions.DecisionSettingUnavailable, match="decider"):
        decisions.configured_decider(blank)
    with pytest.raises(decisions.DecisionSettingUnavailable, match="nonempty"):
        decisions.text_setting(dict(decider=7), "decider")
    wrong = _series(tmp_path / "wrong", 'decisions = "not a table"\n')
    with pytest.raises(ValueError, match="must be a table"):
        decisions.read_series_decisions(wrong)


def test_actor_without_a_named_decider_is_the_signed_in_user(tmp_path, monkeypatch):
    from darsia.fluidflower import series_geometry
    root = _series(tmp_path / "series", "")
    monkeypatch.setattr(series_geometry, "default_actor", lambda: "signed-in user")
    assert decisions.series_actor(root) == decisions.series_actor(None) == "signed-in user"
    monkeypatch.setattr(series_geometry, "default_actor", lambda: "")
    with pytest.raises(decisions.DecisionSettingUnavailable, match="decider"):
        decisions.series_actor(None)


def test_project_series_root_prefers_the_linked_series(tmp_path):
    linked = SimpleNamespace(manifest=dict(tools=dict(native_series=dict(series_root=str(tmp_path / "series")))),
                             root=tmp_path / "package")
    assert decisions.project_series_root(linked) == tmp_path / "series"
    own = SimpleNamespace(manifest=dict(tools={}), root=tmp_path / "package")
    assert decisions.project_series_root(own) == tmp_path / "package"
    assert decisions.project_series_root(SimpleNamespace(manifest=None, root=None)) is None


def test_pins_and_times_are_exact():
    raw = b"exact bytes"
    pin = dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    assert decisions.checked_pin(pin) == pin and decisions.matches_pin(raw, pin)
    assert not decisions.matches_pin(raw + b" ", pin) and not decisions.matches_pin("text", pin)
    for broken in (dict(pin, size_bytes=0), dict(pin, sha256="A" * 64), dict(pin, extra=1), [pin]):
        with pytest.raises(decisions.DecisionSettingUnavailable):
            decisions.checked_pin(broken)
    assert decisions.utc_text("2026-10-21T10:51:53Z") == "2026-10-21T10:51:53Z"
    for broken in ("2026-10-21 10:51", "2026-10-21T10:51:53+01:00", "yesterday", None):
        with pytest.raises(ValueError):
            decisions.utc_text(broken)


def test_decision_records_are_found_by_id_and_pinned_by_their_canonical_digest():
    record = dict(id="decision-1", utc="2026-10-21T10:51:53Z", decided_by="reviewer", answer="Yes.")
    digest = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    assert decisions.record_digest(record) == digest
    assert decisions.recorded_decision(record) is record
    assert decisions.find_decision_record([dict(id="other"), record], "decision-1", digest=digest) == record
    assert decisions.find_decision_record(record, "decision-1") == record
    for rows in ([record, record], [dict(record, answer="No.")], [dict(id="other")]):
        with pytest.raises(ValueError, match="unchanged recorded"):
            decisions.find_decision_record(rows, "decision-1", digest=digest)
    for broken in (dict(record, decided_by=" "), dict(record, utc=None), "record"):
        with pytest.raises(ValueError, match="id, decider and time"):
            decisions.recorded_decision(broken)

