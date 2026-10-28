"""Literal one-image policy contracts, not a substitute for actual RAW QA.

These pure cases use a neutral recorded decision and a test series' recorded
choice. They perform no acquisition decode and replace no source guard.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from darsia.fluidflower import clearwater_source_choice as choice

HERE = Path(__file__).parent
DISTORTION = [704, 684, 615, 497, 331, 126, -110, -359, -600, -824, -1034]
CA_RED = [0, 640, 512, 512, 512, 640, 640, 640, 768, 768, 768]
CA_BLUE = [0, 384, 512, 640, 640, 512, 384, 256, 256, 256, 384]
RAW = dict(path=str(HERE / "flush" / "DSC00003.ARW"), sha256="f" * 64, size_bytes=21562368)
DECISIONS = dict(clearwater_source=dict(
    record_id="decision-0474-water-photograph",
    record_sha256="6d12176b0bc921788541dc571d89ca8c84ea613b57f7880082ca5171e8f8f76d",
    actor="Test decider", confirmed_utc="2026-10-22T20:50:45Z", run_id="ac45",
    reason="Use the first flush photograph with its source-specific instrument mapping and current own geometry.",
    selected_rule="The decider selected the first flush photograph DSC00003",
    raw=RAW, copied_raw=dict(RAW, path=str(HERE / "copy" / "DSC00003.ARW")),
    camera_jpeg=dict(path=str(HERE / "flush" / "DSC00003.JPG"), sha256="e" * 64, size_bytes=9711196),
    current_model=dict(sha256="c" * 64, size_bytes=268498),
    current_profile=dict(sha256="d" * 64, size_bytes=1564493, profile_id="late", camera_wh=[5472, 3080]),
    instrument=dict(profile_id="water-photograph-display-only", intrinsics_source_profile_id="early",
                    distortion=DISTORTION, ca_red=CA_RED, ca_blue=CA_BLUE, camera_wh=[5472, 3080])))
CHOICE = choice.choice_settings(DECISIONS)


def _descriptor(name):
    return dict(path=str(HERE / name), sha256="a" * 64, size_bytes=12)


def _policy():
    return dict(format=choice.FORMAT, run_id="ac45", authority=_descriptor("answer.json"),
        parent_catalog=_descriptor("catalog.json"), parent_review=_descriptor("review.json"),
        raw=deepcopy(CHOICE["raw"]), copied_raw=deepcopy(CHOICE["copied_raw"]),
        camera_jpeg=deepcopy(CHOICE["camera_jpeg"]),
        current_model=dict(_descriptor("current-model.json"), **CHOICE["current_model"]),
        current_profile=dict(_descriptor("current-profile.json"),
            sha256=CHOICE["current_profile"]["sha256"], size_bytes=1564493,
            profile_id="late", camera_wh=[5472, 3080]),
        instrument=dict(profile_id="water-photograph-display-only",
            intrinsics_source_profile_id="early", distortion=deepcopy(DISTORTION),
            ca_red=deepcopy(CA_RED), ca_blue=deepcopy(CA_BLUE),
            camera_wh=[5472, 3080], output_grid=choice.OUTPUT_GRID),
        display_only=True, historical_geometry_used=False)


def test_exact_recorded_decision_and_instrument_copy_leave_current_geometry_profile_unchanged():
    record = json.loads((Path(__file__).parents[1] /
        "fixtures/clearwater_source_choice/q423.json").read_text(encoding="utf-8"))
    assert choice.authority_record(record, CHOICE) == record
    assert choice.authority_record([dict(id="unrelated"), record], CHOICE) == record
    changed = dict(record, decided_by="somebody else")
    with pytest.raises(ValueError, match="unchanged recorded"):
        choice.authority_record(changed, CHOICE)
    with pytest.raises(ValueError, match="unchanged recorded"):
        choice.authority_record([record, record], CHOICE)
    human = choice.human_authorization(CHOICE)
    assert human["actor"] == "Test decider" and human["selected_raw"] == CHOICE["raw"]
    assert choice._checked_human_authorization(human, _policy(), CHOICE) == human
    assert not any(word in json.dumps(human) for word in ("decision-0474", "decided_by"))
    with pytest.raises(ValueError, match="authorization changed"):
        choice._checked_human_authorization(dict(human, actor="somebody else"), _policy(), CHOICE)
    with pytest.raises(ValueError, match="authorization changed"):
        choice._checked_human_authorization(dict(human, actor=" "), _policy())
    with pytest.raises(ValueError, match="authorization changed"):
        choice._checked_human_authorization(dict(human, run_id="ac44"), _policy())
    early = dict(R=3291.1963476237365, amp=1.0033805150829112,
        cu=2735.941390570268, cv=1540.0423206018281,
        tx=0.008900502454740798, ty=-0.008331840772507504)
    parameters = dict(common=dict(sensor=dict(camera_wh=[5472, 3080], untouched="sensor"),
        lens=dict(ca_red=[1], ca_blue=[2], untouched="lens"),
        profiles=dict(early=dict(distortion=[3], coefficients=early),
                      late=dict(distortion=[4], coefficients=dict(R=7)))))
    before = deepcopy(parameters)
    result, profile_id = choice.instrument_parameters(parameters, _policy())
    assert profile_id == "water-photograph-display-only" and parameters == before
    assert result["common"]["sensor"] == before["common"]["sensor"]
    assert result["common"]["profiles"]["early"] == before["common"]["profiles"]["early"]
    assert result["common"]["profiles"]["late"] == before["common"]["profiles"]["late"]
    assert result["common"]["profiles"][profile_id] == dict(distortion=DISTORTION, coefficients=early)
    result["common"]["profiles"][profile_id]["coefficients"]["R"] = 0
    assert parameters == before
    assert result["common"]["lens"] == dict(ca_red=CA_RED, ca_blue=CA_BLUE, untouched="lens")


@pytest.mark.parametrize("damage", ["run", "raw", "copy", "jpeg", "model", "profile", "instrument", "grid"])
def test_policy_refuses_any_expansion_of_source_geometry_or_instrument_authority(damage):
    policy = _policy()
    assert choice.validate_policy_document(policy, choice=CHOICE) == policy
    if damage == "run":
        policy["run_id"] = "ac44"
    elif damage in {"raw", "copy", "jpeg"}:
        key = {"raw": "raw", "copy": "copied_raw", "jpeg": "camera_jpeg"}[damage]
        policy[key]["sha256"] = "0" * 64
    elif damage == "model":
        policy["current_model"]["sha256"] = "0" * 64
    elif damage == "profile":
        policy["current_profile"]["profile_id"] = "early"
    elif damage == "instrument":
        policy["instrument"]["distortion"][0] += 1
    else:
        policy["current_profile"]["camera_wh"] = [3080, 5472]
    with pytest.raises(ValueError):
        choice.validate_policy_document(policy, choice=CHOICE)


def test_bound_loader_contract_refuses_other_source_model_and_repurposed_profile():
    policy = _policy()
    for options in (dict(run_id="ac44"),
                    dict(model=dict(policy["current_model"], sha256="0" * 64)),
                    dict(profile=dict(policy["current_profile"], profile_id="early")),
                    dict(raw=dict(CHOICE["raw"], path=CHOICE["raw"]["path"].replace("00003", "00005")))):
        with pytest.raises(ValueError):
            choice.validate_policy_document(policy, **options)
    parameters = dict(common=dict(sensor=dict(camera_wh=[1, 1]), lens={}, profiles={}))
    with pytest.raises(ValueError, match="camera-U grid"):
        choice.instrument_parameters(parameters, policy)


def test_saved_policy_must_be_consistent_without_the_series_choice():
    policy = _policy()
    assert choice.validate_policy_document(policy) == policy
    for key, value in (("copied_raw", dict(policy["copied_raw"], size_bytes=1)),
                       ("camera_jpeg", dict(policy["camera_jpeg"], path=str(HERE / "other" / "DSC00003.JPG"))),
                       ("instrument", dict(policy["instrument"], output_grid="another grid"))):
        with pytest.raises(ValueError):
            choice.validate_policy_document(dict(policy, **{key: value}))


def test_series_choice_names_its_missing_or_inconsistent_settings():
    with pytest.raises(ValueError, match="decisions.clearwater_source"):
        choice.choice_settings({})
    broken = deepcopy(DECISIONS)
    broken["clearwater_source"]["copied_raw"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="copied photograph"):
        choice.choice_settings(broken)
    broken = deepcopy(DECISIONS)
    broken["clearwater_source"]["instrument"]["camera_wh"] = [1, 1]
    with pytest.raises(ValueError, match="instrument"):
        choice.choice_settings(broken)


def test_review_builder_exists_only_for_a_series_that_names_its_water_photograph(tmp_path, monkeypatch):
    from darsia.fluidflower import series_decisions
    (tmp_path / "series.toml").write_text('schema_version = 1\nseries_id = "finite"\n', encoding="utf-8")
    assert choice.clearwater_review_builder(tmp_path) is None
    assert choice.clearwater_review_builder(None) is None
    assert choice.clearwater_review_builder(tmp_path / "missing") is None
    monkeypatch.setattr(series_decisions, "read_series_decisions", lambda root: deepcopy(DECISIONS))
    builder = choice.clearwater_review_builder(tmp_path)
    assert builder.run_id == "ac45" and builder.choice == CHOICE
    assert choice.series_choice(tmp_path) == CHOICE
