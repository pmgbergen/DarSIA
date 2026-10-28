"""Preparing a water-photograph review from the series' recorded choice.

The review reader, model and file checks are finite stand-ins; the policy and
authorization documents are written and re-validated by the real code.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import clearwater_source_choice as choice
from darsia.fluidflower import painted_analysis_exclusions as exclusions
from darsia.fluidflower import painted_review_comparison as comparison

from .test_clearwater_source_choice import DECISIONS

RECORD = dict(id="decision-0474-water-photograph", utc="2026-10-22T20:50:45Z", decided_by="reviewer",
              answer="Measure the water line in the first flush photograph DSC00003.")


def _descriptor(path, data=b"finite"):
    Path(path).write_bytes(data)
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    settings = choice.choice_settings(DECISIONS)
    model = dict(path=str(tmp_path / "model.json"), **settings["current_model"])
    profile = dict(path=str(tmp_path / "profile.json"), sha256=settings["current_profile"]["sha256"],
                   size_bytes=settings["current_profile"]["size_bytes"], profile_id="late", camera_wh=[5472, 3080])
    catalog = _descriptor(tmp_path / "catalog.json", json.dumps(
        dict(format="darsia-clearwater-display-selections-v1")).encode())
    authority = _descriptor(tmp_path / "answers.json", json.dumps([RECORD]).encode())
    review = dict(format="review", runs=[dict(run_id="ac45", target_model=model), dict(run_id="ac18")],
                  display_context=dict(image_roles=dict(clearwater=catalog)),
                  review_binding=_descriptor(tmp_path / "review.json"))

    def read_review(path):
        if Path(path) == tmp_path / "source.json":
            return deepcopy(review)
        value = json.loads(Path(path).read_bytes())
        value["review_binding"] = dict(path=str(path), sha256="0" * 64, size_bytes=1)
        return value

    def checked(value, metadata=False):
        if metadata:
            return json.loads(Path(value["path"]).read_bytes())
        return None

    api = SimpleNamespace(read_review=read_review, _checked=checked,
        _model=lambda row: (SimpleNamespace(model=dict(camera_profile=deepcopy(profile))), None, None),
        validate_display_context=lambda context, review: context,
        _json=comparison._json, _same=comparison._same, _descriptor=comparison._descriptor)
    monkeypatch.setattr(choice, "_api", lambda: api)
    monkeypatch.setattr(exclusions, "_review_metadata", lambda fresh: fresh)
    builder = choice.ClearwaterReviewBuilder(settings)
    result = builder(tmp_path / "source.json", tmp_path / "fresh.json", authority=authority)
    return SimpleNamespace(tmp_path=tmp_path, settings=settings, result=result, authority=authority,
                           builder=builder, source=tmp_path / "source.json")


def test_the_series_choice_is_written_as_plain_authorization_and_policy(prepared):
    folder = prepared.tmp_path / "fresh_water_source"
    authorization = (folder / "authority.json").read_bytes()
    assert json.loads(authorization) == choice.human_authorization(prepared.settings)
    assert b"decision-0474" not in authorization and b"decided_by" not in authorization
    policy = json.loads((folder / "source_policy.json").read_bytes())
    assert policy["run_id"] == prepared.builder.run_id == "ac45"
    assert policy["raw"] == prepared.settings["raw"] and policy["copied_raw"] == prepared.settings["copied_raw"]
    assert policy["instrument"] == dict(prepared.settings["instrument"], output_grid=choice.OUTPUT_GRID)
    assert choice.validate_policy_document(policy, choice=prepared.settings) == policy
    assert prepared.result["run_id"] == "ac45" and prepared.result["clearwater_selection_changed"] is True
    fresh = json.loads((prepared.tmp_path / "fresh.json").read_bytes())
    assert fresh["display_context"]["image_roles"]["clearwater"] == prepared.result["source_policy"]


def test_a_changed_decision_record_or_another_run_is_refused(prepared):
    changed = _descriptor(prepared.tmp_path / "changed-answers.json",
                          json.dumps([dict(RECORD, answer="Another answer.")]).encode())
    with pytest.raises(ValueError, match="unchanged recorded"):
        prepared.builder(prepared.source, prepared.tmp_path / "second.json", authority=changed)
    other = choice.ClearwaterReviewBuilder(dict(prepared.settings, run_id="ac44"))
    with pytest.raises(ValueError, match="ac44"):
        other(prepared.source, prepared.tmp_path / "third.json", authority=prepared.authority)
    assert not (prepared.tmp_path / "second.json").exists() and not (prepared.tmp_path / "third.json").exists()
