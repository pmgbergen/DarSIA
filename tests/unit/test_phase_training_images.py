"""A phase model's training images are the labelled frames its recorded request lists."""

import hashlib
import json

import pytest

from darsia.fluidflower import phase_training_images as subject


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode("utf-8")
    path.write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def _model(tmp_path):
    first = _write(tmp_path / "labels" / "one.json", dict(decisions={
        "run1": dict(annotations=[dict(elapsed_h=1.0), dict(elapsed_h=1.0), dict(elapsed_h=2.5)]),
        "run2": dict(annotations=[])}))
    second = _write(tmp_path / "labels" / "two.json", dict(decisions={
        "run1": dict(annotations=[dict(elapsed_h=4.0)]), "run3": dict(annotations=[dict(elapsed_h=0.5)])}))
    older = _write(tmp_path / "older" / "training_request.json", dict(bundles=[
        dict(decisions=dict(snapshot_path=first["path"], sha256=first["sha256"])),
        dict(decisions=dict(snapshot_path=second["path"], sha256=second["sha256"]))], settings={}))
    model = tmp_path / "model"
    _write(model / "training_request.json", dict(inherited_training_request=older))
    return model, first


def test_labelled_images_follow_the_inherited_request_and_count_per_run(tmp_path):
    model, _first = _model(tmp_path)
    result = subject.labelled_training_images(model)
    assert result["bundles"] == 2 and not result["unreadable"]
    assert result["runs"] == {"run1": dict(times_h=[1.0, 2.5, 4.0], labels=4),
                              "run3": dict(times_h=[0.5], labels=1)}


def test_a_changed_label_file_is_named_not_counted(tmp_path):
    model, first = _model(tmp_path)
    (tmp_path / "labels" / "one.json").write_text("{}", encoding="utf-8")
    result = subject.labelled_training_images(model)
    assert list(result["runs"]) == ["run1", "run3"] and result["runs"]["run1"]["labels"] == 1
    assert len(result["unreadable"]) == 1 and "Changed since the model was trained" in result["unreadable"][0]


def test_a_model_without_labelled_bundles_says_so(tmp_path):
    _write(tmp_path / "model" / "training_request.json", dict(settings={}))
    with pytest.raises(ValueError, match="records no labelled training images"):
        subject.labelled_training_images(tmp_path / "model")
