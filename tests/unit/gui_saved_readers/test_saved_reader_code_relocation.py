"""Current-package saved-code identity; no old checkout, numerics or data migration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from darsia.fluidflower import series_native_context as context


def binding(path, raw):
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "new-clone/src/darsia/fluidflower"
    root.mkdir(parents=True)
    document = dict(numeric=[23.859, .6513, True], saved_review_metadata={})
    for section, relative in context._SAVED_READER_MODULES.items():
        local = root / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        raw = ("# exact synthetic saved-reader code: " + relative + "\n").encode()
        if not local.exists(): local.write_bytes(raw)
        original = "C:/nonexistent-old-clone/src/darsia/fluidflower/" + relative
        document["saved_review_metadata"][section] = dict(current_code_source=binding(original, raw),
            producer_identity=dict(sha256="b" * 64), historical={"unchanged": [1, 2.5, True]})
    monkeypatch.setattr(context, "__file__", str(root / "series_native_context.py"))
    return root, document


def test_no_saved_code_document_keeps_values_and_has_no_bindings():
    original = dict(numeric=[1, 1.25, True])
    actual, evidence = context._bind_current_saved_readers(original)
    assert actual == original and actual is not original and evidence == ()


@pytest.mark.parametrize("section", list(context._SAVED_READER_MODULES))
def test_each_retained_reader_moves_only_its_runtime_path_and_keeps_original_identity(package, section):
    root, original = package
    before = deepcopy(original)
    actual, evidence = context._bind_current_saved_readers(original)
    local = (root / context._SAVED_READER_MODULES[section]).resolve()
    item = actual["saved_review_metadata"][section]
    old = before["saved_review_metadata"][section]
    assert item["current_code_source"] == dict(old["current_code_source"], path=str(local))
    assert item["producer_identity"] == old["producer_identity"]
    assert item["historical"] == old["historical"] and actual["numeric"] == original["numeric"]
    assert len(evidence) == 5 and original == before
    row = next(row for row in evidence if row["section"] == section)
    assert row["declared"] == old["current_code_source"] and row["actual"] == item["current_code_source"]


@pytest.mark.parametrize("invalid", ["sha", "size", "bool_size", "wrong_role_path", "unknown_role", "extra_field", "wrong_table"])
def test_substantive_or_unclassified_code_identity_is_refused_without_mutating_document(package, invalid):
    _, value = package
    section = value["saved_review_metadata"]["physics_profiles"]
    if invalid == "sha": section["current_code_source"]["sha256"] = "0" * 64
    elif invalid == "size": section["current_code_source"]["size_bytes"] += 1
    elif invalid == "bool_size": section["current_code_source"]["size_bytes"] = True
    elif invalid == "wrong_role_path": section["current_code_source"]["path"] = "C:/other/phase_saved_review_metadata.py"
    elif invalid == "unknown_role": value["saved_review_metadata"]["unknown"] = deepcopy(section)
    elif invalid == "extra_field": section["current_code_source"]["unbound"] = True
    else: value["saved_review_metadata"] = []
    before = deepcopy(value)
    with pytest.raises(ValueError): context._bind_current_saved_readers(value)
    assert value == before


def test_later_code_mutation_is_not_accepted_as_relocation(package):
    root, value = package
    context._bind_current_saved_readers(value)
    (root / "phase_saved_review_metadata.py").write_bytes(b"# changed reader\n")
    with pytest.raises(ValueError, match="implementation changed"):
        context._bind_current_saved_readers(value)


def test_original_saved_profile_guard_accepts_runtime_binding_and_rejects_old_path(monkeypatch):
    from darsia.fluidflower import phase_saved_review_metadata as saved
    from darsia.fluidflower.native.config import configure_parameters, reset_parameters
    local = Path(saved.__file__).resolve()
    original = dict(saved_review_metadata=dict(physics_profiles=dict(
        current_code_source=binding("C:/nonexistent-old-clone/src/darsia/fluidflower/phase_saved_review_metadata.py", local.read_bytes()),
        roles={"unit-ui": {"marker": "unchanged saved metadata"}})))
    frozen = deepcopy(original)
    monkeypatch.setattr(saved, "_SAVED_PROFILE_FACTORIES", {"unit-ui": lambda row: deepcopy(row)})
    monkeypatch.setattr(saved, "_SAVED_PROFILE_CACHE", {})
    token = configure_parameters(original)
    try:
        with pytest.raises(ValueError, match="not explicitly bound"):
            saved.saved_physics_profile("unit-ui")
    finally: reset_parameters(token)
    runtime, evidence = context._bind_current_saved_readers(original)
    token = configure_parameters(runtime)
    try:
        assert saved.saved_physics_profile("unit-ui") == {"marker": "unchanged saved metadata"}
    finally: reset_parameters(token)
    assert original == frozen and len(evidence) == 1
    assert evidence[0]["actual"]["path"] == str(local)


def test_series_document_raw_SHA_is_still_required_before_runtime_relocation(tmp_path):
    folder = tmp_path / "series"; folder.mkdir()
    raw = b'{"numeric": 7}\n'
    (folder / "native.json").write_bytes(raw)
    (folder / "series.toml").write_text('[native]\nconfiguration="native.json"\nsha256="' + "0" * 64 + '"\n', encoding="utf8")
    with pytest.raises(ValueError, match="Source artifact changed"):
        context.read_series_native_binding(folder)
    assert (folder / "native.json").read_bytes() == raw
