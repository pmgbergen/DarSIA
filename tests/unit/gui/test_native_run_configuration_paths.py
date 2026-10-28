"""The native chooser and row status must use the same ordinary run binding."""
import json
from pathlib import Path

import pytest

from darsia.fluidflower.native_darsia_app import run_configurations
from darsia.fluidflower.workflow_row_status import load_status_context


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def manifest(root, binding=None, *, include_binding=True):
    text = 'schema_version=1\n[runs]\nids=["ac28","ac52"]\n'
    if include_binding:
        text += '[runs.merged_files]\nac28=' + json.dumps(binding) + '\n'
    return put(root / "series.toml", text)


def test_explicit_relative_merged_file_wins_over_existing_default_and_binds_status(tmp_path):
    manifest(tmp_path, "config/ordinary/ac28.toml")
    old = put(tmp_path / "config/merged/ac28.toml", 'origin="older"\n')
    current = put(tmp_path / "config/ordinary/ac28.toml", 'origin="active"\n')
    choice = run_configurations(tmp_path)[0]
    assert choice[0] == "ac28" and choice[1] == current
    context = load_status_context(tmp_path, choice[0], choice[1])
    assert context.run_bound and context.config["origin"] == "active"
    assert old.read_text(encoding="utf-8") == 'origin="older"\n'


def test_absolute_merged_binding_is_used_without_joining_the_series_root(tmp_path):
    current = put(tmp_path / "external/ac28.toml", 'origin="bound"\n')
    root = tmp_path / "series"
    manifest(root, current.as_posix())
    assert run_configurations(root)[0][1] == current


def test_missing_explicit_merged_binding_never_borrows_an_older_default_or_partial(tmp_path):
    manifest(tmp_path, "config/ordinary/missing.toml")
    put(tmp_path / "config/merged/ac28.toml", 'origin="older"\n')
    put(tmp_path / "config/runs/ac28.toml", 'origin="partial"\n')
    run, path, issue = run_configurations(tmp_path)[0]
    assert run == "ac28" and path is None
    assert "config/ordinary/missing.toml" in issue


@pytest.mark.parametrize("binding", ["", False, 12])
def test_invalid_explicit_binding_is_reported_without_fallback(tmp_path, binding):
    manifest(tmp_path, binding)
    put(tmp_path / "config/merged/ac28.toml", 'origin="older"\n')
    run, path, issue = run_configurations(tmp_path)[0]
    assert run == "ac28" and path is None and "invalid" in issue


def test_absent_explicit_mapping_preserves_default_order_and_all_files(tmp_path):
    manifest(tmp_path, include_binding=False)
    paths = [put(tmp_path / f"config/merged/{run}.toml", f'run="{run}"\n')
             for run in ("ac28", "ac52")]
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    choices = run_configurations(tmp_path)
    assert [(run, path) for run, path, _ in choices] == list(zip(("ac28", "ac52"), paths))
    assert all(p.read_bytes() == value for p, value in before.items())


def test_run_partial_does_not_replace_missing_merged_settings_when_common_has_content(tmp_path):
    manifest(tmp_path, include_binding=False)
    put(tmp_path / "config/common.toml", "setting=1\n")
    put(tmp_path / "config/runs/ac28.toml", 'run="ac28"\n')
    run, path, issue = run_configurations(tmp_path)[0]
    assert run == "ac28" and path is None and "split" in issue


def test_mixed_explicit_and_default_bindings_keep_declared_run_order(tmp_path):
    manifest(tmp_path, "ordinary/ac28.toml")
    explicit = put(tmp_path / "ordinary/ac28.toml", 'run="ac28"\n')
    default = put(tmp_path / "config/merged/ac52.toml", 'run="ac52"\n')
    choices = run_configurations(tmp_path)
    assert [(run, path) for run, path, _ in choices] == [("ac28", explicit), ("ac52", default)]
    assert all(load_status_context(tmp_path, run, path).run_bound for run, path, _ in choices)


def test_directory_at_explicit_file_path_is_not_a_configuration(tmp_path):
    manifest(tmp_path, "not_a_file.toml")
    (tmp_path / "not_a_file.toml").mkdir()
    assert run_configurations(tmp_path)[0][1] is None
