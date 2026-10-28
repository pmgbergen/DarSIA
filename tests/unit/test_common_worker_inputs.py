"""Scientific finalization retains original admitted input versions and locators."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import common_worker_closure as closure
from darsia.fluidflower.pipeline.store_runtime import ResolvedInput, verify_inputs
from darsia_cache.policy import CachePolicyError


def admitted_input(tmp_path):
    """Hash one local input at the original once-per-run admission boundary."""
    path = tmp_path / "input.bin"
    path.write_bytes(b"original input")
    reference = dict(root_alias="input", relative_path=path.name, role="image",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    row = ResolvedInput(reference, path, ("source.original",))
    task = dict(run="sample", input_catalog=dict(inputs=[dict(reference=reference,
        uses=list(row.uses))]))
    return task, row, SimpleNamespace(verified=verify_inputs(task, (row,)))


def test_final_worker_input_pins_equal_the_original_admitted_versions(tmp_path, monkeypatch):
    """Recording admitted input identity performs no new stat or full byte hash."""
    from darsia.fluidflower.pipeline import store_runtime
    task, row, binding = admitted_input(tmp_path)
    def forbidden(*args, **kwargs):
        """Refuse any accidental worker-side duplicate input verification."""
        pytest.fail("Worker closure repeated original input admission")
    with monkeypatch.context() as patch:
        patch.setattr(store_runtime, "verify_inputs", forbidden)
        patch.setattr(Path, "stat", forbidden)
        rows = closure._original_input_rows(binding, (row,))
        result = closure._closed_input_bindings(task, binding, rows)
    assert result[0]["version"] == list(binding.verified.versions[str(row.path)])
    assert result[0]["path"] == str(row.path) and result[0]["reference"] == row.reference


def test_replaced_file_keeps_original_admission_version_until_controller_close(tmp_path):
    """The retried controller, rather than the worker, detects file replacement."""
    from darsia_cache.common_qualification import _reverify_source_inputs
    task, row, binding = admitted_input(tmp_path)
    original_version = binding.verified.versions[str(row.path)]
    preserved = tmp_path / "preserved-original.bin"
    row.path.rename(preserved)
    row.path.write_bytes(preserved.read_bytes())
    resolved = closure._closed_input_bindings(task, binding, (row,))
    assert resolved[0]["version"] == list(original_version)
    stores = tmp_path / "stores.toml"
    stores.write_bytes(b'[synthetic]\n')
    state = SimpleNamespace(task=task, stores_source=dict(path=str(stores),
        sha256=hashlib.sha256(stores.read_bytes()).hexdigest(), size_bytes=stores.stat().st_size))
    with pytest.raises(CachePolicyError, match="identity changed"):
        _reverify_source_inputs(state, None, resolved)
    assert preserved.is_file() and row.path.is_file()


def test_re_resolving_an_alias_cannot_substitute_an_identical_new_locator(tmp_path):
    """A new resolution cannot replace the worker's exact original locator row."""
    _, row, binding = admitted_input(tmp_path)
    second = tmp_path / "other-location.bin"
    second.write_bytes(row.path.read_bytes())
    with pytest.raises(CachePolicyError, match="resolution changed"):
        closure._original_input_rows(binding, (ResolvedInput(row.reference, second, row.uses),))
