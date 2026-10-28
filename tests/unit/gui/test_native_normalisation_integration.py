"""Boundary tests for the native lazy adapter; no original Studio is started."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


from darsia.fluidflower import normalisation_step as normalisation
from darsia.fluidflower import native_phase_subpage as adapter
from .test_native_phase_steps import FakeStudio


@pytest.fixture
def native_environment(monkeypatch):
    """Keep normalisation live while recording unrelated display-reader setup."""
    import darsia.fluidflower.native_phase_steps as steps_module
    import darsia.fluidflower.raw_production_review as production_review
    import darsia.fluidflower.ui.model_training as studio_module

    reader_installations = []

    def record_reader_installation():
        """Record the lazy adapter hook without opening production review sources."""
        reader_installations.append(True)

    monkeypatch.setattr(production_review, "install_readers", record_reader_installation)
    # Reuse the existing B1 test fixture, with its inert methods and widgets.
    # No production SharedPhaseModelTrainingPage instance is constructed.
    monkeypatch.setattr(studio_module, "SharedPhaseModelTrainingPage", FakeStudio)
    monkeypatch.setitem(
        sys.modules, "darsia.fluidflower.normalisation_step", normalisation
    )
    return SimpleNamespace(steps=steps_module, reader_installations=reader_installations)


def _series(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    (root / "series.toml").write_text(
        f'series_id = "{name}"\n[runs]\nids = ["one"]\n', encoding="utf-8"
    )
    return root


def test_constructor_and_context_binding_are_lazy_no_start_metadata_reads(
    qtbot, monkeypatch, native_environment, tmp_path
):
    calls = []
    monkeypatch.setattr(
        normalisation, "read_series_context", lambda *args: calls.append(args)
    )
    page = adapter.NativePhaseModelStudioSubpage("A", tmp_path / "not opened")
    qtbot.addWidget(page)
    page.bind_context("B", tmp_path / "also not opened")
    assert page.studio is None and page.native_steps is None
    assert calls == []
    assert native_environment.reader_installations == []


def test_lazy_load_creates_one_default_and_binds_real_readiness(
    qtbot, native_environment, tmp_path
):
    root = _series(tmp_path, "A")
    original = (root / "series.toml").read_bytes()
    page = adapter.NativePhaseModelStudioSubpage("A", root)
    qtbot.addWidget(page)
    page._load_studio()
    assert not page._load_failed
    assert native_environment.reader_installations == [True]
    widget = page.native_steps.normalisation
    assert isinstance(widget, normalisation.NormalisationStep)
    assert widget.series_root == root and widget.series_id == "A"
    assert widget.ready is False and page.native_steps._ready is False
    qtbot.waitUntil(lambda: widget.can_close() and not widget.metadata_busy())
    assert "Missing" in widget.status.text()
    widget._emit_ready(True)
    assert page.native_steps._ready is True
    page._load_studio()
    assert page.native_steps.normalisation is widget
    assert native_environment.reader_installations == [True]
    assert (root / "series.toml").read_bytes() == original


def test_switch_series_clears_saved_source_and_closes_gate(
    qtbot, native_environment, tmp_path
):
    first, second = _series(tmp_path, "A"), _series(tmp_path, "B")
    page = adapter.NativePhaseModelStudioSubpage("A", first)
    qtbot.addWidget(page)
    page._load_studio()
    widget = page.native_steps.normalisation
    widget.source_folder = tmp_path / "previous explicit source"
    widget._emit_ready(True)
    page.bind_context("B", second)
    assert widget.series_root == second and widget.series_id == "B"
    assert widget.source_folder is None and widget.snapshot is None
    assert widget.ready is False and page.native_steps._ready is False
    qtbot.waitUntil(lambda: widget.can_close() and not widget.metadata_busy())
    assert native_environment.reader_installations == [True]


def test_explicit_readonly_preview_widget_is_not_replaced(
    qtbot, native_environment, tmp_path
):
    root = _series(tmp_path, "A")
    original = (root / "series.toml").read_bytes()
    preview = normalisation.NormalisationStep(allow_save=False)
    page = adapter.NativePhaseModelStudioSubpage(
        "A", root, step_options={"normalisation": preview}
    )
    qtbot.addWidget(page)
    page._load_studio()
    assert not page._load_failed
    assert native_environment.reader_installations == [True]
    assert page.native_steps.normalisation is preview
    preview.set_source(tmp_path / "explicit missing evidence")
    qtbot.waitUntil(lambda: preview.can_close() and not preview.metadata_busy())
    assert not preview.ready and not preview.save_button.isEnabled()
    assert (root / "series.toml").read_bytes() == original


def test_explicit_none_opt_out_does_not_construct_default(
    qtbot, monkeypatch, native_environment, tmp_path
):
    def forbidden_default(*args, **kwargs):
        raise AssertionError("Explicit normalisation=None must be preserved")

    monkeypatch.setattr(normalisation, "NormalisationStep", forbidden_default)
    page = adapter.NativePhaseModelStudioSubpage(
        "A", tmp_path, step_options={"normalisation": None}
    )
    qtbot.addWidget(page)
    page._load_studio()
    assert not page._load_failed
    assert native_environment.reader_installations == [True]
    assert page.native_steps.normalisation is None and page.native_steps._ready is False
