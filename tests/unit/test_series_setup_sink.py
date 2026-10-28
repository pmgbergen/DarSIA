"""Actual private files/serializers under a typed whole-volume reservation.

Volume observations use the admission fixture's controlled OS seam; the actual
ledger, exclusive streams, original PNG/export/Apply paths and retained files
are used. The tiny RAW producer fixture is not a native-accuracy qualification.
"""
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower.pipeline.series_setup_sink import SetupMetadataSink, current_sink
from darsia.fluidflower import corrected_setup as setup
from .test_series_setup_admission import _admission
from .test_painted_setup_apply import named_setup, _painting, _files  # noqa: F401
from .test_painted_adoption_table import imported, synthetic_payload_admission  # noqa: F401


def test_stream_limit_preserves_the_actual_written_prefix(tmp_path, monkeypatch):
    admission = _admission(tmp_path, monkeypatch, metadata_cap=10)
    root = admission.test_context.metadata_root
    sink = SetupMetadataSink(admission)
    path = root / "retained.bin"
    with sink.open_exclusive(path) as stream:
        assert stream.write(b"1234") == 4
        with pytest.raises(ValueError, match="before write"):
            stream.write(b"5678901")
    assert path.read_bytes() == b"1234"
    assert sink.evidence()["actual_created_bytes"] == 4
    assert not sink.evidence()["task_retirement_claimed"]


def test_known_oversize_refuses_before_create_and_existing_bytes_survive(tmp_path, monkeypatch):
    admission = _admission(tmp_path, monkeypatch, metadata_cap=10, metadata_file_cap=5)
    root = admission.test_context.metadata_root
    sink = SetupMetadataSink(admission)
    path = root / "new.bin"
    with pytest.raises(ValueError, match="before write"):
        sink.write_bytes(path, b"123456")
    assert not path.exists()
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        sink.write_bytes(path, b"new")
    assert path.read_bytes() == b"original" and sink.written_bytes == 0


def test_rename_and_replacement_never_refund_created_bytes(tmp_path, monkeypatch):
    admission = _admission(tmp_path, monkeypatch, metadata_cap=10)
    root = admission.test_context.metadata_root
    sink = SetupMetadataSink(admission)
    sink.write_bytes(root / "staged.bin", b"123456")
    (root / "old.bin").write_bytes(b"original")
    sink.rename(root / "staged.bin", root / "old.bin", replace=True)
    assert (root / "old.bin").read_bytes() == b"123456"
    assert sink.written_bytes == 6
    with pytest.raises(ValueError, match="before write"):
        sink.write_bytes(root / "next.bin", b"12345")
    assert not (root / "next.bin").exists()


def test_original_png_serializer_has_exact_bytes_and_no_fd_bypass(tmp_path, monkeypatch):
    admission = _admission(tmp_path, monkeypatch, metadata_cap=1024**2)
    root = admission.test_context.metadata_root
    sink = SetupMetadataSink(admission)
    pixels = np.arange(6 * 4 * 3, dtype=np.uint8).reshape(6, 4, 3)
    original = pixels.copy()
    image = Image.fromarray(pixels)
    expected = BytesIO()
    image.save(expected, format="PNG")
    with sink.activate():
        assert current_sink() is sink
        setup._save_output_png(image, root / "actual.png")
        with sink.open_exclusive(root / "guard.bin") as stream:
            with pytest.raises(OSError, match="bounded writes"):
                stream.fileno()
            with pytest.raises(AttributeError):
                stream.truncate(100)
    assert current_sink() is None
    assert (root / "actual.png").read_bytes() == expected.getvalue()
    assert np.array_equal(pixels, original)
    assert sink.written_bytes == len(expected.getvalue())


def test_inactive_or_foreign_domain_refuses_before_any_write(tmp_path, monkeypatch):
    admission = _admission(tmp_path, monkeypatch, metadata_cap=1024)
    sink = SetupMetadataSink(admission)
    foreign = tmp_path / "not-an-output" / "blocked.json"
    with pytest.raises((ValueError, RuntimeError), match="domain|root|outside|reservation"):
        sink.mkdir(foreign.parent)
    assert not foreign.parent.exists()
    admission.release(admission.test_context.owner)
    path = admission.test_context.metadata_root / "after-release.json"
    with pytest.raises((ValueError, RuntimeError), match="bound|active|released|reservation"):
        sink.write_bytes(path, b"{}")
    assert not path.exists()


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_original_export_apply_and_all_control_files_use_the_capped_sink(named_setup, tmp_path, monkeypatch):
    data = named_setup
    admission = _admission(tmp_path / "capacity", monkeypatch, metadata_cap=4 * 1024**2,
                           metadata_roots=[data.project.root, data.series / "config"])
    sink = SetupMetadataSink(admission)
    with sink.activate():
        export = _painting(data, tmp_path)
        attempt = setup.apply_corrected_selected_run(data.project, export, rebuild_facies_by_name=True)
    evidence = sink.evidence()
    paths = {Path(row["path"]) for row in evidence["files"]}
    assert {export.destination / name for name in setup.EXPORT_FILES} <= paths
    assert {export.destination / "export.json", export.destination / "apply-claim.json",
            export.destination / "apply-complete.json", attempt} <= paths
    assert any(path.name.endswith("-before.toml") for path in paths)
    assert any(path.parent == data.series / "config/runs" for path in paths)
    assert evidence["actual_created_bytes"] == sum(row["size_bytes"] for row in evidence["files"])
    assert all(path.is_file() for path in paths)
    assert setup.require_corrected_export_for_build(data.series / "config/merged/one.toml") == export.receipt
    assert evidence["maximum_output_bytes"] == 4 * 1024**2
    assert not evidence["frame_progress_releases_bytes"]


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_real_apply_control_ceiling_refuses_before_replacing_configuration(named_setup, tmp_path, monkeypatch):
    data = named_setup
    export = _painting(data, tmp_path)
    before = {path: path.read_bytes() for path in _files(data)}
    admission = _admission(tmp_path / "capacity", monkeypatch, metadata_cap=32,
                           metadata_roots=[data.project.root, data.series / "config"])
    sink = SetupMetadataSink(admission)
    with sink.activate(), pytest.raises(ValueError, match="before write"):
        setup.apply_corrected_selected_run(data.project, export, rebuild_facies_by_name=True)
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert not (export.destination / "apply-complete.json").exists()
    assert not (export.destination / "apply-claim.json").exists()
