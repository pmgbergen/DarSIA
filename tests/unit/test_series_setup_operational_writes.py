"""Original queue/publication writes spend one actual Setup metadata cap."""
import hashlib
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import job_queue, products_publication
from darsia.fluidflower.pipeline.series_setup_sink import SetupMetadataSink
from .test_series_setup_admission import _admission


def test_original_queue_and_publication_bytes_share_one_closed_allowance(tmp_path, monkeypatch):
    """Actual immutable temporaries and metadata copies retain original bytes."""
    admission = _admission(tmp_path, monkeypatch, metadata_cap=4096)
    root = admission.test_context.metadata_root
    sink = SetupMetadataSink(admission)
    source = tmp_path / "source.json"
    source.write_bytes(b'{"complete":true}\n')
    expected = dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(), bytes=source.stat().st_size)
    with sink.activate():
        job_queue.write_immutable(root / "control.json", {"operation": "apply"})
        folder = products_publication._fresh_root(root / "publication")
        target = products_publication._target(folder, "operation_result.json")
        products_publication._copy(source, expected, target)
    evidence = sink.close()
    assert target.read_bytes() == source.read_bytes()
    assert job_queue.read_json(root / "control.json") == {"operation": "apply"}
    assert evidence["actual_created_bytes"] == (root / "control.json").stat().st_size + source.stat().st_size
    assert set(sink.files) == {str(root / "control.json"), str(target)}
    for action in (lambda: sink.write_bytes(root / "later.json", b"{}"),
                   lambda: sink.mkdir(root / "later"),
                   lambda: sink.rename(target, folder / "moved.json")):
        with pytest.raises(ValueError, match="closed"):
            action()
    with pytest.raises(ValueError, match="closed"):
        with sink.activate():
            pass
    assert target.exists() and not (root / "later").exists()


def test_publication_copy_retains_prefix_and_cannot_close_an_open_writer(tmp_path, monkeypatch):
    """A failed bounded copy preserves its partial and every reserved byte."""
    admission = _admission(tmp_path, monkeypatch, metadata_cap=8)
    root = admission.test_context.metadata_root
    sink = SetupMetadataSink(admission)
    with sink.open_exclusive(root / "retained.bin") as stream:
        stream.write(b"prefix")
        with pytest.raises(ValueError, match="streams"):
            sink.close()
        assert not sink.closed
        with pytest.raises(ValueError, match="before write"):
            stream.write(b"more")
    sink.close()
    assert (root / "retained.bin").read_bytes() == b"prefix"
    assert not any((Path(row["path"]) / "released.json").exists()
                   for row in admission.bound_reservations)


def test_unchanged_bound_reservation_is_not_reopened_for_each_write(tmp_path, monkeypatch):
    """Keep the admitted SHA and original version while avoiding repeated hashes."""
    admission = _admission(tmp_path, monkeypatch, metadata_cap=4096)
    reserved = Path(admission.bound_reservations[0]["path"]) / "reserved.json"
    original_open = Path.open

    def guard(path, mode="r", *args, **kwargs):
        if path == reserved and "r" in mode:
            raise AssertionError("An unchanged admitted reservation was reopened")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    sink = SetupMetadataSink(admission)
    root = admission.test_context.metadata_root
    with sink.activate():
        for name in ("one.json", "two.json", "three.json"):
            sink.write_bytes(root / name, b"{}")
    with original_open(reserved, "ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="changed before read"):
        sink.write_bytes(root / "refused.json", b"{}")
    assert not (root / "refused.json").exists()
