import hashlib
import pytest

from darsia.fluidflower.adoption_check_progress import observe_check_progress
from darsia.fluidflower.bound_verification_cache import BoundVerificationCache


@pytest.mark.parametrize("custom_reader", [True, False])
def test_saved_source_progress_precedes_real_read_and_is_not_repeated_for_cache_hits(tmp_path, monkeypatch, custom_reader):
    source = tmp_path / "source.bin"
    source.write_bytes(b"original source bytes")
    events = []

    def read(path):
        events.append("read")
        raw = path.read_bytes()
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))

    if not custom_reader:
        from darsia.fluidflower.geometry_raster_sources import descriptor
        def read(path):
            result = descriptor(path)
            events.append("read")
            return result
        from darsia.fluidflower import geometry_raster_sources
        monkeypatch.setattr(geometry_raster_sources, "descriptor", read)
    cache = BoundVerificationCache()
    options = dict(reader=read) if custom_reader else {}
    with observe_check_progress(lambda message: events.append(message)):
        first = cache.descriptor(source, **options)
        assert events == ["Verifying saved source: source.bin", "read"]
        assert cache.descriptor(source, expected=first, **options) == first
        assert len(events) == 2
        source.write_bytes(b"changed source content")
        changed = cache.descriptor(source, expected=first, **options)
    assert changed["sha256"] != first["sha256"]
    assert events == ["Verifying saved source: source.bin", "read"] * 2


@pytest.mark.parametrize("reader_name", ["geometry", "corrected"])
def test_strict_stop_from_source_notification_precedes_hash_payload_open(tmp_path, monkeypatch, reader_name):
    from pathlib import Path
    from darsia.fluidflower import geometry_raster_sources, corrected_setup
    source = tmp_path / "bound.bin"
    source.write_bytes(b"saved source")
    def unexpected_open(*args, **kwargs):
        pytest.fail("A cancelled source hash opened its payload")
    def stop(message):
        raise RuntimeError("original stop request")
    monkeypatch.setattr(Path, "open", unexpected_open)
    reader = geometry_raster_sources.descriptor if reader_name == "geometry" else corrected_setup._descriptor
    with observe_check_progress(stop, strict=True), pytest.raises(RuntimeError, match="original stop request"):
        reader(source)
