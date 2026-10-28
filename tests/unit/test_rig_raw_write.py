"""Synthetic ownership/refusal contracts; no RAW decoder or production writes."""

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from darsia.fluidflower import rig_raw_write as write


def reserve(tmp_path, monkeypatch, name="fresh"):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    root = tmp_path / "products"
    root.mkdir(exist_ok=True)
    from darsia.fluidflower import job_filesystem
    monkeypatch.setattr(job_filesystem, "inspect_local_file", lambda path, **kwargs:
        job_filesystem.LocalFileAdmission(True, "synthetic-local", "unit test", "unit test", str(root)))
    target = root / name
    authority = dict(destination=str(target), intended_root=str(root), forbidden_raw_roots=[str(raw)],
                     maximum_output_bytes=100_000, maximum_file_bytes=20_000)
    return write.OwnedRawOutput(target, authority, operation_id="a" * 32,
        request=dict(synthetic=True)), authority


def test_atomic_reservation_refuses_a_peer_even_when_only_owner_marker_exists(tmp_path, monkeypatch):
    owner, authority = reserve(tmp_path, monkeypatch)
    before = (owner.destination / "raw_build_owner.json").read_bytes()
    with pytest.raises(FileExistsError):
        write.OwnedRawOutput(owner.destination, authority, operation_id="b" * 32,
                             request=dict(synthetic=True))
    assert (owner.destination / "raw_build_owner.json").read_bytes() == before


def test_same_operation_cannot_overwrite_or_save_another_rig(tmp_path, monkeypatch):
    owner, _ = reserve(tmp_path, monkeypatch)
    rig = object()
    owner.begin_setup(rig, owner.destination / "figures")
    with pytest.raises(ValueError, match="original rig"):
        owner.begin_save(object(), owner.destination)
    owner.begin_save(rig, owner.destination)
    image = SimpleNamespace(img=np.ones((2, 3)), metadata=lambda: dict(synthetic=True))
    target = owner.destination / "baseline.npz"
    owner.image(image, target)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        owner.image(image, target)
    assert target.read_bytes() == before


def test_owner_marker_tamper_and_root_escape_refuse(tmp_path, monkeypatch):
    owner, _ = reserve(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="leaves"):
        with owner.open(tmp_path / "outside.bin"):
            pytest.fail("writer must not open")
    (owner.destination / "raw_build_owner.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="ownership"):
        owner.check()


def test_unsupported_root_and_protected_acquisition_refuse_before_reservation(tmp_path, monkeypatch):
    from darsia.fluidflower import job_filesystem
    root = tmp_path / "raw"
    root.mkdir()
    target = root / "forbidden"
    authority = dict(destination=str(target), intended_root=str(root), forbidden_raw_roots=[str(root)],
                     maximum_output_bytes=100_000, maximum_file_bytes=20_000)
    with pytest.raises(ValueError, match="protected acquisition"):
        write.OwnedRawOutput(target, authority, operation_id="a" * 32, request={})
    assert not target.exists()
    authority["forbidden_raw_roots"] = [str(tmp_path / "acquisitions")]
    monkeypatch.setattr(job_filesystem, "inspect_local_file", lambda path, **kwargs:
        SimpleNamespace(allowed=False))
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda path, **kwargs:
        job_filesystem.SharedFolderAdmission(False, "unknown", "unsupported", "unit test"))
    with pytest.raises(ValueError, match="supported local or shared filesystem"):
        write.OwnedRawOutput(target, authority, operation_id="a" * 32, request={})
    assert not target.exists()


def test_existing_reparse_ancestor_is_checked_before_resolution(tmp_path, monkeypatch):
    original = Path.lstat
    ancestor = tmp_path / "alias"
    def lstat(path, *args, **kwargs):
        if path == ancestor:
            return SimpleNamespace(st_mode=0o040755, st_file_attributes=0x400)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="reparse"):
        write._canonical(ancestor / "not-created" / "output")


def test_budget_refuses_before_an_over_limit_byte_and_keeps_partial(tmp_path, monkeypatch):
    owner, _ = reserve(tmp_path, monkeypatch)
    target = owner.destination / "partial.bin"
    with pytest.raises(ValueError, match="byte ceiling"):
        with owner.open(target) as stream:
            stream.write(b"kept")
            stream.write(b"x" * owner.authority["maximum_file_bytes"])
    assert target.read_bytes() == b"kept"
    assert owner.failed is True


def test_raw_npz_stream_retains_original_correction_writer_contract(tmp_path, monkeypatch):
    import darsia
    owner, _ = reserve(tmp_path, monkeypatch)
    correction = darsia.CurvatureCorrection(config={})
    correction.cache = dict(grid=np.arange(12, dtype=np.float32).reshape(2, 6), shape=(2, 3))
    target = owner.destination / "shape_correction_0_curvaturecorrection.npz"
    owner.correction(correction, target)
    restored = darsia.CurvatureCorrection()
    restored.load(target)
    np.testing.assert_array_equal(restored.cache["grid"], correction.cache["grid"])


def test_owned_type_correction_keeps_original_archive_members_and_exclusive_output(tmp_path, monkeypatch):
    """The original type serializer writes identical payloads through the bounded stream."""
    import darsia
    owner, _ = reserve(tmp_path, monkeypatch)
    correction = darsia.TypeCorrection(np.float32)
    legacy = tmp_path / "legacy-type.npz"
    correction.save(legacy)
    target = owner.destination / "shape_correction_0_typecorrection.npz"
    owner.correction(correction, target)
    with ZipFile(legacy) as original, ZipFile(target) as actual:
        assert actual.namelist() == original.namelist() == ["class_name.npy", "data_type.npy"]
        assert all(actual.read(name) == original.read(name) for name in original.namelist())
    restored = darsia.TypeCorrection()
    restored.load(target)
    assert restored.data_type is np.float32
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        owner.correction(correction, target)
    assert target.read_bytes() == before and owner.failed is False


def test_owned_type_correction_retains_partial_without_exceeding_operation_cap(tmp_path, monkeypatch):
    """The original NPZ serializer cannot bypass the operation ceiling through its stream."""
    import darsia
    correction = darsia.TypeCorrection(np.float32)
    legacy = tmp_path / "legacy-type.npz"
    correction.save(legacy)
    owner, _ = reserve(tmp_path, monkeypatch)
    ceiling = owner.authority["maximum_output_bytes"]
    initial = sum(path.stat().st_size for path in owner.destination.rglob("*") if path.is_file())
    remaining = ceiling - initial - (legacy.stat().st_size - 1)
    for index in range((remaining + owner.authority["maximum_file_bytes"] - 1)
                       // owner.authority["maximum_file_bytes"]):
        count = min(remaining, owner.authority["maximum_file_bytes"])
        with owner.open(owner.destination / ("retained-" + str(index) + ".bin")) as stream:
            stream.write(b"x" * count)
        remaining -= count
    target = owner.destination / "shape_correction_0_typecorrection.npz"
    with pytest.raises(ValueError, match="byte ceiling"):
        owner.correction(correction, target)
    assert owner.failed is True and target.is_file()
    assert sum(path.stat().st_size for path in owner.destination.rglob("*") if path.is_file()) <= ceiling
    assert target.stat().st_size < legacy.stat().st_size


def test_encoder_chunks_reuse_open_handle_and_do_not_repeat_tree_checks(tmp_path, monkeypatch):
    owner, _ = reserve(tmp_path, monkeypatch)
    counts = dict(check=0, scan=0)
    original_check, original_scan = owner.check, Path.rglob

    def check(*args, **kwargs):
        counts['check'] += 1
        return original_check(*args, **kwargs)

    def scan(path, *args, **kwargs):
        if path == owner.destination:
            counts['scan'] += 1
        return original_scan(path, *args, **kwargs)

    monkeypatch.setattr(owner, 'check', check)
    monkeypatch.setattr(Path, 'rglob', scan)
    path = owner.destination/'chunks.bin'
    with owner.open(path) as stream:
        initial = counts.copy()
        for _ in range(1000):
            stream.write(b'0123456789')
        stream.seek(0)
        stream.write(b'header')
        assert counts == initial  # No NAS path I/O per ZIP/encoder chunk.
    assert counts == dict(check=2, scan=2)
    assert path.read_bytes() == b'header' + (b'0123456789'*1000)[6:]


def test_seek_back_headers_do_not_charge_bytes_twice_and_holes_count(tmp_path, monkeypatch):
    owner, _ = reserve(tmp_path, monkeypatch)
    owner.authority['maximum_output_bytes'] = len(owner.marker) + 20
    path = owner.destination/'seek.bin'
    with owner.open(path) as stream:
        stream.write(b'a'*10)
        stream.seek(0)
        stream.write(b'header')
        stream.seek(19)
        stream.write(b'z')
    assert path.stat().st_size == 20
    assert owner._retained_bytes == len(owner.marker) + 20
    with pytest.raises(ValueError, match='byte ceiling'):
        with owner.open(owner.destination/'over.bin') as stream:
            stream.write(b'x')


def test_nested_buffered_writes_share_the_whole_operation_cap(tmp_path, monkeypatch):
    owner, _ = reserve(tmp_path, monkeypatch)
    owner.authority['maximum_output_bytes'] = len(owner.marker) + 30
    with pytest.raises(ValueError, match='byte ceiling'):
        with owner.open(owner.destination/'outer.bin') as outer:
            outer.write(b'a'*16)
            with owner.open(owner.destination/'inner.bin') as inner:
                inner.write(b'b'*15)
    assert owner.failed
    assert (owner.destination/'outer.bin').read_bytes() == b'a'*16
    assert (owner.destination/'inner.bin').stat().st_size == 0


def test_file_boundary_rejects_marker_change_and_unaccounted_extra_output(tmp_path, monkeypatch):
    owner, _ = reserve(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='ownership'):
        with owner.open(owner.destination/'pending.bin') as stream:
            stream.write(b'retained')
            (owner.destination/'raw_build_owner.json').write_bytes(b'changed')
    assert owner.failed
    owner, _ = reserve(tmp_path, monkeypatch, name='extra')
    owner.authority['maximum_output_bytes'] = len(owner.marker) + 30
    with pytest.raises(ValueError, match='byte ceiling'):
        with owner.open(owner.destination/'pending.bin') as stream:
            stream.write(b'a'*16)
            (owner.destination/'outside-writer.bin').write_bytes(b'b'*15)
    assert owner.failed


def test_image_npz_retains_original_numpy_archive_payload(tmp_path, monkeypatch):
    owner, _ = reserve(tmp_path, monkeypatch)
    image = SimpleNamespace(img=np.arange(120,dtype=np.float64).reshape(8,15),
                            metadata=lambda:dict(height=2.5, origin=[1,2], format='synthetic'))
    expected = tmp_path/'original.npz'
    np.savez(expected,array=image.img,metadata=image.metadata())
    actual = owner.destination/'image.npz'
    owner.image(image,actual)
    with ZipFile(expected) as left, ZipFile(actual) as right:
        assert left.namelist() == right.namelist()
        assert all(left.read(name)==right.read(name) for name in left.namelist())
