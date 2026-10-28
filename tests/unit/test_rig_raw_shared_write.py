"""The owned writer uses the same exclusive primitives for admitted shared outputs."""
import json
import pytest
from darsia.fluidflower import job_filesystem as fs, rig_raw_write as write


@pytest.mark.parametrize('protocol', ['smb', 'nfs'])
def test_shared_raw_output_keeps_exclusive_ownership_and_caps(tmp_path, monkeypatch, protocol):
    root = tmp_path/'shared-output'; root.mkdir()
    target = root/'fresh'
    def output_admission(path):
        return fs.ResultsRootAdmission(True, protocol, False, 'unit test', 'unit test',
            str(root), str(path), fs.SharedFolderAdmission(True, protocol, 'unit test', 'unit test'))
    monkeypatch.setattr(fs, 'inspect_results_root', output_admission)
    authority=dict(destination=str(target), intended_root=str(root), forbidden_raw_roots=[str(tmp_path/'acquisition')],
        maximum_output_bytes=10_000, maximum_file_bytes=4_000)
    owner=write.OwnedRawOutput(target, authority, operation_id='a'*32, request={})
    marker=(target/'raw_build_owner.json').read_bytes()
    assert json.loads(marker)['local_filesystem']['local'] is False
    with owner.open(target/'kept.bin') as stream:stream.write(b'kept')
    with pytest.raises(FileExistsError):
        write.OwnedRawOutput(target, authority, operation_id='b'*32, request={})
    assert (target/'raw_build_owner.json').read_bytes()==marker
    assert (target/'kept.bin').read_bytes()==b'kept'
    with pytest.raises(ValueError, match='byte ceiling'):
        with owner.open(target/'partial.bin') as stream:stream.write(b'x'*4_001)
    assert (target/'partial.bin').stat().st_size==0
