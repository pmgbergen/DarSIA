"""Pure transport/client fixtures only; real OS transport probes are not run."""
from pathlib import Path

import pytest

from darsia.fluidflower.job_filesystem import (
    FilesystemFacts, LocalFilesystemFacts, classify_shared_folder, inspect_shared_folder, inspect_local_file, linux_local_facts,
    known_sync_roots, linux_mount_facts,
)


@pytest.mark.parametrize("protocol", ["smb", "nfs"])
def test_positive_transport_is_not_a_server_clock_or_rename_test(protocol):
    result = classify_shared_folder("//host/share/jobs", FilesystemFacts(protocol, "actual OS provider receipt"))
    assert result.allowed
    assert not result.server_mtime_verified and not result.atomic_rename_tested


@pytest.mark.parametrize("protocol", ["unknown", "ntfs", "ext4", "webdav", "fuse.rclone"])
def test_unc_spelling_does_not_establish_a_real_network_fs(protocol):
    assert not classify_shared_folder("//host/share/jobs", FilesystemFacts(protocol, "fixture")).allowed


@pytest.mark.parametrize("folder", ["OneDrive", "OneDrive - University", "Dropbox", "Google Drive", "Nextcloud"])
def test_known_sync_path_is_refused_even_when_exported_over_smb(folder):
    result = classify_shared_folder("//host/share/" + folder + "/jobs", FilesystemFacts("smb", "SMB fixture"))
    assert not result.allowed and "synchronization" in result.reason


def test_offline_cache_and_explicit_client_roots_refuse():
    assert not classify_shared_folder("//host/share/jobs", FilesystemFacts("smb", "provider", offline_cache=True)).allowed
    assert not classify_shared_folder("D:/custom-client-name/jobs", FilesystemFacts("smb", "provider"), sync_roots=["d:/CUSTOM-client-name"]).allowed


def test_real_mount_type_uses_longest_mount_and_decodes_spaces():
    mountinfo = "10 1 8:1 / / rw - ext4 /dev/sda rw\n11 10 0:3 / /mnt/server\\040data rw - cifs //server/share rw\n12 11 0:4 / /mnt/server\\040data/overlay rw - fuse.rclone remote: rw\n"
    assert linux_mount_facts("/mnt/server data/jobs", mountinfo).protocol == "smb"
    facts = linux_mount_facts("/mnt/server data/overlay/jobs", mountinfo)
    assert facts.protocol == "unknown" and facts.synchronized
    assert linux_mount_facts("/local", mountinfo).protocol == "unknown"


def test_nfs_and_malformed_mount_metadata_are_explicit():
    assert linux_mount_facts("/nfs/job", "1 0 0:1 / /nfs rw - nfs4 host:/export rw").protocol == "nfs"
    with pytest.raises(ValueError, match="Malformed"):
        linux_mount_facts("/nfs/job", "not mountinfo")


def test_startup_adapter_uses_injected_facts_without_mutating_local_fixture(tmp_path):
    path = tmp_path / "existing_shared_fixture"
    path.mkdir()
    before = tuple(path.iterdir())
    seen = []
    result = inspect_shared_folder(path, filesystem_provider=lambda root: seen.append(root) or FilesystemFacts("nfs", "injected transport"), sync_roots=[])
    assert result.allowed and seen == [path]
    assert tuple(path.iterdir()) == before


def test_client_metadata_is_readonly_and_custom_dropbox_root_detected(tmp_path):
    path = tmp_path / "dropbox_info.json"
    raw = b'{"personal":{"path":"/custom/synced-store"}}'
    path.write_bytes(raw)
    roots = known_sync_roots(environment={"OneDriveCommercial": "/office/files"}, dropbox_info_paths=[path])
    assert set(roots) == {"/office/files", "/custom/synced-store"}
    assert path.read_bytes() == raw


@pytest.mark.parametrize("filesystem,local", [("ext4", True), ("xfs", True), ("nfs4", False), ("cifs", False), ("fuse.sshfs", False), ("overlay", None), ("unknown", None)])
def test_local_map_requires_positive_actual_deepest_mount(filesystem, local):
    data = "10 1 8:1 / / rw - ext4 /dev/sda rw\n11 10 0:3 / /mnt/site rw - " + filesystem + " source rw\n"
    result = linux_local_facts("/mnt/site/private/stores.toml", data)
    assert result.local is local and result.filesystem == filesystem


def test_ambiguous_or_missing_linux_local_mount_is_unknown():
    data = "1 0 8:1 / /mnt/site rw - ext4 /dev/sda rw\n2 0 0:1 / /mnt/site rw - nfs host:/root rw\n"
    assert linux_local_facts("/mnt/site/stores.toml", data).local is None
    assert linux_local_facts("/other/stores.toml", data).local is None
    with pytest.raises(ValueError, match="Malformed"):
        linux_local_facts("/settings", "broken")


def test_missing_local_settings_uses_only_existing_parent_and_no_mkdir(tmp_path):
    target = tmp_path / "not-created" / "stores.toml"
    seen = []
    result = inspect_local_file(target, locality_provider=lambda parent: seen.append(parent) or LocalFilesystemFacts("fixture", True, "positive fixture"), sync_roots=[])
    assert result.allowed and seen == [tmp_path] and not target.parent.exists()


def test_sync_folder_refuses_before_local_provider(tmp_path):
    target = tmp_path / "OneDrive" / "stores.toml"
    result = inspect_local_file(target, locality_provider=lambda _: pytest.fail("sync refusal must precede probe"), sync_roots=[])
    assert not result.allowed and result.filesystem == "synchronized"


def test_resolved_remote_settings_are_not_a_local_looking_path(tmp_path):
    result = inspect_local_file(tmp_path / "stores.toml", locality_provider=lambda _: LocalFilesystemFacts("nfs", False, "network mount"), sync_roots=[])
    assert not result.allowed and "network mount" in result.evidence
