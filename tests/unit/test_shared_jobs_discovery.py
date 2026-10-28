"""Publication discovery uses real metadata and isolated local registrations."""

import json

import pytest

from darsia.fluidflower import native_jobs_registry as registry_module
from darsia.fluidflower.shared_job_folder import SharedJobFolder
from darsia.fluidflower.shared_jobs_discovery import discover_published_jobs


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Production registrations still require UNC. Fixtures never contact a NAS.
    monkeypatch.setattr(registry_module, "require_unc", lambda value: None)
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    return (SharedJobFolder(shared_root), tmp_path / "series",
            registry_module.JobsUIRegistry(tmp_path / "local-registry"))


def job(folder, name, series_id="own", **changes):
    root = folder.path / name
    root.mkdir()
    document = dict(pipeline="native-phase-n1", expected_runs=["run-a"],
                    precision="float32", commit="a" * 40, series_id=series_id)
    document.update(changes)
    (root / "job.json").write_text(json.dumps(document), encoding="utf-8")
    folder.publish(root, master_machine="original-master")
    return root


def test_new_machine_discovers_only_published_own_series_and_can_select(fixture):
    folder, series, registry = fixture
    own = job(folder, "own")
    job(folder, "foreign", "other")
    job(folder, "unknown", None)
    unpublished = folder.path / "unpublished"
    unpublished.mkdir()
    (unpublished / "job.json").write_text(
        '{"series_id":"own","expected_runs":["run-a"],"precision":"float32"}'
    )
    before = {path: path.read_bytes() for path in folder.path.rglob("*") if path.is_file()}
    rows = discover_published_jobs(folder.path, series, "own", registry)
    assert [row["root"] for row in rows] == [str(own)]
    assert rows[0]["series_label"] == "own"
    registry.set_active(series, "own", own)
    assert registry.jobs(series, "own")[0]["active"] is True
    assert {path: path.read_bytes() for path in folder.path.rglob("*") if path.is_file()} == before


def test_repeated_discovery_preserves_local_archive_label_and_selection(fixture, monkeypatch):
    folder, series, registry = fixture
    archived = job(folder, "archived")
    active = job(folder, "active")
    registry.register_job(series, "own", archived, label="Keep this label")
    registry.set_archived(series, "own", archived, True, actor="fixture")
    registry.register_job(series, "own", active)
    before = registry.context(series, "own")
    register = registry.register_job
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[2])
        return register(*args, **kwargs)

    monkeypatch.setattr(registry, "register_job", observed)
    assert [row["root"] for row in discover_published_jobs(folder, series, "own", registry)] == [str(active)]
    assert calls == []
    assert registry.context(series, "own") == before
    added = job(folder, "added")
    rows = discover_published_jobs(folder, series, "own", registry, include_archived=True)
    assert calls == [added]
    assert len(rows) == 3 and rows[0]["archived"] is True
    state = registry.context(series, "own")
    assert state["jobs"][:2] == before["jobs"]
    assert state["active_job"] == str(active)
    discover_published_jobs(folder, series, "own", registry)
    assert calls == [added]


def test_duplicate_publication_roots_register_once(fixture, monkeypatch):
    folder, series, registry = fixture
    own = job(folder, "own")
    publication = folder.jobs()[0]
    monkeypatch.setattr(folder, "jobs", lambda: [publication, dict(publication)])
    calls = []
    register = registry.register_job

    def observed(*args, **kwargs):
        calls.append(args[2])
        return register(*args, **kwargs)

    monkeypatch.setattr(registry, "register_job", observed)
    assert len(discover_published_jobs(folder, series, "own", registry)) == 1
    assert calls == [own]


def test_nonexact_series_are_not_registered(fixture):
    folder, series, registry = fixture
    job(folder, "case-different", "OWN")
    job(folder, "whitespace-different", " own ")
    job(folder, "unsupported-series-declaration", {"id": "own"})
    assert discover_published_jobs(folder, series, "own", registry) == []
    assert not registry.root.exists()


@pytest.mark.parametrize("changes,message", [
    ({"precision": None}, "precision is not explicit"),
    ({"expected_runs": []}, "distinct expected_runs"),
])
def test_malformed_own_series_job_surfaces_normal_validation_refusal(fixture, changes, message):
    folder, series, registry = fixture
    root = job(folder, "malformed", **changes)
    before = (root / "job.json").read_bytes()
    with pytest.raises(ValueError, match=message):
        discover_published_jobs(folder, series, "own", registry)
    assert not registry.root.exists()
    assert (root / "job.json").read_bytes() == before


@pytest.mark.parametrize("folder_name", [None, "", "not-created-yet"])
def test_empty_or_uncreated_shared_folder_keeps_local_jobs(fixture, folder_name):
    folder, series, registry = fixture
    own = job(folder, "own")
    registry.register_job(series, "own", own)
    selected = folder.path / folder_name if folder_name else folder_name
    before = registry.context(series, "own")
    assert discover_published_jobs(selected, series, "own", registry) == registry.jobs(series, "own")
    assert registry.context(series, "own") == before


def test_changed_publication_or_job_never_registers(fixture, monkeypatch):
    folder, series, registry = fixture
    own = job(folder, "own")
    publication = folder.jobs()[0]
    (own / "job.json").write_bytes((own / "job.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="metadata changed"):
        discover_published_jobs(folder, series, "own", registry)
    assert not registry.root.exists()
    # Also reject a change between the authenticated index read and local import.
    monkeypatch.setattr(folder, "jobs", lambda: [publication])
    with pytest.raises(ValueError, match="during discovery"):
        discover_published_jobs(folder, series, "own", registry)
    assert not registry.root.exists()


def test_moved_job_folder_does_not_hide_other_registered_jobs(fixture):
    import shutil
    folder, series, registry = fixture
    (series / "old").mkdir(parents=True)
    moved = job(SharedJobFolder(series / "old"), "moved")
    kept = job(folder, "kept")
    registry.register_job(series, "own", moved)
    registry.register_job(series, "own", kept)
    shutil.rmtree(moved)  # Moved or deleted outside DarSIA.
    assert [row["root"] for row in registry.jobs(series, "own")] == [str(kept)]
    assert registry.missing_jobs(series, "own") == [str(moved)]
    rows = discover_published_jobs(folder, series, "own", registry)
    assert [row["root"] for row in rows] == [str(kept)]


def test_changed_registered_job_metadata_is_still_refused(fixture):
    folder, series, registry = fixture
    series.mkdir()
    changed = job(folder, "changed")
    registry.register_job(series, "own", changed)
    (changed / "job.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata SHA changed"):
        registry.jobs(series, "own")


def test_unreachable_series_folder_keeps_the_original_read_error(fixture):
    import shutil
    folder, series, registry = fixture
    gone = job(folder, "gone")
    registry.register_job(series, "own", gone)  # The series folder itself is unreachable.
    shutil.rmtree(gone)
    assert registry.missing_jobs(series, "own") == []
    with pytest.raises(OSError):
        registry.jobs(series, "own")


def test_moved_job_folder_is_not_a_known_job_for_dependency_checks(fixture):
    import shutil
    folder, series, registry = fixture
    (series / "old").mkdir(parents=True)
    moved = job(SharedJobFolder(series / "old"), "moved")
    kept = job(folder, "kept")
    registry.register_job(series, "own", moved)
    registry.register_job(series, "own", kept)
    shutil.rmtree(moved)
    assert [row["root"] for row in registry.known_jobs()] == [str(kept)]
