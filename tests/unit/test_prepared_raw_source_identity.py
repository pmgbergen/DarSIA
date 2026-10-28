"""A RAW reader's source record must name the exact acquisition it was given."""
from darsia.fluidflower.pipeline.series import same_raw_source

ACQUISITION = {"path": r"\\nas\share\raw\DSC00002.ARW", "sha256": "a" * 64, "size_bytes": 21316608}


def stay(path):
    return path


def test_reader_record_without_size_names_the_same_acquisition():
    recorded = {"path": ACQUISITION["path"], "sha256": ACQUISITION["sha256"]}
    assert same_raw_source(recorded, ACQUISITION, stay)


def test_recorded_size_must_match():
    recorded = dict(ACQUISITION, size_bytes=ACQUISITION["size_bytes"] + 1)
    assert not same_raw_source(recorded, ACQUISITION, stay)
    assert same_raw_source(dict(ACQUISITION), ACQUISITION, stay)


def test_other_bytes_are_refused():
    recorded = {"path": ACQUISITION["path"], "sha256": "b" * 64}
    assert not same_raw_source(recorded, ACQUISITION, stay)


def test_declared_relocation_is_the_same_acquisition():
    moved = r"\\nas\share\project\raw\DSC00002.ARW"
    recorded = {"path": moved, "sha256": ACQUISITION["sha256"]}
    assert same_raw_source(recorded, ACQUISITION, lambda path: moved)
    assert not same_raw_source(recorded, ACQUISITION, stay)


def test_path_spelling_is_compared_as_a_windows_path():
    recorded = {"path": ACQUISITION["path"].upper().replace("\\", "/"), "sha256": ACQUISITION["sha256"]}
    assert same_raw_source(recorded, ACQUISITION, stay)


def test_a_missing_record_is_refused():
    assert not same_raw_source(None, ACQUISITION, stay)
    assert not same_raw_source({"sha256": ACQUISITION["sha256"]}, ACQUISITION, stay)
