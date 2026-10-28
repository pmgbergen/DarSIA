"""Optional storage of the per-pixel colour-path coordinate (default off).

Synthetic frames only. With the option off every output must stay
byte-identical to the writer of the integration commit before this change (563e562f,
run from its own source tree). With the option on, each frame also writes
colour_path_coordinate.npz: float32 on the native grid, NaN outside the
colour-quantifiable support, bit-identical to the parameter the frame's own
interpretation passed to the colour-path value table.
"""

from __future__ import annotations

from copy import deepcopy
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import types

import numpy as np
import pytest

from darsia.fluidflower import storage_profile as profile
from darsia.fluidflower.mass import n1_models
from darsia.fluidflower.mass import n1_pipeline as n1
from darsia.fluidflower.mass.production_precision import PixelMath

REPOSITORY = Path(__file__).resolve().parents[2]
# The exported patch's six old source blobs match this local integration base.
# The private 2d8ccda3 commit object is not part of a format-patch export.
HEAD_COMMIT = "563e562f5ca6ae4deefaab9c91a617bfc97270b8"
ON = {"profile": "full", "store_colour_coordinate": True}
LEAN_ON = {"profile": "lean", "store_colour_coordinate": True}


def _module(relative, name):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEAN = _module("tests/unit/test_lean_storage.py", "_colour_lean_helpers")


def _pixel_domain(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fixture = _module(
        "tests/fluidflower/test_n1_pixel_domain_integration.py",
        "_colour_domain_fixture",
    )
    config, _ = fixture.frozen_run.__wrapped__(
        tmp_path, types.SimpleNamespace(param={})
    )
    return config


@pytest.fixture
def synthetic(tmp_path):
    return LEAN._synthetic(tmp_path)


def _same_bits(first, second):
    return LEAN._same_bits(first, second)


# ------------------------------------------------------------------- setting


def test_option_is_off_by_default_and_strict():
    assert profile.DEFAULT_STORE_COLOUR_COORDINATE is False
    assert profile.stores_colour_coordinate({}) is False
    assert (
        profile.stores_colour_coordinate({"storage_profile": {"profile": "lean"}})
        is False
    )
    assert profile.stores_colour_coordinate({"storage_profile": ON}) is True
    assert profile.stores_colour_coordinate({"storage_profile": LEAN_ON}) is True
    # The resolved profile itself is unchanged by the option.
    assert profile.resolve({"storage_profile": LEAN_ON}) == dict(
        profile="lean", colour_intermediates="keep"
    )
    for bad in ("yes", 1, None):
        with pytest.raises(ValueError):
            profile.stores_colour_coordinate(
                {"storage_profile": {"profile": "full", "store_colour_coordinate": bad}}
            )
    with pytest.raises(ValueError):  # The entry still needs its profile.
        profile.stores_colour_coordinate(
            {"storage_profile": {"store_colour_coordinate": True}}
        )
    # A series' mass stage reads the template: series and template must agree.
    profile.require_template_colour_coordinate({}, {})
    profile.require_template_colour_coordinate(
        {"storage_profile": ON}, {"storage_profile": ON}
    )
    for series, template in (
        ({"storage_profile": ON}, {}),
        ({}, {"storage_profile": ON}),
    ):
        with pytest.raises(ValueError, match="mass template"):
            profile.require_template_colour_coordinate(series, template)


# ------------------------------------------- option off = writer before change

_DRIVER = r"""
import gzip, json, os, sys, threading, time, uuid, zipfile
from datetime import datetime, timezone
from pathlib import Path

import threadpoolctl

# Normalize process facts only in the saved IPP observations. Windows library
# discovery still needs the real process ID, and Python keeps its real threads.
_pools = threadpoolctl.threadpool_info
threadpoolctl.threadpool_info = lambda: sorted(
    _pools(), key=lambda row: json.dumps(row, sort_keys=True, default=str))

import darsia.fluidflower.mass.n1_pipeline as n1
import darsia._opencv_runtime as opencv_runtime

_observe_ipp = opencv_runtime.require_ipp_off


def fixed_process_observation(stage):
    record = _observe_ipp(stage)  # Keep the actual IPP check.
    record.update(pid=4242, thread_id=4243, native_thread_id=4244)
    return record


opencv_runtime.require_ipp_off = fixed_process_observation

spec = json.loads(Path(sys.argv[1]).read_text())
state = {}
fixed = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)


class FixedDatetime:
    @staticmethod
    def now(tz=None):
        return fixed


class FixedTime:
    def __getattr__(self, name):
        return getattr(time, name)

    @staticmethod
    def time():
        return 1792000000.0


time.perf_counter = lambda: float(next(state["counter"]))
uuid.uuid4 = lambda: uuid.UUID(int=next(state["ids"]))
zipfile.time = FixedTime()
gzip.time = FixedTime()
n1.datetime = FixedDatetime
outcomes = {}
for case in spec["cases"]:
    state["counter"], state["ids"] = iter(range(1, 10 ** 6)), iter(range(1, 10 ** 6))
    task = dict(run=case["run"], configuration=case["configuration"], commit=spec["commit"],
                job_id="job", **case.get("task", {}))
    outcomes[case["name"]] = n1.run_n1(task, Path(case["output"]), "cpu", "float32")
Path(spec["outcomes"]).write_text(json.dumps(outcomes, sort_keys=True, default=str))
"""


def _head_source(tmp_path):
    """The package source of the commit before this change, from git."""
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(REPOSITORY), "archive", HEAD_COMMIT, "src"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("The commit before this change is not available in this checkout")
    root = tmp_path / "head_tree"
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(root, filter="data")
    return root / "src"


def _drive(tmp_path, source, cases, label):
    """Run the cases with one package source in a fresh interpreter, fixed clocks."""
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    spec = tmp_path / (label + "_spec.json")
    outcomes = tmp_path / (label + "_outcomes.json")
    spec.write_text(
        json.dumps(dict(cases=cases, commit=LEAN.CODE, outcomes=str(outcomes))),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-B", "-P", str(driver), str(spec)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(source)),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(outcomes.read_text(encoding="utf-8"))


def test_option_off_is_byte_identical_to_the_writer_before_this_change(
    synthetic, tmp_path
):
    head = _head_source(tmp_path)
    cases = []
    for name, config, setting in (
        ("n1_full", synthetic, None),
        ("n1_lean", synthetic, {"profile": "lean"}),
        ("pixel_domain_full", _pixel_domain(tmp_path / "domain"), None),
    ):
        config = deepcopy(config)
        if setting is not None:
            config["storage_profile"] = setting
        bound = LEAN._bind(tmp_path / (name + "_config.json"), config)
        cases.append(
            dict(
                name=name,
                run=config["run"],
                configuration=bound,
                output=str(tmp_path / "run" / name),
                # N55: the job's choice, carried by its task; ignored before N55.
                task={} if setting is None else dict(storage_profile=setting["profile"]),
            )
        )
    expected = _drive(tmp_path, head, cases, "head")
    (tmp_path / "run").rename(tmp_path / "head_run")
    actual = _drive(tmp_path, REPOSITORY / "src", cases, "current")
    old, new = LEAN._tree(tmp_path / "head_run"), LEAN._tree(tmp_path / "run")
    assert sorted(old) == sorted(new)
    assert [name for name in old if old[name] != new[name]] == []
    assert expected == actual
    assert not any(name.endswith(profile.COLOUR_COORDINATE_ARCHIVE) for name in new)
    assert any(
        name.startswith("n1_lean/") and profile.STORE_ARCHIVE in name for name in new
    )
    frames = sum(len(outcome["records"]) for outcome in actual.values())
    assert frames >= 5 and sum(name.endswith("products.npz") for name in new) == frames


# ---------------------------------------------------------------- option on


class _Spy:
    """Record, per frame, the parameter each facies passed to the value table.

    Observes the public pixel-math calls only, independently of the capture:
    the first colour-stage per-facies call after each prepare_response start is
    the frame's main interpretation.
    """

    def __init__(self, monkeypatch):
        self.calls, self.frames, self.current = [], [], None
        per_facies, interpret = PixelMath.per_facies, PixelMath.interpret_color_path
        prepare = n1_models.prepare_response
        spy = self

        def spy_per_facies(math, signal, labels, models, *, ignore_labels=()):
            colour = all(
                isinstance(model, n1_models.TypedColor) for model in models.values()
            )
            if colour:
                spy.calls.append(
                    dict(
                        labels=np.array(labels, copy=True),
                        ignore=[int(x) for x in ignore_labels],
                        parameters=[],
                    )
                )
                spy.current = spy.calls[-1]
            try:
                return per_facies(
                    math, signal, labels, models, ignore_labels=ignore_labels
                )
            finally:
                spy.current = None

        def spy_interpret(math, parametrization, path_distances, values):
            if spy.current is not None:
                spy.current["parameters"].append(np.array(parametrization, copy=True))
            return interpret(math, parametrization, path_distances, values)

        def spy_prepare(*args, **kwargs):
            start = len(spy.calls)
            response = prepare(*args, **kwargs)
            spy.frames.append(
                dict(first_call=start, quantifiable=response["quantifiable"].copy())
            )
            return response

        monkeypatch.setattr(PixelMath, "per_facies", spy_per_facies)
        monkeypatch.setattr(PixelMath, "interpret_color_path", spy_interpret)
        monkeypatch.setattr(n1_models, "prepare_response", spy_prepare)

    def expected(self, frame):
        """The in-memory coordinate of one frame, rebuilt from the spied calls."""
        record = self.frames[frame]
        call = self.calls[record["first_call"]]
        labels = call["labels"]
        active = [
            label for label in np.unique(labels) if int(label) not in call["ignore"]
        ]
        assert len(active) == len(call["parameters"])
        coordinate = np.full(labels.shape, np.nan, np.float32)
        for label, parameter in zip(active, call["parameters"]):
            assert parameter.dtype == np.float32
            coordinate[labels == label] = parameter
        coordinate[~record["quantifiable"]] = np.nan
        return coordinate


def _coordinate(root):
    with np.load(
        root / profile.COLOUR_COORDINATE_ARCHIVE, allow_pickle=False
    ) as archive:
        assert archive.files == [profile.COLOUR_COORDINATE_ARRAY]
        return archive[profile.COLOUR_COORDINATE_ARRAY].copy()


def _assert_equal_with_nan(stored, expected):
    assert (
        stored.dtype == expected.dtype == np.float32 and stored.shape == expected.shape
    )
    assert np.array_equal(np.isnan(stored), np.isnan(expected))
    finite = ~np.isnan(expected)
    assert stored[finite].tobytes() == expected[finite].tobytes()


@pytest.mark.parametrize("route", ["n1", "pixel_domain"])
def test_stored_coordinate_equals_the_in_memory_coordinate_bit_for_bit(
    route, synthetic, tmp_path, monkeypatch
):
    config = synthetic if route == "n1" else _pixel_domain(tmp_path / "domain")
    spy = _Spy(monkeypatch)
    outcome = LEAN._run(tmp_path, "on", config, ON)
    frames = list(LEAN._frames(outcome))
    assert len(spy.frames) == len(frames) >= 1
    for index, (root, result) in enumerate(frames):
        stored = _coordinate(root)
        expected = spy.expected(index)
        _assert_equal_with_nan(stored, expected)
        assert np.isfinite(stored).any() and np.isnan(stored).any()
        assert (
            stored.shape == LEAN._archive(root / "products.npz")["phase_labels"].shape
        )


def _check_recorded(root, result, outcome_record):
    record = result[profile.COLOUR_COORDINATE_RECORD]
    stored = _coordinate(root)
    file = n1.asset(root / profile.COLOUR_COORDINATE_ARCHIVE)
    assert record["format"] == profile.COLOUR_COORDINATE_FORMAT
    assert (record["path"], record["array"]) == (
        profile.COLOUR_COORDINATE_ARCHIVE,
        profile.COLOUR_COORDINATE_ARRAY,
    )
    assert (record["sha256"], record["size_bytes"]) == (
        file["sha256"],
        file["size_bytes"],
    )
    assert record["identity"] == n1._array_identity(stored)
    assert np.dtype(record["identity"]["dtype"]) == np.float32
    commit = json.loads((root / "commit.json").read_bytes())
    for inventory in (commit["files"], outcome_record["files"]):
        entry = inventory[profile.COLOUR_COORDINATE_ARCHIVE]
        assert (entry["sha256"], entry["size_bytes"]) == (
            file["sha256"],
            file["size_bytes"],
        )
    return stored


def test_on_writes_the_field_in_full_and_lean_and_changes_nothing_else(
    synthetic, tmp_path
):
    off = LEAN._run(tmp_path, "off", synthetic)
    full = LEAN._run(tmp_path, "full_on", synthetic, ON)
    lean = LEAN._run(tmp_path, "lean_on", synthetic, LEAN_ON)
    lean_off = LEAN._run(tmp_path, "lean_off", synthetic, {"profile": "lean"})
    triples = zip(
        LEAN._frames(off),
        LEAN._frames(full),
        LEAN._frames(lean),
        LEAN._frames(lean_off),
        full["records"],
        lean["records"],
    )
    for (o_root, o), (f_root, f), (l_root, l), (lo_root, lo), f_rec, l_rec in triples:
        assert profile.COLOUR_COORDINATE_RECORD not in o
        assert not (o_root / profile.COLOUR_COORDINATE_ARCHIVE).exists()
        stored_full = _check_recorded(f_root, f, f_rec)
        stored_lean = _check_recorded(l_root, l, l_rec)
        assert _same_bits(stored_full, stored_lean)
        assert (
            f[profile.COLOUR_COORDINATE_RECORD] == l[profile.COLOUR_COORDINATE_RECORD]
        )
        # Storage only: every other file is unchanged by the option.
        for name in ("products.npz", "native_geometry.npz"):
            assert (o_root / name).read_bytes() == (f_root / name).read_bytes(), name
        assert (lo_root / "products.npz").read_bytes() == (
            l_root / "products.npz"
        ).read_bytes()
        for first, second in (
            (o_root, f_root),
            (lo_root, l_root),
        ):  # gzip headers hold a time
            with (
                gzip.open(first / "native_statistics.json.gz") as a,
                gzip.open(second / "native_statistics.json.gz") as b,
            ):
                assert a.read() == b.read()
        assert {p.name for p in f_root.iterdir()} == {
            p.name for p in o_root.iterdir()
        } | {profile.COLOUR_COORDINATE_ARCHIVE}
        assert {p.name for p in l_root.iterdir()} == {
            p.name for p in lo_root.iterdir()
        } | {profile.COLOUR_COORDINATE_ARCHIVE}
        drop = {"compute_seconds", "opencv_ipp", profile.COLOUR_COORDINATE_RECORD}
        assert {k: v for k, v in o.items() if k not in drop} == {
            k: v for k, v in f.items() if k not in drop
        }
        assert {
            k: v for k, v in lo.items() if k not in drop | {"lean_storage", "static"}
        } == {k: v for k, v in l.items() if k not in drop | {"lean_storage", "static"}}


def test_pixel_domain_route_writes_the_field(tmp_path):
    config = _pixel_domain(tmp_path / "domain")
    outcome = LEAN._run(tmp_path, "domain_on", config, ON)
    for (root, result), record in zip(LEAN._frames(outcome), outcome["records"]):
        stored = _check_recorded(root, result, record)
        assert _same_bits(profile.read_colour_coordinate(result, root), stored)


# ------------------------------------------------------ readers and the index


def test_reader_returns_the_verified_field_and_refuses_absence_or_change(
    synthetic, tmp_path
):
    off = LEAN._run(tmp_path, "off", synthetic)
    lean = LEAN._run(tmp_path, "lean_on", synthetic, LEAN_ON)
    (o_root, o), (root, result) = next(LEAN._frames(off)), next(LEAN._frames(lean))
    value = profile.read_colour_coordinate(result, root)
    assert _same_bits(value, _coordinate(root))
    with pytest.raises(KeyError, match="no colour-path coordinate"):
        profile.read_colour_coordinate(o, o_root)
    changed = deepcopy(result)
    changed[profile.COLOUR_COORDINATE_RECORD]["identity"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recorded identity"):
        profile.read_colour_coordinate(changed, root)
    changed = deepcopy(result)
    changed[profile.COLOUR_COORDINATE_RECORD]["format"] = "other"
    with pytest.raises(ValueError, match="Unknown colour-path coordinate record"):
        profile.read_colour_coordinate(changed, root)
    path = root / profile.COLOUR_COORDINATE_ARCHIVE
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA or identity|size differs"):
        profile.read_colour_coordinate(result, root, reader=n1.DataReader())


def test_products_index_lists_the_field_as_a_map(synthetic, tmp_path):
    from darsia.fluidflower.pipeline import publication_runtime as runtime

    outcome = LEAN._run(tmp_path, "lean", synthetic, LEAN_ON)
    token = "d" * 32
    claim = types.SimpleNamespace(
        task=dict(run=synthetic["run"]), token=token, output=str(tmp_path / "lean")
    )
    producers = []
    for record in outcome["records"]:
        root = Path(record["nas_directory"])
        producers.append(
            dict(
                case_id=record["frame_id"],
                attempt_directory=str(tmp_path / "lean"),
                manifest=outcome["manifest"],
                commit=n1.asset(root / "commit.json"),
                frame_registration=outcome["manifest"],
                publication=outcome["manifest"],
                result=n1.asset(root / "result.json"),
            )
        )
    result = dict(
        outcome,
        attempt_id=token,
        frame_producers=producers,
        table=n1.asset(Path(outcome["manifest"]["path"])),
    )
    rows, _, _ = runtime.extract_products(
        result, claim, "main", producer_origin_for_attempt=None
    )
    ids = {row["product_id"]: row for row in rows}
    for record in outcome["records"]:
        key = "frames/" + record["frame_id"] + "/" + profile.COLOUR_COORDINATE_ARCHIVE
        assert ids[key]["kind"] == "map"
        assert (
            ids[key]["sha256"]
            == record["files"][profile.COLOUR_COORDINATE_ARCHIVE]["sha256"]
        )


def test_a_prefix_with_the_field_resumes(synthetic, tmp_path):
    task, old, source = LEAN._retired_prefix(tmp_path, synthetic, ON)
    assert (old / "frames" / "sample_1" / profile.COLOUR_COORDINATE_ARCHIVE).exists()
    outcome = n1.run_n1(
        dict(task, resume_from=source), old.parent / ("b" * 32), "cpu", "float32"
    )
    assert outcome["resumed_frame_count"] == 1
    for root, result in LEAN._frames(outcome):
        profile.read_colour_coordinate(result, root)
