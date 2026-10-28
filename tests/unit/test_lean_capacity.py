"""Real lean/full writers and retained retries fit conservative DATA bounds."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import storage_profile as storage
from darsia.fluidflower.mass import n1_pipeline as producer
from darsia.fluidflower.mass.production_precision import PixelMath
from darsia.fluidflower.pipeline import io_ops, job_queue
from darsia.fluidflower.pipeline import lean_capacity as capacity

from .test_lean_storage import synthetic, _run, _frames, _archive, _same_bits, _module  # noqa: F401


@pytest.fixture
def configuration(synthetic):
    value = deepcopy(synthetic)
    value.update(precision="float32", storage_profile={"profile": "lean", "colour_intermediates": "keep"})
    return value


def _source(use, bound):
    assert use in capacity.STORE_ROLES
    return Path(bound["path"])


def _files(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_run_store_header_bound_reads_no_numerical_payload_and_matches_real_store(configuration, tmp_path, monkeypatch):
    config = configuration
    before, settings = _files(tmp_path), deepcopy(config)
    with monkeypatch.context() as patch:
        patch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Header budgeting loaded a numerical array"))
        bound = capacity.run_store_budget(config, path_for_use=_source)
    assert _files(tmp_path) == before and config == settings
    pipeline = producer.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"))
    output = tmp_path / "lean-store"
    output.mkdir()
    actual = storage.LeanRun(output, config, pipeline,
        dict(profile="lean", colour_intermediates="keep"), colour_intermediates_produced=False)
    assert set(bound["arrays"]) == set(storage._STORE_ARRAYS)
    assert (output / storage.STORE_ARCHIVE).stat().st_size <= bound["archive_bytes"]
    assert (output / storage.STORE_INDEX).stat().st_size <= bound["metadata_bytes"]
    arrays = _archive(output / storage.STORE_ARCHIVE)
    for name, value in arrays.items():
        metadata = bound["arrays"][name]
        assert metadata["dtype"] == value.dtype.str
        assert metadata["shape"] == list(value.shape)
        assert metadata["payload_bytes"] == value.nbytes
    assert actual.store_record["archive"]["size_bytes"] <= bound["archive_bytes"]
    assert bound["numerical_payload_loaded"] is False and bound["production_permission_granted"] is False


def test_plain_full_publication_members_and_sizes_fit_all_fallback_tables(configuration, tmp_path):
    config = configuration
    bound = capacity.fallback_archives(config, canvas_hw=(32, 48), precision="float32")
    full = _run(tmp_path, "full-reference", config, dict(profile="full"))
    for root, record in _frames(full):
        products, geometry = _archive(root / "products.npz"), _archive(root / "native_geometry.npz")
        assert set(products) <= set(bound["product_payloads"])
        assert set(bound["product_payloads"]) - set(products) == {"approved_analysis_gap_mask"}
        assert set(geometry) == set(bound["geometry_payloads"])
        for name, value in products.items():
            assert value.nbytes <= bound["product_payloads"][name], name
        for name, value in geometry.items():
            assert value.nbytes <= bound["geometry_payloads"][name], name
        assert (root / "products.npz").stat().st_size <= bound["products_bytes"]
        assert (root / "native_geometry.npz").stat().st_size <= bound["geometry_bytes"]


def test_every_static_and_total_fallback_remains_bit_exact_and_reserved(configuration, tmp_path, monkeypatch):
    config = configuration
    store = capacity.run_store_budget(config, path_for_use=_source)
    bound = capacity.phase2_lean_budget(config, canvas_hw=(32, 48), store_bound=store)
    full = _run(tmp_path, "full-fallback-reference", config, dict(profile="full"))
    def unavailable(self):
        raise ValueError("Synthetic missing regeneration rule")
    def unequal(products):
        return np.nextafter(np.asarray(products["mass_total"]), np.float32(np.inf))
    monkeypatch.setattr(storage.LeanFrame, "regeneration", unavailable)
    monkeypatch.setattr(storage, "_rebuild_mass_total", unequal)
    lean = _run(tmp_path, "lean-fallback", config, dict(profile="lean", colour_intermediates="keep"))
    for (old_root, _old), (new_root, new) in zip(_frames(full), _frames(lean), strict=True):
        assert new["static"]["storage"]["regenerated"] == []
        assert "mass_total" not in new["lean_storage"]["products"]["omitted"]
        for name in ("products.npz", "native_geometry.npz"):
            first, second = _archive(old_root / name), _archive(new_root / name)
            assert first.keys() == second.keys()
            assert all(_same_bits(first[key], second[key]) for key in first)
        assert (new_root / "products.npz").stat().st_size <= bound["parts"]["products"]
        assert (new_root / "native_geometry.npz").stat().st_size <= bound["parts"]["static_fallback"]
        other_bytes = sum(path.stat().st_size for path in new_root.iterdir()
                          if path.is_file() and path.suffix != ".npz")
        assert other_bytes <= bound["parts"]["records"]
    assert bound["successful_omission_assumed"] is False and bound["compression_gain_assumed"] is False


def test_run_store_sources_are_explicit_current_host_aliases_never_foreign_probes(configuration, monkeypatch):
    config = deepcopy(configuration)
    actual = {use: Path(config["shared"][key]["path"]) for use, key in
              zip(capacity.STORE_ROLES[:2], ("physical_reference", "floor_curve"), strict=True)}
    foreign = set()
    for key in ("physical_reference", "floor_curve"):
        text = "Z:/unavailable-source/" + key + ".npz"
        config["shared"][key]["path"] = text
        foreign.add(text)
    original_stat, original_open = Path.stat, Path.open
    def stat(path, *args, **kwargs):
        assert str(path).replace("\\", "/") not in foreign, "Foreign original locator was probed"
        return original_stat(path, *args, **kwargs)
    def opened(path, *args, **kwargs):
        assert str(path).replace("\\", "/") not in foreign, "Foreign original locator was opened"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "open", opened)
    result = capacity.run_store_budget(config, path_for_use=lambda use, bound: actual[use])
    assert result["source_roles"]["mass.physical_reference"]["path"] in foreign


def test_input_mutation_or_wrong_alias_size_cannot_supply_a_store_bound(configuration, monkeypatch):
    config = configuration
    original = capacity._header
    changed = False
    def mutate(path, **options):
        nonlocal changed
        result = original(path, **options)
        if not changed:
            changed = True
            with path.open("ab") as stream:
                stream.write(b"changed")
        return result
    monkeypatch.setattr(capacity, "_header", mutate)
    with pytest.raises(ValueError, match="changed"):
        capacity.run_store_budget(config, path_for_use=_source)


def test_csv_depth_store_uses_byte_bound_without_parsing_measurements(configuration, tmp_path):
    config = deepcopy(configuration)
    path = tmp_path / "depth.csv"
    path.write_text("x,y,depth\n0,0,0.01\n1,1,0.02\n", encoding="utf8")
    config["shared"]["depth_points"] = producer.asset(path)
    result = capacity.run_store_budget(config, path_for_use=_source)
    depth = result["arrays"]["depth_points"]
    assert depth["payload_bytes"] == 24 * path.stat().st_size
    assert depth["payload_bytes"] >= 2 * 3 * 8 and "shape" not in depth
    assert depth["shape_bound"] == [path.stat().st_size, 3]


def test_explicit_array_alias_uses_actual_format_not_locator_extension(configuration, tmp_path):
    config = configuration
    aliases = {}
    for use, key in zip(capacity.STORE_ROLES[:2], ("physical_reference", "floor_curve"), strict=True):
        path = tmp_path / (key + ".source")
        path.write_bytes(Path(config["shared"][key]["path"]).read_bytes())
        aliases[use] = path
    first = capacity.run_store_budget(config, path_for_use=_source)
    second = capacity.run_store_budget(config, path_for_use=lambda use, bound: aliases[use])
    assert first == second


@pytest.mark.parametrize("change", ["drop", "full", "probability", "unknown_option"])
def test_unsupported_retention_or_phase_one_output_refuses_before_resolving_inputs(configuration, change):
    config = deepcopy(configuration)
    if change == "drop":
        config["storage_profile"]["colour_intermediates"] = "drop"
    elif change == "full":
        config["storage_profile"]["profile"] = "full"
    else:
        config["gas_probability_storage" if change == "probability" else "probability_storage"] = {}
    with pytest.raises(ValueError):
        capacity.run_store_budget(config, path_for_use=lambda *args: pytest.fail("Invalid request resolved an input"))


def test_retained_retry_files_are_counted_against_admission_not_projected_use(configuration, tmp_path, monkeypatch):
    config = configuration
    store = capacity.run_store_budget(config, path_for_use=_source)
    bound = capacity.phase2_lean_budget(config, canvas_hw=(32, 48), store_bound=store)
    assert bound["serialization_attempts"] == len(job_queue.TRANSIENT_RETRY_DELAYS) + 1
    assert bound["admitted_frame_bytes"] == bound["committed_frame_bytes"] * 4
    assert bound["admitted_run_bytes"] == bound["committed_run_bytes"] * 4
    assert bound["admitted_job_bytes"] == bound["admitted_frame_bytes"] * len(config["frames"]) + bound["admitted_run_bytes"]
    assert bound["production_permission_granted"] is False and bound["qualification_completed"] is False
    assert bound["measured"] is False
    original_retry = job_queue.retry_transient
    monkeypatch.setattr(io_ops, "retry_transient", lambda action:
        original_retry(action, delays=(0., 0., 0.), wait=lambda delay: None))
    original = np.lib.format.write_array
    calls = []
    def interrupted(stream, value, **options):
        calls.append(True)
        original(stream, value, **options)
        if len(calls) < 4:
            error = OSError("Synthetic retry after serialization")
            error.winerror = 64
            raise error
    monkeypatch.setattr(np.lib.format, "write_array", interrupted)
    values = np.random.default_rng(11).random((32, 48), dtype=np.float32)
    target = tmp_path / "retry-frame"
    target.mkdir()
    result = producer._write_npz(target / "products.npz", {"mass_total": values})
    files = list(target.iterdir())
    assert len(files) == len(calls) == 4
    from darsia.fluidflower.pipeline.option_capacity import npz_bound
    assert sum(path.stat().st_size for path in files) <= 4 * npz_bound({"mass_total": values.nbytes})
    assert _same_bits(_archive(target / "products.npz")["mass_total"], values)
    assert result["size_bytes"] <= npz_bound({"mass_total": values.nbytes})


@pytest.mark.parametrize("change", ["source", "archive", "metadata", "authority"])
def test_stale_or_tampered_store_bound_refuses(configuration, change):
    config = configuration
    store = capacity.run_store_budget(config, path_for_use=_source)
    if change == "source":
        store["source_roles"]["mass.physical_reference"]["sha256"] = "e" * 64
    elif change == "archive":
        store["archive_bytes"] -= 1
    elif change == "metadata":
        store["metadata_bytes"] -= 1
    else:
        store["production_permission_granted"] = True
    with pytest.raises(ValueError, match="Run-store bound"):
        capacity.phase2_lean_budget(config, canvas_hw=(32, 48), store_bound=store)


def make_sealed_configuration(tmp_path):
    """The existing real saved-stage fixture, including genuine N15/N16 inputs."""
    saved = _module("tests/unit/test_storage_n1_saved_products.py", "_lean_capacity_saved_fixture")
    fixture = _module("tests/fixtures/n1_phase2_fixture.py", "_lean_capacity_stage_fixture")
    config = saved._genuine_n16_config(tmp_path, fixture.tiny_config.__wrapped__(tmp_path),
                                     fixture.phase1_data.__wrapped__(), adjacent=True)
    config.update(precision="float32", storage_profile=dict(profile="lean", colour_intermediates="keep"))
    return config


def test_genuine_pixel_hidden_and_extensive_outputs_fit_the_same_lossless_fallback_contract(tmp_path):
    config = make_sealed_configuration(tmp_path)
    shape = config["frames"][0]["current"]["phase_shape_hw"]
    store = capacity.run_store_budget(config, path_for_use=_source)
    bound = capacity.phase2_lean_budget(config, canvas_hw=shape, store_bound=store)
    full = _run(tmp_path, "full-domain-reference", config, dict(profile="full"))
    lean = _run(tmp_path, "lean-domain", config, dict(profile="lean", colour_intermediates="keep"))
    assert bound["fallback_archives"]["declared"] == ["pixel_domain", "storage"]
    for (old_root, old), (new_root, new) in zip(_frames(full), _frames(lean), strict=True):
        for archive, table in (("products.npz", "product_payloads"), ("native_geometry.npz", "geometry_payloads")):
            arrays = _archive(old_root / archive)
            assert set(arrays) <= set(bound["fallback_archives"][table])
            assert all(value.nbytes <= bound["fallback_archives"][table][name] for name, value in arrays.items())
        products = storage.read_products(new, new_root)
        geometry = storage.read_static_fields(new, new_root)
        assert all(_same_bits(value, _archive(old_root / "products.npz")[name]) for name, value in products.items())
        assert all(_same_bits(value, _archive(old_root / "native_geometry.npz")[name]) for name, value in geometry.items())
        for path in old_root.glob("storage_*.npz"):
            first, second = _archive(path), _archive(new_root / path.name)
            assert first.keys() == second.keys()
            assert all(_same_bits(first[name], second[name]) for name in first)
        assert len(geometry) == 11
        assert new["n16_extensive_producer"]["native_geometry"] == new["static"]
    assert bound["scientific_artifact_bound"] is True
    assert bound["includes_queue_control_history"] is False


def test_actual_lean_metadata_writers_stay_in_scoping_thread(tmp_path, monkeypatch):
    from threading import get_ident
    config = make_sealed_configuration(tmp_path)
    output = tmp_path / "scoped-lean"
    origin = get_ident()
    observed = []
    def instrument(original):
        def publish(path, *args, **kwargs):
            path = Path(path)
            if path.is_relative_to(output) and path.name.endswith((".json", ".json.gz")):
                observed.append((path.name, get_ident(), io_ops.metadata_limit(path)))
            return original(path, *args, **kwargs)
        return publish
    monkeypatch.setattr(io_ops, "publish_file", instrument(io_ops.publish_file))
    monkeypatch.setattr(job_queue, "_publish_bytes", instrument(job_queue._publish_bytes))
    outcome = _run(tmp_path, output.name, config, dict(profile="lean", colour_intermediates="keep"))
    assert outcome["complete"] is True
    assert {name for name, _, _ in observed} >= {
        "manifest.json", "lean_run_static.json", "native_statistics.json.gz", "result.json",
        "commit.json", "storage_completed.json", "storage_extensive_frame.json"}
    assert all(thread == origin and limit == io_ops.LEAN_METADATA_BYTES
               for _, thread, limit in observed)
    assert io_ops.metadata_limit(output / "result.json") is None


def test_pixel_domain_adds_exactly_one_guarded_run_record_for_every_serialization_attempt(tmp_path):
    """The genuine pixel configuration reserves its scalar table beyond both ordinary run records."""
    config = make_sealed_configuration(tmp_path)
    shape = config["frames"][0]["current"]["phase_shape_hw"]
    store = capacity.run_store_budget(config, path_for_use=_source)
    pixel = capacity.phase2_lean_budget(config, canvas_hw=shape, store_bound=store)
    plain_config = deepcopy(config)
    plain_config.pop("pixel_domain")
    plain_config.pop("storage", None)
    plain = capacity.phase2_lean_budget(plain_config, canvas_hw=shape, store_bound=store)
    assert pixel["run_metadata_files"] == plain["run_metadata_files"] + 1 == 3
    assert set(pixel["run_record_files"]) - set(plain["run_record_files"]) == {"mass_table.csv"}
    assert pixel["committed_run_bytes"] - plain["committed_run_bytes"] == io_ops.LEAN_METADATA_BYTES
    assert pixel["admitted_run_bytes"] - plain["admitted_run_bytes"] == (
        pixel["serialization_attempts"] * io_ops.LEAN_METADATA_BYTES)
    assert pixel["serialization_attempts"] == 1 + len(job_queue.TRANSIENT_RETRY_DELAYS)
