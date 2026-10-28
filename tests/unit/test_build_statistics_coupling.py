"""Tiny actual saved fields and optical models through the N1 statistics path."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import darsia

from darsia.fluidflower.mass import n1_pipeline
from darsia.fluidflower.mass.production_precision import PixelMath
from darsia.fluidflower.phase.inputs import TemporalNativeInput
from darsia.fluidflower.pipeline import runner, series
from .test_local_build_inputs import local_build_case, local_build_sources, _prepared_case
from .test_native_optical_source import saved, _json
from .test_staged_data import phase1_data
from ..fixtures.tiny_saved_model_fixture import tiny_config


def test_actual_saved_build_and_optical_models_extract_once_without_calibration(
        local_build_case, tiny_config, tmp_path, monkeypatch):
    case = local_build_case
    old = deepcopy(tiny_config[0])
    old_frame = old["frames"][0]
    # The saved optical basis must explicitly contain the adopted Freewater
    # ID, even when its optical response is ignored. Never borrow a sand ID.
    colour = tmp_path / "freewater_colour"
    darsia.ColorPathInterpolation(darsia.ColorPath(colors=[np.ones(3), np.full(3, .8)]),
        darsia.ColorMode.RELATIVE, [0., 1.]).save(colour)
    signal = tmp_path / "freewater_signal"
    darsia.PWTransformation([0., 1.], [0., 1.]).save(signal)
    optics = old["shared"]["optics"]
    optics["color_paths"]["0"] = n1_pipeline.asset(colour.with_suffix(".json"))
    optics["signals"]["0"] = n1_pipeline.asset(signal.with_suffix(".csv"))
    metadata = n1_pipeline.DataReader().json(optics["metadata"])
    metadata.update(label_ids=[0, 1, 2], ignore_labels=[0, 2])
    optics["metadata"] = _json(tmp_path / "local_optics.json", metadata)
    selected = case.configuration["statistics"]["shared"]
    old["shared"].update({key: selected[key] for key in
        ("porosity_by_facies", "swi_by_facies", "unquantified_facies")})
    old["geometry_fields"] = dict(source_to_native="H",
        original_domain_to_native="H_ac28_to_frame_native",
        carried_homography_provenance="original_H_ac28_to_native")
    case.configuration["statistics"] = {key: value for key, value in old.items()
        if key not in {"format", "run", "frames", "calibration"}}
    case.spec["mass_spec"] = {key: value for key, value in old_frame.items()
        if key not in {"case_id", "frame_index", "elapsed_h", "source_binding",
                       "current", "baseline", "labels", "classes", "area"}}
    case.phase_frame["temporal_input"] = TemporalNativeInput(
        case.phase_frame["full_log"], np.ones((3, 3), bool), (0, 0), {})
    adapter, bound, frame, inferred, _ = _prepared_case(case)
    inferred.update(seam_masks={}, post_masks={})
    products = runner._phase_outputs(tmp_path / "phase", case.spec, frame, inferred)
    baseline = deepcopy(old_frame["baseline"])
    baseline["binding"]["geometry"]["H"] = np.eye(3).tolist()
    spec = series.bind_mass_frame(case.spec, case.native["record"], products, baseline=baseline)
    config = dict(bound["statistics"], run=bound["run"], frames=bound["frames"], data_aliases=())
    pipeline = n1_pipeline.N1Pipeline(config, precision="float32", pixel_math=PixelMath("float32"),
        statistics_only=True, local_build=adapter)
    calls = []
    loaded = []
    original = pipeline.load_frame
    def load(value):
        calls.append(value["case_id"])
        result = original(value)
        loaded.append(result[0])
        return result
    monkeypatch.setattr(pipeline, "load_frame", load)
    result = pipeline.extract_statistics(spec)
    assert calls == [case.spec["case_id"]]
    assert "calibration" not in pipeline.config
    report = result["statistics"]["local_build_accounting"]
    assert report["reporting"]["pixel_count"] == 9
    assert report["excluded"]["material_2"]["phase255_pixels"] == 2
    assert report["known_zero_porosity_material"]["pixel_count"] == 2
    assert np.array_equal(loaded[0].facies, case.sources.facies)
    assert pipeline._local_prepared is None


def test_original_preparation_chart_artifact_matches_saved_effective_labels(tmp_path):
    from darsia.fluidflower import painting_chart_transfer as chart
    from darsia.fluidflower.pipeline.local_build_chart import read_chart
    from .test_painting_chart_transfer import inputs
    proposal = chart.propose_chart_transfer("ac23", **inputs())
    approval = chart.confirm_chart_transfer(proposal, actor="Reviewer", reason="Confirm the displayed mask")
    arrays = chart.apply_chart_transfer(proposal, approval)
    folder = tmp_path / "ac23"
    folder.mkdir()
    archive = folder / "chart_labels.npz"
    np.savez(archive, **arrays)
    artifact = n1_pipeline.asset(archive)
    artifact["path"] = "ac23/chart_labels.npz"
    preparation = _json(tmp_path / "index.json", dict(chart_transfers={"ac23": dict(
        artifact=artifact, proposal=proposal.metadata(), approval=approval)}))
    context = SimpleNamespace(run="ac23", labels=arrays["effective_labels"], provenance={})
    mask, provenance = read_chart({"preparation": preparation}, reader=n1_pipeline.DataReader(), context=context)
    np.testing.assert_array_equal(mask, arrays["chart_mask"])
    assert provenance["approval"] == approval
    context.labels = context.labels.copy()
    context.labels[0, 0] = 17
    with pytest.raises(ValueError, match="saved Build"):
        read_chart({"preparation": preparation}, reader=n1_pipeline.DataReader(), context=context)


def test_worker_hands_actual_build_to_n1_before_any_frame_execution(local_build_case, tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import phase_statistics
    case = local_build_case
    adapter = case.make_adapter()
    checked = SimpleNamespace(physical_inputs=lambda: adapter)
    inputs = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(series, "SeriesInputs", lambda *args, **kwargs: inputs)
    class ReachedNativeStatistics(Exception):
        pass
    def native(config, **options):
        assert options["local_build"] is adapter
        assert "current_build" not in options
        assert config["physical_correction"] == case.sources.operation_result
        assert config["frames"] == case.configuration["frames"]
        raise ReachedNativeStatistics
    monkeypatch.setattr(n1_pipeline, "N1Pipeline", native)
    with pytest.raises(ReachedNativeStatistics):
        phase_statistics._phase1_frames(case.configuration, {}, tmp_path, "cpu", "float32",
            None, {}, {}, 0., None, current_build=checked)


def test_metric_model_is_part_of_the_original_current_build_input_binding():
    from darsia_cache.current_build_binding import checked_current_build_binding, current_build_binding_inputs
    from .test_current_build_binding import binding_TESTONLY, descriptor_TESTONLY
    binding, config = binding_TESTONLY()
    metric = descriptor_TESTONLY("measured-model.json")
    binding["metric_model"] = metric
    assert checked_current_build_binding(binding, config=config)["metric_model"] == metric
    assert ("current_build.metric_model", metric, "configuration") in list(current_build_binding_inputs(binding))
    binding["metric_model"] = {"path": metric["path"]}
    with pytest.raises(ValueError):
        checked_current_build_binding(binding, config=config)
