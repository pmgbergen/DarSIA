"""P1 consumes the precise real Setup optics package, with no second path fit."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest

from darsia.fluidflower.mass.n1_pipeline import asset, _write_json
from darsia.fluidflower.pipeline import current_build_job as subject


@pytest.mark.parametrize("change", [None, "image-basis", "another-run", "legacy-pooled", "signal-scale"])
def test_p1_reuses_setup_package_and_corrections_once(tmp_path, monkeypatch, change):
    from darsia.fluidflower.pipeline import (current_build_preparation, current_statistics_recipe,
        phase_statistics, local_build, current_clearwater)
    from darsia.fluidflower import store_map
    root, series = tmp_path / "job", tmp_path / "series"
    root.mkdir(); series.mkdir()
    manifest = series / "series.toml"
    manifest.write_text("# bound test series\n")
    selected = _write_json(root / "selection.json", dict(input_authority=dict(current_series=asset(manifest))))
    base = dict(source_binding={"source": "same correction"}, baseline={"source": "same pixels"})
    basis = _write_json(root / "image_basis.json", dict(runs={"sample": base}))
    bounds = dict(min_value_aq=0., max_value_aq=1., min_value_g=0., max_value_g=1.)
    from darsia.fluidflower.mass.pooled_optical_paths import SETUP_METHOD
    from darsia.fluidflower.pipeline.setup_colour_paths import SIGNAL_AXIS_CONTRACT
    axis = dict(contract=SIGNAL_AXIS_CONTRACT, scales_by_facies={"1":2.},
        source_cohort=["sample","another-run"], frozen_for_production=True,
        units="corrected-clearwater-relative-rgb-euclidean-distance",
        scale_rule="maximum-supported-path-length-per-facies")
    def package_for(run):
        metadata = _write_json(root / (run + "_optical_metadata.json"), dict(flash_bounds=bounds,
            recipe=dict(method=SETUP_METHOD,fit_runs=[run],runs={run:{}}),source_bindings={run:{}},
            signal_axis=deepcopy(axis),path_audit={"1":dict(status="supported")}))
        distance = 1. if run == "sample" else 2.
        native = dict(color_path=dict(colors=[[0.,0.,0.],[distance,0.,0.]]),values=[0.,distance/2])
        return dict(metadata=metadata, color_paths={"1": _write_json(root/(run+"-path.json"), native)},
                   signals={"1": _write_json(root/(run+"-signal.json"), {"run":run})},
                   flash=_write_json(root/(run+"-flash.json"), {"run":run}))
    package = package_for("sample")
    other_package = package_for("another-run")
    source = dict(base, changed=True) if change == "image-basis" else base
    paths_value = dict(format="darsia-setup-colour-paths-v2",scope="per-run",complete=True,
        runs=["sample","another-run"],operations={"sample": {}}, input=dict(image_basis={"sample": source}),
        signal_axis=deepcopy(axis),packages_by_run={"sample":package,"another-run":other_package})
    if change == "another-run":
        paths_value["packages_by_run"]["sample"] = other_package
    elif change == "legacy-pooled":
        paths_value.update(format="darsia-setup-colour-paths-v1",package=package)
        paths_value.pop("packages_by_run")
    elif change == "signal-scale":
        paths_value["signal_axis"]["scales_by_facies"]["1"] = 1.
    paths = _write_json(root / "paths.json", paths_value)
    job = dict(job_id="test", execution_mode=subject.MODE, pipeline="native-phase-statistics", target_machine="Main",
        expected_runs=["sample"], inputs=[dict(run="sample", operation_result={}, metric_model={})],
        series_id="test", series_root=str(series), results_root=str(tmp_path / "results" / "test"),
        selection=selected, setup_colour_paths=paths, setup_image_basis=basis)
    monkeypatch.setattr(store_map, "read_store_map", lambda _: SimpleNamespace(roots=lambda _: {"results": str(tmp_path / "results")}))
    monkeypatch.setattr(subject, "_capacity", lambda *args: nullcontext())
    calls = []
    def statistics(*args, **kwargs):
        assert kwargs["fresh_images"] is True
        return dict(statistics=dict(shared=dict(optics=None, clearwater=None)),
                    flash_bounds=bounds, clearwater_source={"actual": "chosen clearwater"})
    monkeypatch.setattr(current_statistics_recipe, "retained_statistics_recipe", statistics)
    def prepare(*args, **kwargs):
        assert kwargs["prepared_photometry"] == base
        calls.append("prepare")
        return dict(produced={"one": SimpleNamespace(receipt={})}, prepared_native={"one": {"same": "native"}})
    monkeypatch.setattr(current_build_preparation, "prepare_current_build_acquisitions", prepare)
    def clear(*args, **kwargs):
        calls.append("clear")
        return dict(clearwater={"same": "prepared wash"})
    monkeypatch.setattr(current_clearwater, "prepare_current_clearwater", clear)
    def config(prepared, selection, statistics):
        assert statistics["statistics"]["shared"]["optics"] == package
        assert statistics["statistics"]["shared"]["clearwater"] == {"same": "prepared wash"}
        return dict(frames=[dict(case_id="one")])
    monkeypatch.setattr(subject, "phase_configuration", config)
    monkeypatch.setattr(local_build, "LocalBuildInputs", lambda *args, **kwargs: "same local physical inputs")
    def phase(*args, **kwargs):
        calls.append("p1")
        return {"complete": True}
    monkeypatch.setattr(phase_statistics, "run_phase1_local", phase)
    monkeypatch.setattr(subject, "_run_fresh_optical_job", lambda *args: pytest.fail("Setup paths must not be fitted again"))
    if change:
        error = {"image-basis":"image basis", "another-run":"different optical sources",
                 "legacy-pooled":"separately for each run", "signal-scale":"different shared signal scale"}[change]
        with pytest.raises(ValueError, match=error):
            subject._run_current_build_run(job, job["inputs"][0], "a" * 40, None, lambda *_: None)
        assert calls == []
    else:
        assert subject._run_current_build_run(job, job["inputs"][0], "a" * 40, None, lambda *_: None)["complete"]
        assert calls == ["prepare", "clear", "p1"]
