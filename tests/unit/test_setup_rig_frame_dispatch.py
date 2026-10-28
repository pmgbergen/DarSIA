"""Synthetic operational contracts; no real rig build or external data."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import setup_rig as dispatch
from .test_series_setup_capacity import _json, prepared_raw_request  # noqa: F401


def _config(tmp_path, *, frame=None, labels="labels.npz", facies="facies.npz"):
    text = f'[labeling]\nlabels = "{labels}"\n'
    if frame is not None:
        text += f'frame = "{frame}"\n'
    text += f'[facies]\npath = "{facies}"\nprops = "facies.csv"\n'
    path = tmp_path / "run.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _image(*, shape=(2, 3), origin=(0.0, 2.0)):
    return SimpleNamespace(
        scalar=True,
        series=False,
        space_dim=2,
        indexing="ij",
        img=np.zeros(shape, dtype=np.uint8),
        dimensions=[2.0, 3.0],
        origin=np.array(origin),
        voxel_size=[1.0, 1.0],
    )


@pytest.mark.parametrize("frame", [None, "raw"])
def test_raw_dispatch_uses_exact_original_class_and_arguments(tmp_path, monkeypatch, frame):
    path = _config(tmp_path, frame=frame)
    calls = []
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda *args: calls.append(args))
    dispatch.setup_rig(dispatch.Rig, path, True)
    assert calls == [(dispatch.Rig, path, True)]


@pytest.mark.parametrize("frame", [None, "raw", "rig-corrected"])
def test_explicit_raw_request_reaches_original_setup_without_changing_frame_guards(
    tmp_path, monkeypatch, frame
):
    """Forward the exact descriptor after the existing corrected-frame admission."""
    from darsia.fluidflower import corrected_setup

    path = _config(tmp_path, frame=frame)
    request = dict(path="C:/synthetic/request.json", sha256="a" * 64, size_bytes=17)
    before = dict(request)
    events = []
    result = object()

    def admitted(actual):
        """Record the original corrected export check before setup dispatch."""
        assert actual is path
        events.append("admitted")

    def original(cls, actual, show, *, raw_build_request):
        """Observe the original callable receiving the unchanged request descriptor."""
        assert actual is path and show is True and raw_build_request is request
        assert cls is dispatch.Rig if frame != "rig-corrected" else issubclass(cls, dispatch.Rig)
        events.append("setup")
        return result

    monkeypatch.setattr(corrected_setup, "require_corrected_export_for_build", admitted)
    # This forwarding test has an intentionally fictional request, not method data.
    monkeypatch.setattr(dispatch, "_phase_map_build_policy", lambda binding: None)
    monkeypatch.setattr(dispatch, "_original_setup_rig", original)
    assert dispatch.setup_rig(dispatch.Rig, path, True, raw_build_request=request) is result
    assert events == (["admitted", "setup"] if frame == "rig-corrected" else ["setup"])
    assert request == before


@pytest.mark.parametrize("method", ["phase_model", "colour_path", None])
def test_raw_method_is_selected_from_exact_bound_series(prepared_raw_request, method):
    _task, config, path, request = prepared_raw_request
    series = Path(request["series_context"]["manifest"]["path"])
    series.write_text('' if method is None else f'[method]\nactive = "{method}"\n')
    from darsia.fluidflower.rig_raw_build import _descriptor
    request["series_context"]["manifest"] = _descriptor(series)
    binding = _json(path, request)
    policy = dispatch._phase_map_build_policy(binding)
    if method == "phase_model":
        assert policy["method"] == method
        assert policy["series_manifest"] == request["series_context"]["manifest"]
    else:
        assert policy is None
    series.write_text('[method]\nactive = "phase_model"\nchanged = true\n')
    with pytest.raises(ValueError, match="changed"):
        dispatch._phase_map_build_policy(binding)


def test_explicit_raw_request_does_not_bypass_corrected_export_refusal(tmp_path, monkeypatch):
    """A request descriptor cannot replace the original corrected-frame check."""
    from darsia.fluidflower import corrected_setup

    path = _config(tmp_path, frame="rig-corrected")

    def refused(actual):
        """Stop at the existing export check without reaching the original builder."""
        raise ValueError("Synthetic corrected export refusal")

    monkeypatch.setattr(corrected_setup, "require_corrected_export_for_build", refused)
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda *args, **kwargs: pytest.fail("Setup reached"))
    with pytest.raises(ValueError, match="corrected export refusal"):
        dispatch.setup_rig(dispatch.Rig, path, raw_build_request=dict(path="C:/synthetic/request.json"))


def test_corrected_dispatch_inherits_original_setup_and_keeps_original_depth_call(
    tmp_path, monkeypatch
):
    path = _config(tmp_path, frame="rig-corrected")
    calls = []
    admissions = []
    from darsia.fluidflower import corrected_setup
    monkeypatch.setattr(corrected_setup, "require_corrected_export_for_build", lambda paths: admissions.append(paths))
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda *args: calls.append(args))
    dispatch.setup_rig(dispatch.Rig, path, False)
    cls, actual_path, show = calls[0]
    assert cls is not dispatch.Rig and issubclass(cls, dispatch.Rig)
    assert cls.setup is dispatch.Rig.setup
    assert cls.setup_depth is dispatch.Rig.setup_depth
    assert cls.setup_facies_props is dispatch.Rig.setup_facies_props
    assert actual_path is path and show is False
    assert admissions == [path]


def test_corrected_virtual_calls_select_original_false_apis_without_setup_copy(monkeypatch):
    calls = []

    class ExistingRig:
        def setup(self):
            self.setup_labels(Path("labels.npz"), apply_corrections=True, log="log")
            self.setup_facies(
                Path("facies.npz"), apply_corrections=True, log="log", show_plot=True
            )

        def setup_labels(self, **kwargs):
            calls.append(("labels", kwargs))

        def setup_facies(self, **kwargs):
            calls.append(("facies", kwargs))

    monkeypatch.setattr(dispatch.darsia, "imread", lambda path: _image())
    cls = dispatch.corrected_rig_class(ExistingRig)
    assert cls.setup is ExistingRig.setup
    rig = cls()
    rig.baseline = _image()
    rig.setup()
    assert calls == [
        ("labels", dict(path=Path("labels.npz"), apply_corrections=False, log="log")),
        (
            "facies",
            dict(path=Path("facies.npz"), apply_corrections=False, log="log", show_plot=True),
        ),
    ]


@pytest.mark.parametrize(
    "image, field",
    [(_image(shape=(3, 2)), "shape"), (_image(origin=(1.0, 2.0)), "origin")],
)
def test_corrected_grid_mismatch_refuses_before_original_resize(monkeypatch, image, field):
    calls = []

    class ExistingRig:
        def setup_labels(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(dispatch.darsia, "imread", lambda path: image)
    rig = dispatch.corrected_rig_class(ExistingRig)()
    rig.baseline = _image()
    with pytest.raises(ValueError, match=rf"grid mismatch \({field}\)"):
        rig.setup_labels(Path("labels.npz"), apply_corrections=True)
    assert calls == []


def test_unknown_frame_refuses_before_original_dispatch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda *args: calls.append(args))
    path = _config(tmp_path, frame="unknown")
    with pytest.raises(ValueError, match="Unsupported labeling.frame"):
        dispatch.setup_rig(dispatch.Rig, path)
    assert calls == []


@pytest.mark.parametrize("labels,facies", [("", "facies.npz"), ("labels.npz", "")])
def test_corrected_requires_explicit_original_fields(tmp_path, labels, facies):
    path = _config(tmp_path, frame="rig-corrected", labels=labels, facies=facies)
    with pytest.raises(ValueError, match="requires an explicit"):
        dispatch.configured_frame(path)


def test_frame_opt_in_uses_original_ordered_config_merge(tmp_path):
    base = _config(tmp_path)
    override = tmp_path / "frame.toml"
    override.write_text('[labeling]\nframe = "rig-corrected"\n', encoding="utf-8")
    assert dispatch.configured_frame([base, override]) == "rig-corrected"


def test_facies_frame_is_rejected_before_original_numeric_id_parser(tmp_path):
    path = _config(tmp_path, frame="rig-corrected")
    path.write_text(path.read_text(encoding="utf-8") + 'frame = "rig-corrected"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="facies.frame is not supported"):
        dispatch.configured_frame(path)


def test_native_cli_raw_routes_original_preset_without_reimplementing_steps(tmp_path, monkeypatch):
    import sys
    from darsia.presets.workflows import user_interface_setup as original

    path = _config(tmp_path)
    calls = []
    monkeypatch.setattr(sys, "argv", ["setup", "--config", str(path), "--all"])
    monkeypatch.setattr(original, "preset_setup", lambda: calls.append("original"))
    dispatch.main()
    assert calls == ["original"]


def test_native_raw_depth_only_does_not_require_unrelated_map_sections(tmp_path, monkeypatch):
    import sys
    from darsia.presets.workflows import user_interface_setup as original

    path = tmp_path / "depth-only.toml"
    path.write_text('[depth]\ndepth_map="original.npz"\n', encoding="utf-8")
    calls = []
    monkeypatch.setattr(sys, "argv", ["setup", "--config", str(path), "--depth"])
    monkeypatch.setattr(original, "preset_setup", lambda: calls.append("original"))
    dispatch.main()
    assert calls == ["original"]


def test_native_cli_corrected_build_only_and_full_setup_refuses_before_dispatch(
    tmp_path, monkeypatch
):
    import sys

    path = _config(tmp_path, frame="rig-corrected")
    calls = []
    monkeypatch.setattr(dispatch, "setup_rig", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["setup", "--config", str(path), "--rig"])
    dispatch.main()
    assert calls == [(dispatch.Rig, [path], False)]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["setup", "--config", str(path), "--all"])
    with pytest.raises(ValueError, match="Build run model only"):
        dispatch.main()
    assert calls == []


@pytest.mark.parametrize("field", ["labels", "facies"])
def test_legacy_generic_scalar_false_npz_is_accepted_without_rewriting(tmp_path, field):
    import darsia

    path = tmp_path / (field + ".npz")
    values = np.arange(6, dtype=np.uint8).reshape(2, 3)
    image = darsia.Image(
        values, space_dim=2, scalar=False, series=False,
        dimensions=[2.0, 3.0], origin=[0.0, 2.0], indexing="ij",
    )
    image.save(path)
    before = path.read_bytes()
    restored = darsia.imread(path)
    assert restored.scalar is False and restored.series is False
    assert restored.img.ndim == restored.space_dim == 2
    assert restored.range_dim == 0
    assert np.array_equal(restored.img, values)
    dispatch._require_exact_grid(path, _image(shape=(2, 3, 3)), field)
    assert path.read_bytes() == before
    assert restored.scalar is False
    assert np.array_equal(restored.img, values)


@pytest.mark.parametrize(
    "scalar, series",
    [(False, False), (True, True), (False, True)],
    ids=["vector", "scalar-series", "generic-series"],
)
def test_real_vector_or_series_archive_is_rejected(tmp_path, scalar, series):
    import darsia

    path = tmp_path / "not-a-single-scalar.npz"
    image = darsia.Image(
        np.zeros((2, 3, 2), dtype=np.uint8), space_dim=2,
        scalar=scalar, series=series, time=[0.0, 1.0] if series else None,
        dimensions=[2.0, 3.0], origin=[0.0, 2.0], indexing="ij",
    )
    image.save(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="must be a single scalar image"):
        dispatch._require_exact_grid(path, _image(), "labels")
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "field, value",
    [
        ("space_dim", 3),
        ("indexing", "ji"),
        ("shape", (3, 2)),
        ("dimensions", [2.0, np.nextafter(3.0, 4.0)]),
        ("origin", np.array([np.nextafter(0.0, 1.0), 2.0])),
        ("voxel_size", [1.0, np.nextafter(1.0, 2.0)]),
    ],
)
def test_legacy_false_flag_does_not_exempt_any_exact_grid_check(monkeypatch, field, value):
    image = _image()
    image.scalar = False
    if field == "shape":
        image.img = np.zeros(value, dtype=np.uint8)
    elif field == "space_dim":
        image.space_dim = value
        image.img = np.zeros((2, 3, 1), dtype=np.uint8)
    else:
        setattr(image, field, value)
    monkeypatch.setattr(dispatch.darsia, "imread", lambda path: image)
    with pytest.raises(ValueError, match=rf"grid mismatch \({field}\)"):
        dispatch._require_exact_grid(Path("labels.npz"), _image(), "labels")
