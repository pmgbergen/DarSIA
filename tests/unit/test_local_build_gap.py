"""Execute the actual private collector gap branch with tiny fixture arrays.

No RAW files, Build constructor, inference model, jobs or product writes run.
Only top-level package bootstrap is replaced, to avoid live cache admission;
the complete collector module and its dependency modules are really imported.
"""
from dataclasses import dataclass
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


PRIVATE = Path(__file__).resolve().parents[2] / "src/darsia/fluidflower"
# Dependencies come from this repository's own source tree.
BASE = PRIVATE.parents[1]


@pytest.fixture
def subject(monkeypatch):
    monkeypatch.syspath_prepend(str(BASE))
    package = ModuleType("darsia")
    package.__path__ = [str(BASE / "darsia")]
    package.__spec__ = importlib.machinery.ModuleSpec("darsia", loader=None, is_package=True)
    monkeypatch.setitem(sys.modules, "darsia", package)
    parent = importlib.import_module("darsia.fluidflower.pipeline")
    name = "darsia.fluidflower.pipeline.build_raw_phase"
    path = PRIVATE / "pipeline/build_raw_phase.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(parent, "build_raw_phase", module, raising=False)
    assert Path(module.__file__).resolve() == path.resolve()
    return module


@dataclass
class Context:
    origin_xy: tuple
    rig_domain: np.ndarray
    hardware_excluded: np.ndarray
    modelled: np.ndarray
    filled_observed: np.ndarray
    filled_unobserved: np.ndarray
    parameters: dict


@pytest.fixture
def collector(subject, monkeypatch, tmp_path):
    gap = np.array([[False, True], [False, False]], dtype=bool)
    domain = np.ones((2, 2), dtype=bool)
    context = Context((0., 0.), domain, np.zeros_like(gap), domain.copy(),
                      domain.copy(), np.zeros_like(gap), {})
    original_frame = {"context": context}
    events, fault = [], {"fill_gap": False}
    provenance = {"fixture_approved_gap": True}
    exclusions = importlib.import_module("darsia.fluidflower.painted_analysis_exclusions")

    def transport(build_context, reader, matrix, origin, shape, parameters):
        assert "approved_analysis_gap" in build_context.provenance
        assert origin == (0., 0.) and shape == (2, 2)
        events.append("transport")
        return gap.copy(), provenance.copy()

    def infer(frame, previous, models, parameters):
        events.append("infer")
        # The actual restrict_phase_domain function has removed the gap.
        assert not frame["context"].rig_domain[gap].any()
        assert frame["physical_rig_domain"].all()
        labels = np.zeros((2, 2), dtype=np.uint8)
        labels[gap] = 0 if fault["fill_gap"] else 255
        return {"labels": labels, "classes": {}, "qa": {}}

    def outputs(stage, spec, frame, inferred):
        events.append("products")
        assert np.all(inferred["labels"][gap] == 255)
        assert np.array_equal(inferred["classes"]["approved_analysis_gap"], gap)
        assert inferred["classes"]["physical_rig_domain"].all()
        assert inferred["qa"]["approved_analysis_gap"] == provenance
        return {"fixture_product": True}

    def write_receipt(path, value):
        events.append("receipt")
        assert value["complete"] is True
        return {"fixture_receipt": True}

    monkeypatch.setattr(exclusions, "transport_build_exclusion", transport)
    monkeypatch.setattr(subject, "infer_labels", infer)
    monkeypatch.setattr(subject, "_phase_outputs", outputs)
    monkeypatch.setattr(subject, "_write_json", write_receipt)
    monkeypatch.setattr(subject, "asset", lambda path: {"fixture_source": str(path)})
    spec = {"case_id": "frame31", "source": {"elapsed_h": .497}}
    bridge_frame = SimpleNamespace(receipt={"fixture_bridge_receipt": True})
    instance = subject.BuildRawPhaseCollector.__new__(subject.BuildRawPhaseCollector)
    instance._closed = instance._failed = False
    instance.pending = None
    instance._index = 0
    instance._selected = [(spec, bridge_frame)]
    instance.output = tmp_path / "unwritten_products"
    instance.bridge = SimpleNamespace(
        context=SimpleNamespace(provenance={"approved_analysis_gap": provenance}),
        reader=SimpleNamespace(json=lambda pin: {"build_to_frame_native_h": np.eye(3).tolist()}),
        proof={"fixture_bridge": True})
    instance.reader = object()
    instance._geometry_parameters = {}
    instance.owner = SimpleNamespace(
        native=lambda spec, stage: {"record": {}, "record_asset": {"fixture_native": True}},
        frame=lambda spec, native: original_frame, previous={"fixture_previous": True},
        model_bundle={}, parameters={})
    instance.configuration_binding = {"fixture_configuration": True}
    instance.configuration = {"phase": {"configuration": {}}, "native": {"configuration": {}}}
    instance.revalidate = lambda: None
    instance._same_render = lambda record, frame: None
    instance._pin_scientific = lambda record: None
    instance._pin_tree = lambda value: None
    return SimpleNamespace(value=instance, events=events, fault=fault, gap=gap,
                           original_frame=original_frame)


def test_actual_next_preserves_gap_255_and_reaches_products(collector):
    state = collector
    row = next(state.value)
    assert state.events == ["transport", "infer", "products", "receipt"]
    assert row is state.value.pending and state.value._index == 0
    assert not state.value._failed
    assert np.all(row["inferred"]["labels"][state.gap] == 255)
    assert state.original_frame["context"].rig_domain.all()
    assert not state.value.output.exists()
    with pytest.raises(RuntimeError, match="Acknowledge"):
        next(state.value)


def test_actual_next_rejects_filled_gap_before_products_or_receipt(collector):
    state = collector
    state.fault["fill_gap"] = True
    with pytest.raises(ValueError, match="filled approved unquantified analysis gaps"):
        next(state.value)
    assert state.events == ["transport", "infer"]
    assert state.value.pending is None and state.value._index == 0
    assert state.value._failed
    assert state.original_frame["context"].rig_domain.all()
    assert not state.value.output.exists()
