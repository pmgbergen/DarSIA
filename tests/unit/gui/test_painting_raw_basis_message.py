"""The existing RAW refusal is visible before saved painting/photo decoding."""
from darsia.fluidflower import series_layer_view as view


def test_missing_raw_basis_refuses_before_drawing_or_any_preparation(qtbot, tmp_path, monkeypatch):
    called, failures, completed = [], [], []
    def forbidden(*args, **kwargs):
        called.append("draw")
        raise AssertionError("A missing RAW basis must refuse before saved-map/photo drawing")
    monkeypatch.setattr(view.lpm, "version_layers", forbidden)
    monkeypatch.setattr(view.lpp, "version_layers", forbidden)
    destination = tmp_path / "uncreated-preparation"
    worker = view._AdoptionWorker({"run_id": "sample"}, None,
        {"geometry_index": tmp_path / "missing-index.json", "runs": ["sample"],
         "destination": destination})
    worker.failed.connect(failures.append)
    worker.finished.connect(completed.append)
    worker.run()
    assert len(failures) == 1
    assert "The rigs need their RAW basis before the map can be adopted" in failures[0]
    assert not completed and not called
    assert not destination.exists()
