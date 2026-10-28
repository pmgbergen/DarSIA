"""Coalesced native metadata lookup, with deliberately out-of-order completion."""
from darsia.fluidflower import calibration_review_native_catalog_ui as catalog
from .test_calibration_review_panels import _Signal


def test_late_native_catalogue_cannot_repopulate_another_series(qtbot, monkeypatch, tmp_path):
    workers = []
    class Lookup:
        def __init__(self, generation, context, registry, machine, parent):
            self.generation, self.context = generation, context
            self.ready, self.finished = _Signal(), _Signal()
            self.interruptions = self.waits = 0
            workers.append(self)
        def start(self): pass
        def requestInterruption(self): self.interruptions += 1
        def wait(self): self.waits += 1
        def deleteLater(self): pass
    monkeypatch.setattr(catalog, "_Lookup", Lookup)
    controller, delivered = catalog.NativeReviewCatalogue(), []
    controller.ready.connect(lambda *args: delivered.append(args))
    controller.refresh("ffac", tmp_path / "ac")
    controller.refresh("ffum", tmp_path / "um")
    assert len(workers) == 1 and workers[0].interruptions == 1
    workers[0].ready.emit(workers[0].generation, workers[0].context, ["old"], [])
    assert delivered == []
    workers[0].finished.emit()
    assert len(workers) == 2
    workers[1].ready.emit(workers[1].generation, workers[1].context, ["current"], [])
    assert delivered == [(workers[1].context, ["current"], [])]
    controller.stop()
    assert workers[1].waits == 1
    workers[1].ready.emit(workers[1].generation, workers[1].context, ["late"], [])
    assert len(delivered) == 1
