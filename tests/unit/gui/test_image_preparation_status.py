"""Only real completed preparation may be presented as ready."""
from threading import Event

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from darsia.fluidflower.image_preparation_status import ImagePreparationStatus


class Choice(QWidget):
    context_updated = Signal()

    def __init__(self):
        super().__init__()
        self.series_root, self.series_id, self.original = "series-a", "a", b"saved choices"


def test_saved_choices_do_not_imply_completed_processing(qtbot):
    choice = Choice()
    qtbot.addWidget(choice)
    result = dict(complete=False, text="Image preparation: Pending")
    panel = ImagePreparationStatus(choice, loader=lambda *args: result.copy())
    qtbot.addWidget(panel)
    qtbot.waitUntil(lambda: not panel.reader.busy)
    assert panel.status.text() == "Image preparation: Pending"
    assert not panel.status.styleSheet()
    result.update(complete=True, text="Image preparation completed for 2 runs")
    panel.refresh_button.click()
    qtbot.waitUntil(lambda: not panel.reader.busy)
    assert "completed" in panel.status.text()
    assert "color:" in panel.status.styleSheet()


def test_old_series_completion_cannot_mark_new_series_ready(qtbot):
    choice = Choice()
    qtbot.addWidget(choice)
    entered, release, retired = Event(), Event(), Event()
    def read(root, *_args):
        if root == "series-a":
            entered.set()
            release.wait(3)
            retired.set()
            return dict(complete=True, text="Old series ready")
        return dict(complete=False, text="New series pending")
    panel = ImagePreparationStatus(choice, loader=read)
    qtbot.addWidget(panel)
    try:
        qtbot.waitUntil(entered.is_set)
        choice.series_root, choice.series_id = "series-b", "b"
        choice.context_updated.emit()
        qtbot.waitUntil(lambda: not panel.reader.busy)
        release.set()
        qtbot.waitUntil(retired.is_set)
        qtbot.wait(20)
        assert panel.status.text() == "New series pending"
        assert not panel.status.styleSheet()
    finally:
        release.set()


def test_unsaved_form_cannot_keep_green_completed_status(qtbot, tmp_path):
    from .test_image_preparation_preview import choice as make_choice
    choice, _path = make_choice(qtbot, tmp_path)
    choice.save_button.click()
    qtbot.waitUntil(choice.can_close)
    calls = []
    def load(*args):
        calls.append(args)
        return dict(complete=True, text="Prepared saved choices")
    panel = ImagePreparationStatus(choice, loader=load)
    qtbot.addWidget(panel)
    qtbot.waitUntil(lambda: not panel.reader.busy)
    assert "color:" in panel.status.styleSheet()
    choice.stable_band.setValue(4)
    assert "Unsaved" in panel.status.text()
    assert not panel.status.styleSheet()
    assert len(calls) == 1


def test_status_resolves_canonical_filter_to_manifest_jobs_owner(tmp_path):
    import toml
    from darsia.fluidflower.image_preparation_status import PreparationStatusSource
    from darsia.fluidflower.image_normalisation import choice_record
    source = PreparationStatusSource()
    calls = []
    class Registry:
        def jobs(self, root, series):
            calls.append((root, series))
            return []
    source.registry = Registry()
    document = dict(series_id="SAMPLE_X", method=dict(active="phase_model"), runs=dict(ids=["r1"]),
        image_normalisation=choice_record("SAMPLE_X", ["r1"], "off", color_method="off"))
    original = toml.dumps(document).encode()
    result = source(tmp_path, "samplex", original)
    assert calls == [(tmp_path, "SAMPLE_X")]
    assert result["complete"] is False
    assert "No completed image correction" in result["text"]


def test_completed_preparation_is_reported_as_ready(qtbot):
    from darsia.fluidflower.normalisation_step import NormalisationStep
    states = []

    class Probe:
        ready = False
        _emit_ready = NormalisationStep._emit_ready
        prepared_completion = NormalisationStep.prepared_completion
        _publish_ready = NormalisationStep._publish_ready

        class readiness_changed:
            @staticmethod
            def emit(value):
                states.append(value)

    probe = Probe()
    probe._emit_ready(False)  # Historical lighting evidence is not saved.
    probe.prepared_completion(True)  # The Setup job completed this image basis.
    assert probe.ready is True and states == [False, True]
    probe.prepared_completion(False)
    assert probe.ready is False
