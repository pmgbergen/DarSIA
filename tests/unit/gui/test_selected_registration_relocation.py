import json

from darsia.fluidflower.ui import model_training


def test_selected_registration_is_relocated_at_gui_boundary_without_changing_frozen_reader(tmp_path, monkeypatch):
    from darsia.fluidflower import phase_saved_registration as registration
    from darsia.fluidflower import phase_saved_selected_registration as selected
    from darsia.fluidflower import phase_saved_review_metadata as metadata
    before = metadata.__file__
    frozen_bytes = open(before, "rb").read()
    old = tmp_path / "old" / "registrations" / "model.json"
    moved = tmp_path / "actual-results" / "registrations" / "model.json"
    moved.parent.mkdir(parents=True)
    moved.write_text(json.dumps(dict(registration_kind="ffac_selected_camera_policy_a_p1_final3_review")))
    monkeypatch.setattr(registration, "registration_path", lambda model_id: old)
    monkeypatch.setattr(metadata, "configured_parameters", lambda section: dict(directory="registrations"))
    monkeypatch.setattr(model_training, "resolve_review_path", lambda path: moved if path == old else path)
    called = []
    def saved(root, *, results):
        called.append(results)
        return dict(model_identity=["model", "identity"], pipeline_sha256="actual-pipeline")
    monkeypatch.setattr(selected, "selected_saved_review", saved)
    value = model_training._registered_saved_phase_pipeline(tmp_path / "model",
        dict(approved_g3_gas=True), ["model", "identity"])
    assert value == "actual-pipeline" and called == [tmp_path / "actual-results"]
    assert open(before, "rb").read() == frozen_bytes
