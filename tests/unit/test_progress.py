from darsia.presets.workflows.analysis.progress import normalize_progress_event


def test_normalize_progress_event_payloads() -> None:
    valid = normalize_progress_event(
        {
            "event": "image_progress",
            "step": "mass",
            "image_path": "/tmp/img.png",
            "image_index": 2,
            "image_total": 10,
            "image_duration_s": 1.2,
            "step_elapsed_s": 8.4,
        }
    )
    assert valid is not None
    assert valid["event"] == "image_progress"
    assert valid["step"] == "mass"
    assert valid["image_index"] == 2
    assert valid["image_total"] == 10
    assert normalize_progress_event({"event": "invalid", "step": "mass"}) is None
    assert normalize_progress_event("not-a-dict") is None
