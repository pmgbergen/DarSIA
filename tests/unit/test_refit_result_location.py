from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import job_filesystem
from darsia.fluidflower.pipeline import local_refit


@pytest.mark.parametrize("location", [r"C:\experiment\results", r"\\lab-server\experiment\results"])
def test_refit_accepts_the_existing_results_admission_for_local_and_shared_storage(monkeypatch, location):
    calls = []

    def inspect(path):
        calls.append(path)
        return SimpleNamespace(allowed=True, results_root=location)

    monkeypatch.setattr(job_filesystem, "inspect_results_root", inspect)
    assert local_refit._local(location) == Path(location)
    assert calls == [Path(location)]


def test_refit_rejects_an_unavailable_results_location_without_creating_it(monkeypatch, tmp_path):
    destination = tmp_path / "unavailable"
    monkeypatch.setattr(job_filesystem, "inspect_results_root", lambda path: SimpleNamespace(
        allowed=False, reason="Unsupported filesystem"))
    with pytest.raises(ValueError, match="Unsupported filesystem"):
        local_refit._local(destination)
    assert not destination.exists()
