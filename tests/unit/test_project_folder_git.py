"""A series project folder may be its own Git repository (local fixtures only).

Only the DarSIA source checkout itself is refused as a place for per-user and
generated project files; any other Git working tree is an ordinary project.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import reviewer_preferences as preferences
from darsia.fluidflower import series_paths
from darsia.fluidflower.pages.image_correction.selection_dialog import (
    ImageSelectionPlanDialog,
)


def _imported_from(tmp_path, monkeypatch, *parts):
    """Pretend the DarSIA package is imported from ``tmp_path/<parts>/darsia``."""
    package = tmp_path.joinpath(*parts, "darsia", "fluidflower")
    package.mkdir(parents=True)
    monkeypatch.setattr(series_paths, "package_root", lambda: package)
    return package


def _checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "darsia_checkout"
    (checkout / ".git").mkdir(parents=True)
    _imported_from(tmp_path, monkeypatch, "darsia_checkout", "src")
    return checkout


def _git_project(tmp_path):
    project = tmp_path / "series_repository"
    (project / ".git").mkdir(parents=True)
    return project.resolve()


def test_source_checkout_is_the_git_tree_that_holds_the_imported_package():
    package = Path(series_paths.__file__).resolve().parents[1]
    expected = next(
        (folder for folder in package.parents if (folder / ".git").exists()), None
    )
    assert series_paths.darsia_source_checkout() == expected


def test_fake_source_layout_is_detected_as_the_checkout(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path, monkeypatch)
    assert series_paths.darsia_source_checkout() == checkout


def test_installed_copy_has_no_source_checkout_even_inside_a_repository(
    tmp_path, monkeypatch
):
    (tmp_path / "project" / ".git").mkdir(parents=True)
    _imported_from(tmp_path, monkeypatch, "project", ".venv", "Lib", "site-packages")
    assert series_paths.darsia_source_checkout() is None


def test_no_git_ancestor_means_no_source_checkout(tmp_path, monkeypatch):
    _imported_from(tmp_path, monkeypatch, "unpacked", "src")
    checkout = series_paths.darsia_source_checkout()
    # Only a Git folder above the temporary tree (none in a plain sandbox).
    assert checkout is None or not checkout.is_relative_to(tmp_path)


def test_reviewer_name_is_kept_in_a_project_that_is_its_own_git_repository(
    tmp_path, monkeypatch
):
    _checkout(tmp_path, monkeypatch)
    project = _git_project(tmp_path)
    path = preferences.save_reviewer_name(project, " Reviewer ")
    assert path == project / "config" / "phase_model_studio_reviewer.json"
    assert preferences.load_reviewer_name(project) == "Reviewer"


def test_reviewer_name_is_also_kept_in_a_subfolder_of_another_repository(
    tmp_path, monkeypatch
):
    _checkout(tmp_path, monkeypatch)
    project = _git_project(tmp_path) / "DarSIA"
    project.mkdir()
    preferences.save_reviewer_name(project, "Reviewer")
    assert preferences.load_reviewer_name(project) == "Reviewer"


def test_reviewer_preferences_refuse_the_darsia_source_checkout(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path, monkeypatch)
    inside = checkout / "examples" / "series"
    inside.mkdir(parents=True)
    with pytest.raises(ValueError, match="outside the DarSIA source checkout"):
        preferences.save_reviewer_name(inside, "Reviewer")
    with pytest.raises(ValueError, match="outside the DarSIA source checkout"):
        preferences.load_reviewer_name(checkout)
    assert not (inside / "config").exists()


@pytest.mark.parametrize(
    "label,folder", [("TOML plan", "config"), ("new JSON receipt", "results")]
)
def test_image_correction_plans_may_be_saved_in_a_git_project(
    tmp_path, monkeypatch, label, folder
):
    _checkout(tmp_path, monkeypatch)
    project = _git_project(tmp_path)
    target = project / folder / "plan.toml"
    dialog = SimpleNamespace(project_root=project)
    assert (
        ImageSelectionPlanDialog._output_path(dialog, str(target), label=label)
        == target
    )


@pytest.mark.parametrize(
    "label,folder", [("TOML plan", "config"), ("new JSON receipt", "results")]
)
def test_image_correction_plans_are_never_saved_in_the_darsia_checkout(
    tmp_path, monkeypatch, label, folder
):
    checkout = _checkout(tmp_path, monkeypatch)
    project = (checkout / "series").resolve()
    project.mkdir()
    dialog = SimpleNamespace(project_root=project)
    with pytest.raises(ValueError, match="outside the DarSIA source checkout"):
        ImageSelectionPlanDialog._output_path(
            dialog, str(project / folder / "plan.toml"), label=label
        )
