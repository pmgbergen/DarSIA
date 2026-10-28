"""The published-dataset code holds no name from the user's local deny-list.

The deny-list lives only in the user's publication settings, never in code;
without one this check has nothing to compare and is skipped.
"""

from pathlib import Path

import pytest

import darsia.fluidflower.published_dataset as package
from darsia.fluidflower.published_dataset.scan import NameScanner
from darsia.fluidflower.published_dataset.settings import load_deny_list

ROOT = Path(package.__file__).resolve().parents[4]
CHECKED = [Path(package.__file__).parent, Path(__file__).parent,
           ROOT / "tests" / "unit" / "gui" / "test_published_dataset_build_dialog.py"]


def checked_files():
    """The source and test files of the published-dataset package."""
    for root in CHECKED:
        yield from ([root] if root.is_file() else sorted(path for path in root.rglob("*.py")))


def test_code_holds_no_deny_listed_name():
    deny = load_deny_list()
    if not deny:
        pytest.skip("No local deny-list in the publication settings")
    scanner = NameScanner(deny=deny)
    hits = []
    for path in checked_files():
        relative = path.relative_to(ROOT).as_posix()
        hits += [hit for hit in scanner.path(relative, file=relative) if hit.kind == "name"]
        hits += [hit for hit in scanner.content(path.read_bytes(), file=relative, suffix=".txt")
                 if hit.kind == "name"]
    assert not hits, "\n".join(hit.describe() for hit in hits)
