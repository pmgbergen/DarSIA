"""Alias lookup by the path's own folders gives the same result as trying every alias."""
from pathlib import Path
import random

import pytest

from darsia.fluidflower.storage_paths import _longest_source, _normalized, _resolve_with


def _brute_force(path, rows):
    """The original longest-first scan over every alias row."""
    result = _normalized(path)
    seen, applied = set(), set()
    while True:
        key = result.casefold()
        if key in seen:
            raise ValueError("cycle")
        seen.add(key)
        for source, destination in rows:
            if key == source or key.startswith(source + "/"):
                if source in applied:
                    raise ValueError("cycle")
                applied.add(source)
                result = destination + result[len(source):]
                if len(result) == 2 and result[1] == ":":
                    result += "/"
                break
        else:
            return Path(result)


def _rows(table):
    return sorted(table.items(), key=lambda row: len(row[0]), reverse=True)


def test_lookup_matches_the_full_scan_on_random_trees():
    generator = random.Random(7)
    parts = ["a", "b", "ab", "c", "data", "raw data", "x.toml"]
    for _ in range(200):
        table = {}
        for _ in range(generator.randint(1, 8)):
            depth = generator.randint(1, 4)
            source = "//nas/share/" + "/".join(generator.choice(parts) for _ in range(depth))
            table[source.casefold()] = "c:/moved/" + "/".join(generator.choice(parts) for _ in range(2))
        rows = _rows(table)
        for _ in range(20):
            depth = generator.randint(1, 6)
            path = "//nas/share/" + "/".join(generator.choice(parts) for _ in range(depth))
            try:
                expected = _brute_force(path, rows)
            except ValueError:
                with pytest.raises(ValueError):
                    _resolve_with(path, rows)
                continue
            assert _resolve_with(path, rows) == expected


def test_a_source_matches_only_at_a_folder_boundary():
    table = {"//nas/share/ab": "c:/moved"}
    assert _longest_source("//nas/share/abc/file", table) is None
    assert _longest_source("//nas/share/ab/file", table) == "//nas/share/ab"
    assert _longest_source("//nas/share/ab", table) == "//nas/share/ab"


def test_the_longest_source_wins_and_chains_resolve():
    table = {"//nas/share/a": "//nas/share/b", "//nas/share/a/deep": "c:/deep", "//nas/share/b": "c:/final"}
    rows = _rows(table)
    assert _resolve_with("//nas/share/a/deep/f.json", rows) == Path("c:/deep/f.json")
    assert _resolve_with("//nas/share/a/other/f.json", rows) == Path("c:/final/other/f.json")
