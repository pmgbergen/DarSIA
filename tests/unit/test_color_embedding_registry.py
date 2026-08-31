"""Unit tests for ColorEmbeddingRegistry (array-of-tables [[color_path/range/channel]] format)."""

import textwrap
from pathlib import Path

import pytest

from darsia.presets.workflows.config.color_embedding_registry import (
    ColorEmbeddingRegistry,
)
from darsia.presets.workflows.config.data_registry import DataRegistry
from darsia.presets.workflows.config.roi import RoiConfig
from darsia.presets.workflows.config.roi_registry import RoiRegistry
from darsia.signals.color import (
    ColorChannelEmbedding,
    ColorPathEmbedding,
    ColorRangeEmbedding,
)


def _write_toml(tmp_path: Path, content: str, filename: str = "config.toml") -> Path:
    """Write TOML content to a temp file and return its path."""
    p = tmp_path / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content))
    return p


def _make_registry_with_roi(name: str = "test_roi") -> RoiRegistry:
    """Create a RoiRegistry with a single ROI for testing."""
    reg = RoiRegistry()
    roi = RoiConfig()
    roi.load({"name": name, "corner_1": [0.1, 0.2], "corner_2": [0.8, 0.9]})
    reg.register(name, roi)
    return reg


def _make_data_registry(tmp_path: Path) -> DataRegistry:
    """Create a DataRegistry with dummy baseline and data path entries."""
    dummy = tmp_path / "dummy.jpg"
    dummy.touch()
    sec = {
        "data_path": [
            {"name": "baseline_imgs", "paths": ["dummy.jpg"]},
            {"name": "cal_imgs", "paths": ["dummy.jpg"]},
        ]
    }
    return DataRegistry().load(sec, data_folder=tmp_path)


class TestColorEmbeddingRegistryStructuralValidation:
    """Test structural validation of the array-of-tables format."""

    def test_missing_color_sections_succeeds_with_empty_registry(self, tmp_path):
        """If no [color_path/range/channel] sections exist, registry loads with zero entries."""
        toml_path = _write_toml(tmp_path, "")
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=None,
            results=None,
        )
        assert len(registry.keys()) == 0

    def test_color_path_as_nested_dict_raises_error(self, tmp_path):
        """[color_path] as nested tables (not array-of-tables) raises ValueError."""
        toml_path = _write_toml(
            tmp_path,
            """
            [color_path.test_entry]
            """,
        )
        with pytest.raises(
            ValueError,
            match="must be an array-of-tables format.*use \\[\\[color_path\\]\\]",
        ):
            ColorEmbeddingRegistry().load(
                path=toml_path,
                data=None,
                results=None,
            )

    def test_missing_name_field_in_path_raises_error(self, tmp_path):
        """[[color_path]] entry without 'name' raises ValueError."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            mode = "relative"
            """,
        )
        with pytest.raises(ValueError, match="missing required 'name'"):
            ColorEmbeddingRegistry().load(
                path=toml_path,
                data=None,
                results=None,
            )

    def test_duplicate_name_raises_error(self, tmp_path):
        """Duplicate color embedding names raise ValueError."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            name = "duplicate"
            mode = "relative"
            basis = "labels"
            data = ["cal_imgs"]
            baseline = "baseline_imgs"

            [[color_range]]
            name = "duplicate"
            color_space = "RGB"
            range = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
            """,
        )
        with pytest.raises(ValueError, match="duplicated.*globally unique"):
            ColorEmbeddingRegistry().load(
                path=toml_path,
                data=tmp_path,
                results=tmp_path,
            )


class TestColorEmbeddingRegistryEagerValidation:
    """Test that field validation happens eagerly at .load() time."""

    def test_invalid_calibration_mode_raises_at_load_time(self, tmp_path):
        """Bad calibration_mode in a path entry raises at .load(), not .resolve()."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            name = "bad_mode"
            calibration_mode = "invalid_mode"
            data = ["cal_imgs"]
            baseline = "baseline_imgs"
            """,
        )
        with pytest.raises(ValueError, match="calibration_mode.*invalid_mode"):
            ColorEmbeddingRegistry().load(
                path=toml_path,
                data=tmp_path,
                results=tmp_path,
            )


    def test_missing_required_color_space_raises_at_load_time(self, tmp_path):
        """Missing 'color_space' in a range entry raises at .load()."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_range]]
            name = "no_colorspace"
            range = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
            """,
        )
        with pytest.raises(ValueError, match="color_space.*required"):
            ColorEmbeddingRegistry().load(
                path=toml_path,
                data=None,
                results=None,
            )


class TestColorEmbeddingRegistryPathType:
    """Test successful loading and resolution of path entries."""

    def test_minimal_path_entry_loads(self, tmp_path):
        """A minimal [[color_path]] loads successfully."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            name = "color_path"
            data = ["cal_imgs"]
            baseline = "baseline_imgs"
            """,
        )
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
        )
        assert "color_path" in registry
        embedding = registry.resolve("color_path")
        assert isinstance(embedding, ColorPathEmbedding)
        assert embedding.embedding_id == "color_path"
        assert embedding.mode.value == "relative"  # default
        assert embedding.basis.value == "labels"  # default


class TestColorEmbeddingRegistryRangeType:
    """Test successful loading and resolution of range entries."""

    def test_range_entry_loads(self, tmp_path):
        """A [[color_range]] loads successfully."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_range]]
            name = "green_range"
            color_space = "RGB"
            range = [[0.0, 1.0], [0.2, 0.8], [0.0, 1.0]]
            """,
        )
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=None,
            results=None,
        )
        embedding = registry.resolve("green_range")
        assert isinstance(embedding, ColorRangeEmbedding)
        assert embedding.embedding_id == "green_range"
        assert embedding.color_space == "RGB"
        assert len(embedding.ranges) == 3
        assert embedding.ranges[1] == (0.2, 0.8)


class TestColorEmbeddingRegistryChannelType:
    """Test successful loading and resolution of channel entries."""

    def test_channel_entry_loads(self, tmp_path):
        """A [[color_channel]] loads successfully."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_channel]]
            name = "red_channel"
            color_space = "RGB"
            channel = "r"
            """,
        )
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=None,
            results=None,
        )
        embedding = registry.resolve("red_channel")
        assert isinstance(embedding, ColorChannelEmbedding)
        assert embedding.embedding_id == "red_channel"
        assert embedding.color_space == "RGB"
        assert embedding.channel == "r"

    def test_channel_with_mask_loads(self, tmp_path):
        """Channel entry with an inline mask sub-table loads successfully."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_channel]]
            name = "masked_channel"
            color_space = "RGB"
            channel = "r"
            mask = { color_space = "HSV", range = [[0.2, 0.4], [0.5, 1.0], [0.0, 1.0]] }
            """,
        )
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=None,
            results=None,
        )
        embedding = registry.resolve("masked_channel")
        assert embedding.mask_embedding is not None
        assert isinstance(embedding.mask_embedding, ColorRangeEmbedding)
        assert embedding.mask_embedding.color_space == "HSV"


class TestColorEmbeddingRegistryMethods:
    """Test registry query methods: resolve_all, keys, __contains__."""

    def test_keys_returns_all_names(self, tmp_path):
        """keys() returns a list of all registered embedding names."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            name = "path_emb"
            data = ["cal_imgs"]
            baseline = "baseline_imgs"

            [[color_range]]
            name = "range_emb"
            color_space = "RGB"
            range = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

            [[color_channel]]
            name = "channel_emb"
            color_space = "RGB"
            channel = "r"
            """,
        )
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
        )
        keys = registry.keys()
        assert len(keys) == 3
        assert set(keys) == {"path_emb", "range_emb", "channel_emb"}

    def test_contains_operator(self, tmp_path):
        """The 'in' operator works on registry (uses __contains__)."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            name = "exists"
            data = ["cal_imgs"]
            baseline = "baseline_imgs"
            """,
        )
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
        )
        assert "exists" in registry
        assert "does_not_exist" not in registry

    def test_resolve_all_returns_dict(self, tmp_path):
        """resolve_all() returns a dict of all registered embeddings."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            name = "emb1"
            data = ["cal_imgs"]
            baseline = "baseline_imgs"

            [[color_range]]
            name = "emb2"
            color_space = "RGB"
            range = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
            """,
        )
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
        )
        all_embeddings = registry.resolve_all()
        assert len(all_embeddings) == 2
        assert "emb1" in all_embeddings
        assert "emb2" in all_embeddings
        assert isinstance(all_embeddings["emb1"], ColorPathEmbedding)
        assert isinstance(all_embeddings["emb2"], ColorRangeEmbedding)


class TestColorEmbeddingRegistryResolve:
    """Test the resolve() method."""

    def test_resolve_by_name_returns_embedding(self, tmp_path):
        """resolve(name) returns the matching ColorEmbedding."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            name = "test"
            data = ["cal_imgs"]
            baseline = "baseline_imgs"
            """,
        )
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
        )
        embedding = registry.resolve("test")
        assert isinstance(embedding, ColorPathEmbedding)

    def test_resolve_unknown_name_raises_keyerror(self, tmp_path):
        """resolve(unknown_name) raises KeyError with available keys."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            name = "exists"
            data = ["cal_imgs"]
            baseline = "baseline_imgs"
            """,
        )
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
        )
        with pytest.raises(KeyError, match="not found.*exists"):
            registry.resolve("unknown")

    def test_resolve_embedding_object_registered(self, tmp_path):
        """resolve(embedding_object) verifies it's registered and returns it."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            name = "test"
            data = ["cal_imgs"]
            baseline = "baseline_imgs"
            """,
        )
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
        )
        embedding = registry.resolve("test")
        # Resolve it again via object reference
        result = registry.resolve(embedding)
        assert result is embedding

    def test_resolve_unregistered_embedding_object_raises_keyerror(self, tmp_path):
        """resolve(unregistered_embedding_object) raises KeyError."""
        toml_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            name = "exists"
            data = ["cal_imgs"]
            baseline = "baseline_imgs"
            """,
        )
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
        )
        # Create an unregistered embedding
        unregistered = ColorPathEmbedding(
            embedding_id="not_registered",
            mode="relative",
            basis="labels",
            root=tmp_path / "color" / "color_path" / "dummy",
        )
        with pytest.raises(KeyError, match="not found in registry"):
            registry.resolve(unregistered)


class TestColorEmbeddingRegistryMultipleFiles:
    """Test loading from multiple TOML files."""

    def test_load_multiple_files_merges_entries(self, tmp_path):
        """Loading from multiple files merges all color entries."""
        file1_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            name = "from_file1"
            data = ["cal_imgs"]
            baseline = "baseline_imgs"
            """,
            "file1.toml",
        )
        file2_path = _write_toml(
            tmp_path,
            """
            [[color_range]]
            name = "from_file2"
            color_space = "RGB"
            range = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
            """,
            "file2.toml",
        )
        registry = ColorEmbeddingRegistry().load(
            path=[file1_path, file2_path],
            data=tmp_path,
            results=tmp_path,
        )
        assert len(registry.keys()) == 2
        assert "from_file1" in registry
        assert "from_file2" in registry

    def test_duplicate_names_across_files_raise_error(self, tmp_path):
        """Same embedding name in different files raises ValueError."""
        file1_path = _write_toml(
            tmp_path,
            """
            [[color_path]]
            name = "duplicate"
            data = ["cal_imgs"]
            baseline = "baseline_imgs"
            """,
            "file1.toml",
        )
        file2_path = _write_toml(
            tmp_path,
            """
            [[color_range]]
            name = "duplicate"
            color_space = "RGB"
            range = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
            """,
            "file2.toml",
        )
        with pytest.raises(ValueError, match="duplicated.*globally unique"):
            ColorEmbeddingRegistry().load(
                path=[file1_path, file2_path],
                data=tmp_path,
                results=tmp_path,
            )
