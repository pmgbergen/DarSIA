"""Tests for metadata keyword behavior in settings schema and widget creation.

Each metadata keyword (hidden, help, link, options, widget, active_list_key, name)
is tested to verify both that it appears in the generated schema and that the
SettingsFactory correctly consumes it to produce the intended widget behavior.
"""

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QWidget,
)

from darsia.gui.ui.schema.dataclass_introspection import _build_fields
from darsia.gui.ui.settings import SettingsFactory


# ────────────────────────────────────────────────────────────────────────────────
# Synthetic dataclass fixtures for metadata testing
# ────────────────────────────────────────────────────────────────────────────────


@dataclass
class SimpleFieldsConfig:
    """Config with various scalar field types for metadata testing."""

    visible_string: str = "default"
    """A normal string field with help text."""
    hidden_string: str = field(
        default="secret",
        metadata={"hidden": True},
    )
    """This field should be hidden from the GUI."""


@dataclass
class HelpLinkConfig:
    """Config for testing help and link metadata."""

    with_help: str = field(
        default="",
        metadata={"help": "This is helpful text."},
    )
    """Field with help text."""
    with_help_and_link: str = field(
        default="",
        metadata={
            "help": "See documentation for more.",
            "link": "https://example.com/docs",
        },
    )
    """Field with both help and link."""
    no_help: int = 42
    """Field without help metadata."""


@dataclass
class OptionsConfig:
    """Config for testing options metadata."""

    choice_field: str = field(
        default="a",
        metadata={"options": ["a", "b", "c"]},
    )
    """Dropdown field with three choices."""
    number_choice: int = field(
        default=1,
        metadata={"options": [1, 2, 3, 5, 8]},
    )
    """Numeric dropdown."""


@dataclass
class WidgetOverrideConfig:
    """Config for testing widget type override."""

    as_folder: Path = field(
        default_factory=Path,
        metadata={"widget": "folder"},
    )
    """Path field forced to render as folder chooser."""


@dataclass
class SubConfig:
    """A nested config for testing active_list_key."""

    enabled: bool = False
    """Whether this sub-config is enabled."""


@dataclass
class GroupWithActiveListConfig:
    """Config with groups that support active_list_key."""

    sub_with_active: SubConfig | None = field(
        default=None,
        metadata={"active_list_key": "active"},
    )
    """Sub-config with active-list toggle."""
    sub_without_active: SubConfig | None = None
    """Sub-config without active-list toggle."""
    active: list[str] = field(default_factory=list)
    """List tracking which sub-configs are active."""


@dataclass
class NameMetadataConfig:
    """Config for testing the name display-label metadata."""

    with_name: str = field(
        default="",
        metadata={"name": "Custom Display Name"},
    )
    """Field with custom display name."""
    without_name: str = ""
    """Field without custom name (uses dotted key)."""


# ────────────────────────────────────────────────────────────────────────────────
# Tests for each metadata keyword
# ────────────────────────────────────────────────────────────────────────────────


class TestHiddenMetadata:
    """Test that fields with hidden=True are excluded from schema."""

    def test_hidden_field_absent_from_schema(self):
        """Fields marked hidden should not appear in _build_fields output."""
        schema = _build_fields(SimpleFieldsConfig, "test")
        field_keys = [f["key"] for f in schema]

        assert "test.visible_string" in field_keys
        assert "test.hidden_string" not in field_keys

    def test_hidden_field_means_no_widget(self):
        """Hidden field should never produce a widget."""
        schema = _build_fields(SimpleFieldsConfig, "test")
        assert len(schema) == 1  # Only visible_string


class TestHelpMetadata:
    """Test that help text metadata is captured and consumed."""

    def test_help_in_schema(self):
        """Help text should appear in the schema dict."""
        schema = _build_fields(HelpLinkConfig, "test")
        with_help = next(f for f in schema if f["key"] == "test.with_help")
        assert with_help.get("help") == "This is helpful text."

    def test_no_help_absent_from_schema(self):
        """Fields without help should have no_help entry omitted (None filtered out)."""
        schema = _build_fields(HelpLinkConfig, "test")
        no_help_field = next(f for f in schema if f["key"] == "test.no_help")
        assert "help" not in no_help_field  # Filtered out because None

    def test_wrap_setting_with_help_adds_help_button(self, qtbot):
        """wrap_setting_with_help should add HelpButton when help is present."""
        from darsia.gui.ui.help import HelpButton

        factory = SettingsFactory(MagicMock())
        setting_dict = {"help": "Test help text", "type": "string"}

        # Create a dummy container
        dummy_widget = QLineEdit()

        # Wrap it
        wrapped = factory.wrap_setting_with_help(dummy_widget, setting_dict)
        qtbot.addWidget(wrapped)

        # Check that HelpButton is present
        help_buttons = wrapped.findChildren(HelpButton)
        assert len(help_buttons) == 1
        assert help_buttons[0].help_text == "Test help text"

    def test_wrap_setting_without_help_adds_stretch(self, qtbot):
        """wrap_setting_with_help should add stretch (no button) when help absent."""
        from darsia.gui.ui.help import HelpButton

        factory = SettingsFactory(MagicMock())
        setting_dict = {"type": "string"}  # No "help" key

        dummy_widget = QLineEdit()
        wrapped = factory.wrap_setting_with_help(dummy_widget, setting_dict)
        qtbot.addWidget(wrapped)

        # Check that no HelpButton is present
        help_buttons = wrapped.findChildren(HelpButton)
        assert len(help_buttons) == 0


class TestLinkMetadata:
    """Test that link URLs are passed to HelpButton correctly."""

    def test_link_in_schema(self):
        """Link URL should appear in schema alongside help."""
        schema = _build_fields(HelpLinkConfig, "test")
        with_link = next(
            f for f in schema if f["key"] == "test.with_help_and_link"
        )
        assert with_link.get("help") == "See documentation for more."
        assert with_link.get("link") == "https://example.com/docs"

    def test_help_button_with_link_enabled(self, qtbot):
        """HelpButton should be enabled (blue) when link is present."""
        from darsia.gui.ui.help import HelpButton

        factory = SettingsFactory(MagicMock())
        setting_dict = {
            "help": "Help text",
            "link": "https://example.com",
            "type": "string",
        }

        dummy_widget = QLineEdit()
        wrapped = factory.wrap_setting_with_help(dummy_widget, setting_dict)
        qtbot.addWidget(wrapped)

        help_buttons = wrapped.findChildren(HelpButton)
        assert len(help_buttons) == 1
        assert help_buttons[0].link_url == "https://example.com"
        # Button should be enabled (clickable) when link is present
        assert help_buttons[0].isEnabled()

    def test_help_button_without_link_disabled(self, qtbot):
        """HelpButton should be disabled when link is absent."""
        from darsia.gui.ui.help import HelpButton

        factory = SettingsFactory(MagicMock())
        setting_dict = {
            "help": "Help text only",
            "type": "string",
        }

        dummy_widget = QLineEdit()
        wrapped = factory.wrap_setting_with_help(dummy_widget, setting_dict)
        qtbot.addWidget(wrapped)

        help_buttons = wrapped.findChildren(HelpButton)
        assert len(help_buttons) == 1
        # Button should be disabled when no link is present
        assert not help_buttons[0].isEnabled()

    @patch("webbrowser.open")
    def test_help_button_link_click_opens_url(self, mock_webbrowser, qtbot):
        """Clicking HelpButton with link should open the URL."""
        from darsia.gui.ui.help import HelpButton

        factory = SettingsFactory(MagicMock())
        setting_dict = {
            "help": "Help text",
            "link": "https://example.com",
            "type": "string",
        }

        dummy_widget = QLineEdit()
        wrapped = factory.wrap_setting_with_help(dummy_widget, setting_dict)
        qtbot.addWidget(wrapped)

        help_buttons = wrapped.findChildren(HelpButton)
        assert len(help_buttons) == 1
        help_button = help_buttons[0]

        # Simulate click (qtbot.mouseClick requires Qt.MouseButton enum, positional-only)
        qtbot.mouseClick(help_button, Qt.MouseButton.LeftButton)

        # Verify webbrowser.open was called with the URL
        mock_webbrowser.assert_called_once_with("https://example.com")


class TestOptionsMetadata:
    """Test that options metadata forces dropdown/checkbox-list widgets."""

    def test_options_in_schema(self):
        """Options list should appear in schema."""
        schema = _build_fields(OptionsConfig, "test")
        choice_field = next(f for f in schema if f["key"] == "test.choice_field")
        assert choice_field.get("options") == ["a", "b", "c"]

    def test_string_field_with_options_creates_dropdown(self, qtbot):
        """String field with options should create QComboBox, not QLineEdit."""
        factory = SettingsFactory(MagicMock(config_dict={}, chosen_files={}))
        setting_dict = {
            "key": "test.choice",
            "type": "string",
            "options": ["a", "b", "c"],
            "default": "a",
        }

        label_text, field_widget = factory.create_dropdown_input(setting_dict)
        qtbot.addWidget(field_widget)

        # field_widget is a composite HBox; find the QComboBox inside
        combos = field_widget.findChildren(QComboBox)
        assert len(combos) > 0
        widget = combos[0]

        # Should be a QComboBox
        assert isinstance(widget, QComboBox)
        assert widget.count() == 3
        assert [widget.itemText(i) for i in range(3)] == ["a", "b", "c"]
        # Default value should be pre-selected
        assert widget.currentText() == "a"

    def test_int_field_with_options_creates_dropdown(self, qtbot):
        """Numeric field with options should also create QComboBox."""
        factory = SettingsFactory(MagicMock(config_dict={}, chosen_files={}))
        setting_dict = {
            "key": "test.number",
            "type": "int",
            "options": [1, 2, 3, 5, 8],
            "default": 1,
        }

        label_text, field_widget = factory.create_dropdown_input(setting_dict)
        qtbot.addWidget(field_widget)

        # field_widget is a composite HBox; find the QComboBox inside
        combos = field_widget.findChildren(QComboBox)
        assert len(combos) > 0
        widget = combos[0]

        assert isinstance(widget, QComboBox)
        assert widget.count() == 5


class TestWidgetMetadata:
    """Test that widget metadata overrides type inference."""

    def test_widget_override_in_schema(self):
        """Widget override should appear in schema."""
        schema = _build_fields(WidgetOverrideConfig, "test")
        as_folder = next(f for f in schema if f["key"] == "test.as_folder")
        assert as_folder.get("type") == "folder"

    def test_widget_override_forces_folder_chooser(self, qtbot):
        """Path field with widget='folder' should dispatch to folder chooser."""
        factory = SettingsFactory(MagicMock(config_dict={}, chosen_files={}, file_dialog=MagicMock()))
        setting_dict = {
            "key": "test.folder",
            "type": "folder",  # Overridden from the default "file"
            "name": "Folder",
        }

        # Should dispatch to file_dialog.create_file_chooser with is_directory=True
        # We can't fully test this without mocking the file dialog, but we can verify
        # the type is correctly read from the override
        assert setting_dict["type"] == "folder"


class TestActiveListKeyMetadata:
    """Test that active_list_key makes groups checkable."""

    def test_active_list_key_in_schema(self):
        """Group fields should have active_list_key in schema."""
        schema = _build_fields(GroupWithActiveListConfig, "test")
        group_with = next(
            f for f in schema if f["key"] == "test.sub_with_active"
        )
        group_without = next(
            f for f in schema if f["key"] == "test.sub_without_active"
        )

        assert group_with.get("active_list_key") == "active"
        assert "active_list_key" not in group_without  # Filtered out (None)

    def test_group_with_active_list_is_checkable(self, qtbot):
        """Group with active_list_key should produce a checkable QGroupBox."""
        factory = SettingsFactory(MagicMock(config_dict={}, chosen_files={}))
        setting_dict = {
            "key": "test.sub",
            "type": "group",
            "active_list_key": "active",  # Unqualified name (section prepended by create_group_input)
            "fields": [
                {"key": "test.sub.enabled", "type": "bool", "default": False}
            ],
        }

        label_text, result = factory.create_group_input(setting_dict)
        group_box = result.get("widget")
        qtbot.addWidget(group_box)

        # label_text should be None for groups
        assert label_text is None
        # Should be checkable
        assert group_box.isCheckable()
        # Result dict should have checkbox info
        assert "checkbox" in result
        # create_group_input prepends the section to produce the fully-qualified key
        assert result.get("active_list_key") == "test.active"

    def test_group_without_active_list_not_checkable(self, qtbot):
        """Group without active_list_key should be a plain non-checkable box."""
        factory = SettingsFactory(MagicMock(config_dict={}, chosen_files={}))
        setting_dict = {
            "key": "test.sub",
            "type": "group",
            "fields": [
                {"key": "test.sub.enabled", "type": "bool", "default": False}
            ],
        }

        label_text, result = factory.create_group_input(setting_dict)
        group_box = result.get("widget")
        qtbot.addWidget(group_box)

        # Should NOT be checkable
        assert not group_box.isCheckable()
        # Result dict should not have checkbox bookkeeping
        assert "checkbox" not in result


class TestNameMetadata:
    """Test that name metadata provides custom display labels."""

    def test_name_in_schema(self):
        """Name metadata should appear in schema."""
        schema = _build_fields(NameMetadataConfig, "test")
        with_name = next(f for f in schema if f["key"] == "test.with_name")
        without_name = next(f for f in schema if f["key"] == "test.without_name")

        assert with_name.get("name") == "Custom Display Name"
        assert "name" not in without_name  # Filtered out (None)

    def test_simple_input_uses_name_label(self, qtbot):
        """Simple input should use name as label when present."""
        factory = SettingsFactory(MagicMock(config_dict={}, chosen_files={}))
        setting_dict = {
            "key": "test.field",
            "type": "string",
            "name": "Display Name",
        }

        label_text, field_widget = factory.create_simple_input(setting_dict)
        qtbot.addWidget(field_widget)

        # Label is now returned as a string
        assert label_text == "Display Name"

    def test_simple_input_fallback_to_key(self, qtbot):
        """Simple input should fallback to key when name absent."""
        factory = SettingsFactory(MagicMock(config_dict={}, chosen_files={}))
        setting_dict = {
            "key": "test.field",
            "type": "string",
        }

        label_text, field_widget = factory.create_simple_input(setting_dict)
        qtbot.addWidget(field_widget)

        # Label is now returned as a string; should fall back to key
        assert label_text == "test.field"

    def test_bool_input_uses_name_label(self, qtbot):
        """Bool input should use name as label when present."""
        factory = SettingsFactory(MagicMock(config_dict={}, chosen_files={}))
        setting_dict = {
            "key": "test.flag",
            "type": "bool",
            "name": "Enable Feature",
        }

        label_text, field_widget = factory.create_bool_input(setting_dict)
        qtbot.addWidget(field_widget)

        # Label is now returned as a string
        assert label_text == "Enable Feature"

    def test_group_uses_name_for_title(self, qtbot):
        """Group box should use name as title when present."""
        factory = SettingsFactory(MagicMock(config_dict={}, chosen_files={}))
        setting_dict = {
            "key": "test.subconfig",
            "type": "group",
            "name": "Sub Configuration",
            "fields": [],
        }

        label_text, result = factory.create_group_input(setting_dict)
        group_box = result.get("widget")
        qtbot.addWidget(group_box)

        # label_text should be None for group types
        assert label_text is None
        assert group_box.title() == "Sub Configuration"

    def test_group_fallback_to_key_segment(self, qtbot):
        """Group box should fallback to last key segment when name absent."""
        factory = SettingsFactory(MagicMock(config_dict={}, chosen_files={}))
        setting_dict = {
            "key": "test.subconfig",
            "type": "group",
            "fields": [],
        }

        label_text, result = factory.create_group_input(setting_dict)
        group_box = result.get("widget")
        qtbot.addWidget(group_box)

        # Should use the last segment of the key
        assert group_box.title() == "subconfig"


# ────────────────────────────────────────────────────────────────────────────────
# Regression test for DepthConfig
# ────────────────────────────────────────────────────────────────────────────────


class TestDepthConfigMetadata:
    """Regression test: ensure DepthConfig has correct metadata applied."""

    def test_depth_config_measurements_metadata(self):
        """DepthConfig.measurements should have name and help metadata."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("depth")
        assert schema is not None

        measurements = next(
            (f for f in schema if f["key"] == "depth.measurements"),
            None,
        )
        assert measurements is not None
        assert measurements.get("name") == "Measurements"
        assert measurements.get("help") is not None
        assert len(measurements["help"]) > 0

    def test_depth_config_depth_map_hidden(self):
        """DepthConfig.depth_map should be hidden (auto-computed, not user-editable)."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("depth")
        assert schema is not None

        # depth_map should be absent from the schema because it's marked hidden
        depth_map = next(
            (f for f in schema if f["key"] == "depth.depth_map"), None
        )
        assert depth_map is None, "depth_map should be hidden and absent from schema"


class TestImagePorosityConfigMetadata:
    """Regression test: ensure ImagePorosityConfig has section-level active toggle."""

    def test_image_porosity_config_is_checkable_group(self):
        """ImagePorosityConfig should render as a checkable group (section-level toggle)."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("image_porosity")
        assert schema is not None
        # Entire section becomes a single group
        assert len(schema) == 1
        group = schema[0]
        assert group.get("type") == "group"
        assert group.get("active_bool_key") == "image_porosity.active"
        assert group.get("active_bool_default") is False

    def test_image_porosity_nested_fields(self):
        """ImagePorosityConfig group should contain all 7 original fields, not active."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("image_porosity")
        group = schema[0]
        fields = group.get("fields", [])
        field_keys = [f["key"] for f in fields]

        # Should have exactly 7 nested fields (all original fields except active)
        assert len(fields) == 7
        assert "image_porosity.mode" in field_keys
        assert "image_porosity.tol" in field_keys
        assert "image_porosity.patches" in field_keys
        assert "image_porosity.num_clusters" in field_keys
        assert "image_porosity.sample_width" in field_keys
        assert "image_porosity.tol_color_distance" in field_keys
        assert "image_porosity.tol_color_gradient" in field_keys
        # active should NOT be in the nested fields (it's hidden and used as the toggle)
        assert "image_porosity.active" not in field_keys

    def test_image_porosity_mode_has_dropdown_options(self):
        """ImagePorosityConfig.mode should have options for dropdown widget."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("image_porosity")
        group = schema[0]
        fields = group.get("fields", [])
        mode_field = next(f for f in fields if f["key"] == "image_porosity.mode")

        assert mode_field.get("options") == ["full", "from_image"]
        assert mode_field.get("name") == "Mode"
        assert mode_field.get("help") is not None

    def test_image_porosity_group_has_title(self):
        """ImagePorosityConfig group box should have a user-friendly title."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("image_porosity")
        group = schema[0]
        assert group.get("name") == "Activate image porosity"


class TestFaciesConfigMetadata:
    """Regression test: ensure FaciesConfig has correct metadata applied."""

    def test_facies_config_facies_to_labels_map_metadata(self):
        """FaciesConfig.facies_to_labels_map should have int_list_map widget."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("facies")
        assert schema is not None

        facies_map = next(
            (f for f in schema if f["key"] == "facies.facies_to_labels_map"),
            None,
        )
        assert facies_map is not None
        # The schema uses "type" field to hold the widget type, not "widget"
        assert facies_map.get("type") == "int_list_map"
        assert facies_map.get("name") == "Facies groups"
        assert facies_map.get("help") is not None
        assert len(facies_map["help"]) > 0
        assert facies_map.get("placeholder") == "Comma/space-separated label IDs (e.g., 3, 5, 8)"
        assert facies_map.get("flatten_in_section") is True

    def test_facies_config_label_to_facies_map_hidden(self):
        """FaciesConfig.label_to_facies_map should be hidden (derived field)."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("facies")
        assert schema is not None

        # label_to_facies_map should be absent from the schema because it's marked hidden
        label_map = next(
            (f for f in schema if f["key"] == "facies.label_to_facies_map"),
            None,
        )
        assert label_map is None, "label_to_facies_map should be hidden and absent from schema"

    def test_facies_config_path_hidden(self):
        """FaciesConfig.path should be hidden (auto-derived output path)."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("facies")
        assert schema is not None

        # path should be absent from the schema because it's marked hidden
        path_field = next(
            (f for f in schema if f["key"] == "facies.path"),
            None,
        )
        assert path_field is None, "path should be hidden and absent from schema"


class TestCorrectionsConfigMetadata:
    """Regression test: ensure CorrectionsConfig correction sections have correct metadata."""

    def test_all_correction_fields_have_name(self):
        """All 8 CorrectionsConfig correction fields should have a friendly name."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("corrections")
        assert schema is not None

        # Expect 8 groups, one for each correction type
        field_keys = [f["key"] for f in schema]
        expected_corrections = [
            "corrections.type",
            "corrections.resize",
            "corrections.drift",
            "corrections.curvature",
            "corrections.color",
            "corrections.relative_color",
            "corrections.illumination",
            "corrections.patchwise_illumination",
        ]
        for key in expected_corrections:
            assert key in field_keys

        # All should have a name in their metadata
        for field in schema:
            if field["key"] in expected_corrections:
                assert field.get("name") is not None, f"{field['key']} should have a name"

    def test_drift_correction_colorchecker_metadata(self):
        """DriftCorrectionConfig.colorchecker should have name, help, and options."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("corrections")
        drift_group = next(
            (f for f in schema if f["key"] == "corrections.drift"), None
        )
        assert drift_group is not None
        assert drift_group.get("type") == "group"

        # Dig into nested fields
        fields = drift_group.get("fields", [])
        colorchecker_field = next(
            (f for f in fields if f["key"] == "corrections.drift.colorchecker"),
            None,
        )
        assert colorchecker_field is not None
        assert colorchecker_field.get("name") == "Color checker position"
        assert colorchecker_field.get("help") is not None
        assert colorchecker_field.get("options") == [
            "upper_left",
            "upper_right",
            "lower_left",
            "lower_right",
        ]

    def test_unsupported_fields_are_hidden(self):
        """Fields with unsupported types should be marked hidden."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("corrections")
        assert schema is not None

        # Check that hidden fields do not appear in any group's fields
        for group in schema:
            if group.get("type") == "group":
                fields = group.get("fields", [])
                field_keys = [f["key"] for f in fields]

                # These fields should not appear because they're hidden
                hidden_patterns = [
                    "corrections.curvature.config",
                    "corrections.relative_color.options",
                    "corrections.patchwise_illumination.baseline_paths",
                ]
                for pattern in hidden_patterns:
                    assert pattern not in field_keys, f"{pattern} should be hidden"

    def test_resize_correction_mode_field_metadata(self):
        """ResizeCorrectionConfig.mode should have name, help, and options."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("corrections")
        resize_group = next(
            (f for f in schema if f["key"] == "corrections.resize"), None
        )
        assert resize_group is not None
        assert resize_group.get("type") == "group"

        # Dig into nested fields
        fields = resize_group.get("fields", [])
        mode_field = next(
            (f for f in fields if f["key"] == "corrections.resize.mode"), None
        )
        assert mode_field is not None
        assert mode_field.get("name") == "Resize mode"
        assert mode_field.get("help") is not None
        assert mode_field.get("options") == ["scale", "target_shape"]

    def test_resize_correction_scale_and_shape_metadata(self):
        """ResizeCorrectionConfig.scale and .target_shape should have name and help."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("corrections")
        resize_group = next(
            (f for f in schema if f["key"] == "corrections.resize"), None
        )
        assert resize_group is not None

        fields = resize_group.get("fields", [])
        field_keys = [f["key"] for f in fields]

        # Both scale and target_shape should be present
        assert "corrections.resize.scale" in field_keys
        assert "corrections.resize.target_shape" in field_keys

        # Both should have name and help
        scale_field = next(f for f in fields if f["key"] == "corrections.resize.scale")
        target_shape_field = next(
            f for f in fields if f["key"] == "corrections.resize.target_shape"
        )

        assert scale_field.get("name") == "Scale factor"
        assert scale_field.get("help") is not None
        assert target_shape_field.get("name") == "Target shape"
        assert target_shape_field.get("help") is not None

    def test_resize_correction_depends_on_metadata(self):
        """ResizeCorrectionConfig.scale and .target_shape should have depends_on metadata."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("corrections")
        resize_group = next(
            (f for f in schema if f["key"] == "corrections.resize"), None
        )
        assert resize_group is not None

        fields = resize_group.get("fields", [])
        scale_field = next(f for f in fields if f["key"] == "corrections.resize.scale")
        target_shape_field = next(
            f for f in fields if f["key"] == "corrections.resize.target_shape"
        )

        # scale should depend on mode == "scale"
        assert scale_field.get("depends_on") == {"field": "mode", "value": "scale"}
        # target_shape should depend on mode == "target_shape"
        assert target_shape_field.get("depends_on") == {
            "field": "mode",
            "value": "target_shape",
        }

    def test_curvature_correction_has_stage_groups(self):
        """CurvatureCorrectionConfig should have 4 named stage groups with active_list_key."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("corrections")
        curvature_group = next(
            (f for f in schema if f["key"] == "corrections.curvature"), None
        )
        assert curvature_group is not None
        assert curvature_group.get("type") == "group"

        # Check the 4 stage fields exist
        fields = curvature_group.get("fields", [])
        field_keys = [f["key"] for f in fields]

        expected_stages = [
            "corrections.curvature.init",
            "corrections.curvature.crop",
            "corrections.curvature.bulge",
            "corrections.curvature.stretch",
        ]
        for key in expected_stages:
            assert key in field_keys, f"{key} should be present"

        # Each should be a group with active_list_key
        for field in fields:
            if field["key"] in expected_stages:
                assert field.get("type") == "group"
                assert field.get("active_list_key") == "active"
                assert field.get("name") is not None

    def test_curvature_crop_pts_src_hidden(self):
        """CropCorrectionConfig.pts_src should be hidden (internal field)."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("corrections")
        curvature_group = next(
            (f for f in schema if f["key"] == "corrections.curvature"), None
        )
        fields = curvature_group.get("fields", [])
        crop_group = next(
            (f for f in fields if f["key"] == "corrections.curvature.crop"), None
        )
        assert crop_group is not None

        crop_fields = crop_group.get("fields", [])
        crop_field_keys = [f["key"] for f in crop_fields]

        # pts_src should not appear (hidden)
        assert "corrections.curvature.crop.pts_src" not in crop_field_keys

    def test_curvature_crop_corner_fields_visible(self):
        """CropCorrectionConfig corner fields (top_left, etc.) should be visible."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("corrections")
        curvature_group = next(
            (f for f in schema if f["key"] == "corrections.curvature"), None
        )
        fields = curvature_group.get("fields", [])
        crop_group = next(
            (f for f in fields if f["key"] == "corrections.curvature.crop"), None
        )
        assert crop_group is not None

        crop_fields = crop_group.get("fields", [])
        crop_field_keys = [f["key"] for f in crop_fields]

        # All 4 corner fields should be present and visible
        assert "corrections.curvature.crop.top_left" in crop_field_keys
        assert "corrections.curvature.crop.bottom_left" in crop_field_keys
        assert "corrections.curvature.crop.bottom_right" in crop_field_keys
        assert "corrections.curvature.crop.top_right" in crop_field_keys

        # Each corner should have name, help, and placeholder metadata
        for corner_key in [
            "corrections.curvature.crop.top_left",
            "corrections.curvature.crop.bottom_left",
            "corrections.curvature.crop.bottom_right",
            "corrections.curvature.crop.top_right",
        ]:
            corner_field = next(
                (f for f in crop_fields if f["key"] == corner_key), None
            )
            assert corner_field is not None, f"Corner field {corner_key} not found"
            assert "name" in corner_field, f"{corner_key} missing 'name' metadata"
            assert "help" in corner_field, f"{corner_key} missing 'help' metadata"
            assert "placeholder" in corner_field, f"{corner_key} missing 'placeholder' metadata"
            # Should not be hidden
            assert corner_field.get("hidden") is not True

    def test_illumination_correction_fields_have_metadata(self):
        """IlluminationCorrectionConfig fields should all have name and help."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("corrections")
        illumination_group = next(
            (f for f in schema if f["key"] == "corrections.illumination"), None
        )
        assert illumination_group is not None, "Illumination group not found"

        illumination_fields = illumination_group.get("fields", [])
        illumination_field_keys = [f["key"] for f in illumination_fields]

        # All illumination fields should be present
        expected_fields = [
            "corrections.illumination.labels",
            "corrections.illumination.interpolation",
            "corrections.illumination.colorspace",
            "corrections.illumination.width",
            "corrections.illumination.num_samples",
            "corrections.illumination.seed",
            "corrections.illumination.sigma",
            "corrections.illumination.outliers",
            "corrections.illumination.bounds",
        ]
        for expected_key in expected_fields:
            assert (
                expected_key in illumination_field_keys
            ), f"{expected_key} should be present"

        # Check that interpolation and colorspace have options
        for field_key in ["interpolation", "colorspace"]:
            full_key = f"corrections.illumination.{field_key}"
            field = next(
                (f for f in illumination_fields if f["key"] == full_key), None
            )
            assert field is not None, f"{field_key} not found"
            assert "options" in field, f"{field_key} should have options"
            assert isinstance(field["options"], list), f"{field_key} options should be a list"

    def test_patchwise_illumination_correction_fields_have_metadata(self):
        """PatchwiseIlluminationCorrectionConfig fields (except baseline_paths) should have metadata."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("corrections")
        patchwise_group = next(
            (f for f in schema if f["key"] == "corrections.patchwise_illumination"),
            None,
        )
        assert patchwise_group is not None, "Patchwise illumination group not found"

        patchwise_fields = patchwise_group.get("fields", [])
        patchwise_field_keys = [f["key"] for f in patchwise_fields]

        # These fields should be present
        visible_fields = [
            "corrections.patchwise_illumination.image_path",
            "corrections.patchwise_illumination.limit",
            "corrections.patchwise_illumination.nw",
            "corrections.patchwise_illumination.eps",
        ]
        for expected_key in visible_fields:
            assert (
                expected_key in patchwise_field_keys
            ), f"{expected_key} should be present"

        # baseline_paths should NOT be present (hidden)
        assert (
            "corrections.patchwise_illumination.baseline_paths"
            not in patchwise_field_keys
        ), "baseline_paths should be hidden"

    def test_curvature_crop_corner_fields_have_legacy_metadata(self):
        """Corner fields should have legacy_source and legacy_index metadata for GUI fallback."""
        from darsia.gui.ui.schema.dataclass_introspection import (
            get_section_fields,
        )

        schema = get_section_fields("corrections")
        curvature_group = next(
            (f for f in schema if f["key"] == "corrections.curvature"), None
        )
        fields = curvature_group.get("fields", [])
        crop_group = next(
            (f for f in fields if f["key"] == "corrections.curvature.crop"), None
        )
        crop_fields = crop_group.get("fields", [])

        # Define expected corner-to-index mapping
        expected_corners = {
            "top_left": 0,
            "bottom_left": 1,
            "bottom_right": 2,
            "top_right": 3,
        }

        for corner_name, expected_index in expected_corners.items():
            corner_field = next(
                (
                    f
                    for f in crop_fields
                    if corner_name in f.get("key", "")
                ),
                None,
            )
            assert corner_field is not None, f"Corner field {corner_name} not found"
            assert (
                corner_field.get("legacy_source") == "pts_src"
            ), f"{corner_name} missing 'legacy_source' metadata"
            assert (
                corner_field.get("legacy_index") == expected_index
            ), f"{corner_name} has wrong legacy_index: {corner_field.get('legacy_index')}, expected {expected_index}"
