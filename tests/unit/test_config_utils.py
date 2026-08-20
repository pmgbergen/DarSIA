"""Unit tests for config utilities (_get_key, etc.)."""

import pytest

from darsia.presets.workflows.config.utils import _get_key


class TestGetKeyBlankStringHandling:
    """Test _get_key's handling of blank strings for int/float types."""

    def test_blank_string_int_optional_returns_default(self):
        """Blank string with type_=int and required=False returns default."""
        section = {"value": ""}
        result = _get_key(section, "value", default=None, required=False, type_=int)
        assert result is None

    def test_blank_string_int_optional_with_explicit_default(self):
        """Blank string with type_=int and required=False returns explicit default."""
        section = {"value": ""}
        result = _get_key(section, "value", default=42, required=False, type_=int)
        assert result == 42

    def test_blank_string_float_optional_returns_default(self):
        """Blank string with type_=float and required=False returns default."""
        section = {"value": ""}
        result = _get_key(section, "value", default=None, required=False, type_=float)
        assert result is None

    def test_blank_string_float_optional_with_explicit_default(self):
        """Blank string with type_=float and required=False returns explicit default."""
        section = {"value": ""}
        result = _get_key(section, "value", default=3.14, required=False, type_=float)
        assert result == 3.14

    def test_blank_string_int_required_raises_valueerror(self):
        """Blank string with type_=int and required=True raises ValueError."""
        section = {"value": ""}
        with pytest.raises(ValueError, match="Key 'value'.*blank.*expected a int"):
            _get_key(section, "value", required=True, type_=int)

    def test_blank_string_float_required_raises_valueerror(self):
        """Blank string with type_=float and required=True raises ValueError."""
        section = {"value": ""}
        with pytest.raises(ValueError, match="Key 'value'.*blank.*expected a float"):
            _get_key(section, "value", required=True, type_=float)

    def test_whitespace_only_string_int_optional_returns_default(self):
        """Whitespace-only string (e.g., "  ") with type_=int and required=False returns default."""
        section = {"value": "   "}
        result = _get_key(section, "value", default=None, required=False, type_=int)
        assert result is None

    def test_whitespace_only_string_float_optional_returns_default(self):
        """Whitespace-only string (e.g., "  ") with type_=float and required=False returns default."""
        section = {"value": "   "}
        result = _get_key(section, "value", default=None, required=False, type_=float)
        assert result is None

    def test_valid_int_string_still_casts(self):
        """Valid int string like '42' still casts to int(42)."""
        section = {"value": "42"}
        result = _get_key(section, "value", required=True, type_=int)
        assert result == 42
        assert isinstance(result, int)

    def test_valid_float_string_still_casts(self):
        """Valid float string like '3.5' still casts to float(3.5)."""
        section = {"value": "3.5"}
        result = _get_key(section, "value", required=True, type_=float)
        assert result == 3.5
        assert isinstance(result, float)

    def test_negative_int_string_still_casts(self):
        """Negative int string like '-123' still casts correctly."""
        section = {"value": "-123"}
        result = _get_key(section, "value", required=True, type_=int)
        assert result == -123

    def test_negative_float_string_still_casts(self):
        """Negative float string like '-3.14' still casts correctly."""
        section = {"value": "-3.14"}
        result = _get_key(section, "value", required=True, type_=float)
        assert result == -3.14


class TestGetKeyExistingBehavior:
    """Regression tests: confirm existing key-absent and non-int/float behavior unchanged."""

    def test_key_absent_required_raises_keyerror(self):
        """Key absent with required=True raises KeyError (no change)."""
        section = {}
        with pytest.raises(KeyError, match="Missing key 'value'"):
            _get_key(section, "value", required=True)

    def test_key_absent_optional_returns_default(self):
        """Key absent with required=False returns default (no change)."""
        section = {}
        result = _get_key(section, "value", default=None, required=False)
        assert result is None

    def test_key_absent_optional_with_explicit_default(self):
        """Key absent with required=False returns explicit default (no change)."""
        section = {}
        result = _get_key(section, "value", default="fallback", required=False)
        assert result == "fallback"

    def test_string_without_type_cast_unchanged(self):
        """String value without type_ conversion is returned as-is (no change)."""
        section = {"value": "hello"}
        result = _get_key(section, "value", type_=None)
        assert result == "hello"
        assert isinstance(result, str)

    def test_bool_type_unchanged(self):
        """bool() values are handled as before (bool("") = False, no crash)."""
        section = {"value": ""}
        result = _get_key(section, "value", default=None, required=False, type_=bool)
        # bool("") evaluates to False, and this is kept as-is (no special blank handling for bool)
        assert result is False

    def test_list_type_unchanged(self):
        """list() values are handled as before (list("") = [], no crash)."""
        section = {"value": ""}
        result = _get_key(section, "value", default=None, required=False, type_=list)
        # list("") evaluates to [], and this is kept as-is
        assert result == []


class TestGetKeyRegressionLabeling:
    """Regression test: the exact scenario that triggered the bug."""

    def test_labeling_water_label_blank_optional_int(self):
        """water_label: "" with type_=int and required=False returns default None."""
        section = {"water_label": ""}
        result = _get_key(
            section, "water_label", default=None, required=False, type_=int
        )
        assert result is None

    def test_labeling_colorchecker_label_blank_optional_int(self):
        """colorchecker_label: "" with type_=int and required=False returns default None."""
        section = {"colorchecker_label": ""}
        result = _get_key(
            section, "colorchecker_label", default=None, required=False, type_=int
        )
        assert result is None
