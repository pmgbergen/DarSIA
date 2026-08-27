"""Test that the main() entry point for darsia CLI launcher is callable and correct.

This test verifies that the darsia.gui.__main__.main() function is properly defined
as an entry point. Qt application tests are covered by existing test_main_window_startup.py
which uses qtbot fixtures.
"""

import inspect


def test_main_function_exists():
    """Test that the main() function exists and is callable."""
    from darsia.gui.__main__ import main

    assert callable(main), "main() function should be callable"


def test_main_function_signature():
    """Test that main() has the correct signature (no required args, returns None)."""
    from darsia.gui.__main__ import main

    sig = inspect.signature(main)
    assert len(sig.parameters) == 0, "main() should not require any arguments"
    assert sig.return_annotation in (
        None,
        "None",
        type(None),
    ), "main() should return None"


def test_main_function_has_docstring():
    """Test that main() has a docstring."""
    from darsia.gui.__main__ import main

    assert main.__doc__ is not None, "main() should have a docstring"
    assert (
        "GUI" in main.__doc__ or "application" in main.__doc__
    ), "Docstring should describe the GUI"


def test_main_is_importable():
    """Test that darsia.gui.__main__:main is a valid entry point reference."""
    from darsia.gui import __main__

    assert hasattr(
        __main__, "main"
    ), "darsia.gui.__main__ should have a 'main' attribute"
    assert callable(getattr(__main__, "main")), "main should be callable"
