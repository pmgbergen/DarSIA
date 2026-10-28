"""Actual guided factory binding; no source/photo/proof reader is executed."""
from darsia.fluidflower.painting_guided_review_bindings import guided_gap_dialog
from darsia.fluidflower.painted_camera_jpg_display import read_comparison_preview


def test_guided_factory_wires_original_comparison_reader_and_preserves_caller_injection(qtbot):
    """A wrapped preview must not bypass the original comparison capability.

    Construct the real Qt dialog with no source, so no worker or scientific
    fixture is needed to inspect the exact reader handed to the base owner.
    Explicit caller readers remain the caller's own finite display adapter.
    """
    def unrequested_preview(*_args, **_options):
        raise AssertionError("Constructing an empty review must not read a photo")

    def caller_comparison(*_args, **_options):
        raise AssertionError("No comparison was requested by a human")

    for options, expected in (({}, read_comparison_preview),
                              ({"comparison_reader": caller_comparison}, caller_comparison)):
        widget = guided_gap_dialog(None, actor="Synthetic reviewer",
                                   preview_reader=unrequested_preview, **options)
        qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
        assert widget._comparison_reader is expected
        assert not widget.busy() and widget.worker is None
        assert widget.review is widget.packet is widget.approval is None
        assert widget._candidate is widget._report is None
