"""Real GeometryImageView gestures; the draft spy isolates only the UI seam.

The region backend separately owns scientific admission and rasterization.
No source guard or original painting implementation is replaced here.
"""
from copy import deepcopy
from types import MappingProxyType

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
import pytest

from darsia.fluidflower.painting_deviation_step_ui import DeviationPaintingStep
from darsia.fluidflower.pages.geometry_facies.page import GeometryImageView
from darsia.fluidflower.painting_gap_review_ui import _project


CHOICES = (("air", "Air"), ("free_water", "Free water"), ("sand", "Sand"),
           ("rig_or_frame", "Rig or frame"), ("other", "Other"))


class DraftSpy:
    """Records native strokes, never produces a proof or exclusion decision."""
    def __init__(self, mask):
        self.editable_mask = mask.copy()
        self.class_masks = {key: np.zeros(mask.shape, bool) for key, _ in CHOICES}
        self.actions, self.history, self.future = [], [], []

    @property
    def can_undo(self):
        return bool(self.history)

    @property
    def can_redo(self):
        return bool(self.future)

    def paint(self, *, kind, points_xy_native, class_name, radius_px):
        column, row = np.rint(points_xy_native[0]).astype(int)
        if not (0 <= row < self.editable_mask.shape[0] and 0 <= column < self.editable_mask.shape[1]
                and self.editable_mask[row, column]):
            return False
        self.actions.append(deepcopy(dict(kind=kind, points_xy_native=points_xy_native,
                                         class_name=class_name, radius_px=radius_px)))
        self.history.append(deepcopy(self.class_masks))
        self.future = []
        for mask in self.class_masks.values():
            mask[row, column] = False
        self.class_masks[class_name][row, column] = True
        return True

    def undo(self):
        self.future.append(deepcopy(self.class_masks))
        self.class_masks = self.history.pop()

    def redo(self):
        self.history.append(deepcopy(self.class_masks))
        self.class_masks = self.future.pop()

    def snapshot(self):
        return dict(class_masks={key: mask.copy() for key, mask in self.class_masks.items()},
                    actions=deepcopy(self.actions))


def _packet():
    mask = np.zeros((96, 128), bool)
    mask[36:65, 48:97] = True
    transform = [[1., 0., 6.], [0., 1., 4.], [0., 0., 1.]]
    return dict(run_id="ac18", component_id="c"*64,
        image_rgb=np.full((112, 152, 3), 70, np.uint8), image_encoding="rgb_uint8",
        native_to_photo_h=transform, gap_mask=mask,
        display_grid=dict(format="synthetic-bound-native-grid", shape_hw=[96, 128],
                          native_to_photo_h=transform, model={"synthetic": "one exact model"}),
        image_role="current", image_choices=[dict(image_id="clearwater", available=True)])


@pytest.fixture
def step(qtbot):
    widget = DeviationPaintingStep(class_choices=CHOICES, sand_class="sand")
    qtbot.addWidget(widget)
    widget.resize(1000, 780)
    widget.show()
    packet = _packet()
    draft = DraftSpy(packet["gap_mask"])
    widget.set_context(packet, draft)
    qtbot.waitUntil(lambda: widget._pending_gap_fit is None)
    return widget, packet, draft


def _point(canvas, x, y):
    return canvas.mapFromScene(QPointF(x, y))


def test_readonly_typed_class_mapping_is_displayed_and_forwarded_without_mutation(qtbot):
    packet = _packet()
    draft = DraftSpy(packet["gap_mask"])
    draft.class_masks["air"][42, 60] = True
    readonly = {key: mask.copy() for key, mask in draft.class_masks.items()}
    for mask in readonly.values():
        mask.setflags(write=False)
    draft.class_masks = MappingProxyType(readonly)
    widget = DeviationPaintingStep(class_choices=CHOICES, sand_class="sand")
    qtbot.addWidget(widget)
    widget.set_context(packet, draft)
    widget.set_brightness(140)
    requests = []
    widget.requested.connect(requests.append)
    widget.continue_button.click()
    assert len(requests) == 1 and requests[0]["class_masks"]["air"][42, 60]
    assert set(draft.class_masks) == {key for key, _ in CHOICES}
    assert draft.class_masks["air"][42, 60] and draft.actions == []
    assert all(not mask.flags.writeable for mask in draft.class_masks.values())


def _choose(widget, class_name, mode):
    widget.classification.setCurrentIndex(widget.classification.findData(class_name))
    widget.mode.setCurrentIndex(widget.mode.findData(mode))


def _brush(widget, x, y):
    position = _point(widget.canvas, x, y)
    scene = widget.canvas.mapToScene(position)
    QTest.mousePress(widget.canvas.viewport(), Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseRelease(widget.canvas.viewport(), Qt.MouseButton.LeftButton, pos=position)
    return scene


def test_opens_original_tool_zoomed_to_deviation_without_a_class_or_decision(step):
    widget, packet, draft = step
    assert type(widget.canvas) is GeometryImageView
    assert widget.classification.currentData() is None and widget.canvas.edit_mode == "pan"
    assert draft.actions == [] and not widget.continue_button.isEnabled()
    view = widget.canvas.mapToScene(widget.canvas.viewport().rect()).boundingRect()
    assert view.width() < packet["image_rgb"].shape[1]
    assert view.contains(QPointF(78., 54.))
    widget.mode.setCurrentIndex(widget.mode.findData("brush"))
    assert widget.canvas.edit_mode == "pan" and draft.actions == []
    original = packet["image_rgb"].copy()
    transform = widget.canvas.transform()
    widget.set_brightness(180)
    assert draft.actions == []
    assert widget.canvas.transform() == transform
    np.testing.assert_array_equal(packet["image_rgb"], original)
    _choose(widget, "air", "brush")
    _brush(widget, 50., 38.)  # Still on the photo, outside the outlined native gap.
    assert draft.actions == [] and "changed nothing" in widget.status.text()
    assert not widget.continue_button.isEnabled()


def test_real_original_canvas_brush_maps_to_native_and_undo_redo_stay_local(step):
    widget, packet, draft = step
    notifications = []
    widget.preview_changed.connect(notifications.append)
    _choose(widget, "air", "brush")
    scene = _brush(widget, 76., 54.)
    assert len(draft.actions) == 1 and len(notifications) == 1
    action = draft.actions[0]
    expected = _project([[scene.x(), scene.y()]], np.linalg.inv(packet["native_to_photo_h"]))
    np.testing.assert_allclose(action["points_xy_native"], expected, rtol=0, atol=1e-12)
    assert action["kind"] == "brush" and action["class_name"] == "air"
    assert draft.class_masks["air"].any() and widget.undo.isEnabled()
    widget.undo.click()
    assert not draft.class_masks["air"].any() and widget.redo.isEnabled()
    widget.redo.click()
    assert draft.class_masks["air"].any() and len(notifications) == 3
    assert not packet.get("classification") and packet["gap_mask"].sum() == 29*49


def test_original_polygon_supports_multiple_classes_and_sand_only_requests_map_fix(step):
    widget, packet, draft = step
    requested, map_fixes = [], []
    widget.requested.connect(requested.append)
    widget.map_fix_requested.connect(map_fixes.append)
    _choose(widget, "air", "brush")
    _brush(widget, 68., 50.)
    _choose(widget, "sand", "polygon")
    for x, y in ((82., 55.), (90., 55.), (87., 62.)):
        QTest.mouseClick(widget.canvas.viewport(), Qt.MouseButton.LeftButton,
                         pos=_point(widget.canvas, x, y))
    assert len(draft.actions) == 1
    widget.finish.click()
    assert draft.actions[-1]["kind"] == "polygon" and len(draft.actions[-1]["points_xy_native"]) == 3
    assert draft.class_masks["air"].any() and draft.class_masks["sand"].any()
    assert requested == map_fixes == []
    widget.continue_button.click()
    assert requested == [] and len(map_fixes) == 1
    map_fixes[0]["class_masks"]["sand"][:] = False
    assert draft.class_masks["sand"].any()  # The receiver owns a detached snapshot.


def test_only_explicit_continue_requests_non_sand_classifications(step):
    widget, packet, draft = step
    requested, map_fixes = [], []
    widget.requested.connect(requested.append)
    widget.map_fix_requested.connect(map_fixes.append)
    _choose(widget, "air", "brush")
    _brush(widget, 68., 50.)
    _choose(widget, "free_water", "brush")
    _brush(widget, 84., 58.)
    assert draft.class_masks["air"].any() and draft.class_masks["free_water"].any()
    assert requested == map_fixes == []
    widget.continue_button.click()
    assert len(requested) == 1 and map_fixes == []
    assert requested[0]["actions"] == draft.actions


def test_one_click_and_one_key_switch_keep_exact_pan_zoom_and_draft(step, qtbot):
    widget, packet, draft = step
    _choose(widget, "free_water", "brush")
    _brush(widget, 76., 54.)
    widget.canvas.scale(1.7, 1.7)
    widget.canvas.centerOn(80., 57.)
    _choose(widget, "air", "polygon")
    for x, y in ((78., 53.), (84., 53.)):
        QTest.mouseClick(widget.canvas.viewport(), Qt.MouseButton.LeftButton,
                         pos=_point(widget.canvas, x, y))
    unfinished = list(widget.canvas._edit_points)
    assert len(unfinished) == 2
    transform = widget.canvas.transform()
    centre = widget.canvas.mapToScene(widget.canvas.viewport().rect().center())
    before = draft.snapshot()
    requests = []
    widget.photo_requested.connect(requests.append)
    widget.photo_toggle.click()
    assert requests == ["clearwater"] and not widget.canvas.isEnabled()
    other = dict(packet, image_role="clearwater", image_rgb=np.full_like(packet["image_rgb"], 130))
    other["valid_support"] = np.ones(other["image_rgb"].shape[:2], bool)
    other["valid_support"][0, 0] = False
    widget.set_photo(other)
    assert np.all(widget._plain[0, 0] == 64) and np.all(other["image_rgb"][0, 0] == 130)
    assert widget.canvas.transform() == transform
    assert widget.canvas.mapToScene(widget.canvas.viewport().rect().center()) == centre
    assert draft.actions == before["actions"] and widget.canvas.isEnabled()
    assert widget.canvas._edit_points == unfinished
    for key in draft.class_masks:
        np.testing.assert_array_equal(draft.class_masks[key], before["class_masks"][key])
    widget.activateWindow()
    qtbot.waitUntil(widget.isActiveWindow, timeout=1000)
    widget.canvas.setFocus()
    QTest.keyClick(widget.canvas, Qt.Key.Key_I, Qt.KeyboardModifier.AltModifier)
    qtbot.waitUntil(lambda: len(requests) == 2, timeout=1000)
    assert requests == ["clearwater", "current"]
    widget.set_photo(packet)
    assert widget.canvas.transform() == transform
    assert widget.canvas.mapToScene(widget.canvas.viewport().rect().center()) == centre
    assert draft.actions == before["actions"]
    assert widget.canvas._edit_points == unfinished
    widget.set_suspended(True)
    assert not widget.photo_shortcut.isEnabled() and not widget.photo_toggle.isEnabled()
    widget.request_other_photo()
    assert requests == ["clearwater", "current"]
    widget.set_suspended(False)
    assert widget.photo_shortcut.isEnabled()


@pytest.mark.parametrize("damage", ["grid", "transform", "shape", "fallback", "support_dtype", "support_shape"])
def test_other_photo_refuses_changed_native_identity_shape_or_current_fallback(step, damage):
    widget, packet, draft = step
    widget.photo_toggle.click()
    other = deepcopy(dict(packet, image_role="clearwater"))
    if damage == "grid":
        other["display_grid"]["model"] = {"synthetic": "another model"}
    elif damage == "transform":
        other["native_to_photo_h"][0][2] += 1
    elif damage == "shape":
        other["image_rgb"] = other["image_rgb"][:-1]
    elif damage == "fallback":
        other["image_role"] = "current"
    elif damage == "support_dtype":
        other["valid_support"] = np.ones(other["image_rgb"].shape[:2], np.uint8)
    else:
        other["valid_support"] = np.ones((3, 4), bool)
    with pytest.raises(ValueError, match="exact native display grid|exact boolean display support"):
        widget.set_photo(other)
    assert widget._photo_role == "current" and draft.actions == []
    widget.photo_failed("The selected source was refused")
    assert widget.canvas.isEnabled() and "refused" in widget.status.text()
