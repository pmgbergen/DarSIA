"""The Setup crop is chosen and marked next to the reference quadrilateral."""
import json
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import QGraphicsPolygonItem, QGraphicsScene, QGraphicsView, QWidget

from darsia.fluidflower import layer_painting
from darsia.fluidflower import series_rig_frame_panel as rig_ui
from darsia.fluidflower import setup_crop_area as crop
from darsia.fluidflower.series_geometry import (
    RigFrame,
    create_series_geometry_project,
    native_series_link,
    save_rig_frame,
)

from .test_painting_corrections import view  # noqa: F401

# 100 px per metre: the reference quad is the rig's 2.0 m x 1.0 m inner frame.
QUAD = {"top_left": (10.0, 20.0), "bottom_left": (110.0, 20.0),
        "bottom_right": (110.0, 220.0), "top_right": (10.0, 220.0)}
WIDTH, HEIGHT = 240, 130


class Canvas(QGraphicsView):
    image_point_selected = Signal(float, float)

    def __init__(self, parent):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        pixmap = QPixmap(WIDTH, HEIGHT)
        pixmap.fill()
        self.pixmap_item = self.scene().addPixmap(pixmap)
        self.place_mode = False
        self.photo_display_mapping = object()
        self.preview = None

    def raw_to_scene(self, row, col):
        return QPointF(col, row)


@pytest.fixture
def context(qtbot, monkeypatch, tmp_path):
    photo = tmp_path / "ac1.png"
    photo.write_bytes(b"photo")
    project = create_series_geometry_project(
        series_id="local", series_name="Local", series_root=tmp_path / "series")
    save_rig_frame(project, RigFrame(corners=dict(QUAD), width_m=2.0, height_m=1.0, run_id="ac1",
                                     image=str(photo), image_size=(WIDTH, HEIGHT)))
    monkeypatch.setattr(rig_ui, "rig_frame_from_inputs", lambda reference: None)
    monkeypatch.setattr(rig_ui, "photo_size", lambda path, camera_u=None: (WIDTH, HEIGHT))
    monkeypatch.setattr(layer_painting, "reference_camera_u", lambda project, reference: None)
    page = QWidget()
    qtbot.addWidget(page)
    page.canvas = Canvas(page)
    reference = SimpleNamespace(run_id="ac1", baseline_image=photo)
    panel = rig_ui.RigFramePanel(page, project, lambda: reference, scrollable=False)
    qtbot.addWidget(panel)
    preview = {"run_id": "ac1", "path": str(photo)}
    page.canvas.preview = preview
    panel.use_host_photo((WIDTH, HEIGHT), preview)
    return SimpleNamespace(panel=panel, project=project, page=page, photo=photo)


def click(panel, row, col):
    panel.canvas.image_point_selected.emit(col / (WIDTH - 1), row / (HEIGHT - 1))


def crop_polygons(panel, style):
    return [item for item in panel._overlay if isinstance(item, QGraphicsPolygonItem)
            and item.pen().color().getRgb()[:3] == (255, 170, 0) and item.pen().style() == style]


def corners_of(polygon):
    return [(round(point.y(), 6), round(point.x(), 6)) for point in polygon.polygon()]


def test_the_default_crop_is_the_reference_quad_and_writes_nothing(context):
    panel, manifest = context.panel, context.project.root / "series.json"
    before = manifest.read_bytes()
    panel.load()
    assert panel.crop_same.isChecked() and panel.crop_same.isEnabled()
    assert panel.crop_separate.isEnabled() and not panel.crop_separate.isChecked()
    assert panel.mark_crop.isHidden() and panel.crop_note.isHidden()
    assert panel.crop_status.text() == "Setup crops to the reference quadrilateral."
    assert not crop_polygons(panel, Qt.PenStyle.DashLine) and not crop_polygons(panel, Qt.PenStyle.SolidLine)
    panel.crop_same.click()
    panel.copy_crop.click()
    assert "top_left = [10, 20]" in QGuiApplication.clipboard().text()
    assert "origin" not in QGuiApplication.clipboard().text()
    assert manifest.read_bytes() == before
    assert "crop_choice" not in native_series_link(context.project)


def test_a_separate_area_starts_as_the_reference_quad_and_switching_back_keeps_it(context):
    panel, project = context.panel, context.project
    panel.crop_separate.click()
    assert crop.saved_crop_choice(project) == crop.CROP_SEPARATE_AREA
    area = crop.saved_crop_areas(project)["ac1"]
    assert dict(area.corners) == QUAD and area.image == str(context.photo)
    assert area.image_size == (WIDTH, HEIGHT)
    assert not panel.mark_crop.isHidden() and panel.mark_crop.isEnabled()
    assert not panel.crop_note.isHidden() and "reference run's photograph" in panel.crop_note.text()
    assert "x 0.0 to 2000.0 mm and y 0.0 to 1000.0 mm (2000.0 x 1000.0 mm)" in panel.crop_status.text()
    [rectangle] = crop_polygons(panel, Qt.PenStyle.SolidLine)
    assert corners_of(rectangle) == [QUAD[name] for name in rig_ui.CLICK_ORDER]
    panel.crop_same.click()
    assert crop.saved_crop_choice(project) == crop.CROP_REFERENCE_QUAD
    assert crop.saved_crop_areas(project)["ac1"] == area
    assert panel.mark_crop.isHidden() and not crop_polygons(panel, Qt.PenStyle.SolidLine)


def test_four_clicks_mark_the_area_and_show_the_rectangle_setup_crops_to(context):
    panel, project = context.panel, context.project
    panel.crop_separate.click()
    first = crop.saved_crop_areas(project)["ac1"]
    panel.mark_crop.click()
    assert panel._mode == "crop" and panel.status.text().startswith("Click crop corner 1/4 (Top left)")
    click(panel, 30, 60)
    click(panel, 28, 180)
    panel._cancel_placement()  # Esc: nothing is stored.
    assert not panel.mark_crop.isChecked() and crop.saved_crop_areas(project)["ac1"] == first
    panel.mark_crop.click()
    for row, col in ((30, 60), (28, 180), (90, 182)):
        click(panel, row, col)
    assert panel.status.text().startswith("Click crop corner 4/4 (Bottom left)")
    assert len(crop_polygons(panel, Qt.PenStyle.SolidLine)) == 0
    click(panel, 92, 58)
    assert panel._mode == "" and not panel.mark_crop.isChecked()
    area = crop.saved_crop_areas(project)["ac1"]
    assert {name: tuple(round(v, 6) for v in point) for name, point in area.corners.items()} == {
        "top_left": (30.0, 60.0), "top_right": (28.0, 180.0),
        "bottom_right": (90.0, 182.0), "bottom_left": (92.0, 58.0)}
    assert area.image == str(context.photo) and area.run_id == "ac1"
    # The rectangle lies inside every marked point: x 0.40-1.60 m, y 0.20-0.80 m.
    assert "x 400.0 to 1600.0 mm and y 200.0 to 800.0 mm (1200.0 x 600.0 mm)" in panel.crop_status.text()
    [rectangle] = crop_polygons(panel, Qt.PenStyle.SolidLine)
    assert corners_of(rectangle) == [(30.0, 60.0), (30.0, 180.0), (90.0, 180.0), (90.0, 60.0)]
    assert len(crop_polygons(panel, Qt.PenStyle.DashLine)) == 1
    # Copy crop section copies the crop Setup uses, with the reference quad's coordinates.
    panel.copy_crop.click()
    copied = QGuiApplication.clipboard().text()
    assert "top_left = [30, 60]" in copied and "origin = [0.4, 0.8]" in copied
    assert "width = 1.2" in copied and "height = 0.6" in copied


def test_an_area_without_a_rectangle_is_refused_and_the_saved_area_stays(context):
    panel, project = context.panel, context.project
    panel.crop_separate.click()
    first = crop.saved_crop_areas(project)["ac1"]
    panel.mark_crop.click()
    for row, col in ((30, 100), (30, 110), (90, 130), (90, 120)):
        click(panel, row, col)
    assert crop.saved_crop_areas(project)["ac1"] == first
    assert panel.crop_status.text().startswith("Crop area not saved: The separate crop area leaves no rectangle")


def test_crop_corners_are_clicked_only_on_the_quads_photograph(context, qtbot):
    panel = context.panel
    panel.crop_separate.click()
    other = {"run_id": "ac2", "path": str(context.photo)}
    context.page.canvas.preview = other
    panel.use_host_photo((WIDTH, HEIGHT), other)
    panel.mark_crop.click()
    assert panel._mode == "" and not panel.mark_crop.isChecked()
    assert panel.crop_status.text().startswith("Show the reference run's photograph first.")
    larger = {"run_id": "ac1", "path": str(context.photo)}
    context.page.canvas.preview = larger
    panel.use_host_photo((WIDTH + 2, HEIGHT), larger)
    panel.mark_crop.click()
    assert panel._mode == ""
    assert "another size" in panel.crop_status.text()


def test_geometry_on_the_corrected_grid_cannot_choose_a_separate_area(context):
    panel, project = context.panel, context.project
    current = project.root / "overrides" / "geometry_facies" / "current.json"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text(json.dumps({"configuration": {"frame": "rig-corrected"}}), encoding="utf-8")
    panel.load()
    assert not panel.crop_separate.isEnabled() and "corrected rig grid" in panel.crop_separate.toolTip()
    # A separate choice stored earlier stays visible, with the reason, and can be changed back.
    crop.save_crop_choice(project, crop.CROP_SEPARATE_AREA)
    panel.load()
    assert panel.crop_separate.isChecked() and panel.crop_separate.isEnabled()
    assert not panel.mark_crop.isEnabled() and "corrected rig grid" in panel.crop_status.text()
    panel.crop_same.click()
    assert crop.saved_crop_choice(project) == crop.CROP_REFERENCE_QUAD
    assert not panel.crop_separate.isEnabled()


def test_crop_corner_clicks_pause_the_other_tools_of_the_layer_view(view, monkeypatch):
    store = {"choice": crop.CROP_REFERENCE_QUAD, "areas": {}}
    monkeypatch.setattr(rig_ui, "saved_crop_choice", lambda project: store["choice"])
    monkeypatch.setattr(rig_ui, "save_crop_choice", lambda project, choice, **_: store.update(choice=choice))
    monkeypatch.setattr(rig_ui, "saved_crop_areas", lambda project: dict(store["areas"]))
    monkeypatch.setattr(rig_ui, "save_crop_area",
                        lambda project, area, **_: store["areas"].update({area.run_id: area}))
    panel = view.rig_panel
    panel.frame = RigFrame(corners={"top_left": (0., 0.), "top_right": (0., 79.),
                                    "bottom_right": (59., 79.), "bottom_left": (59., 0.)},
                           width_m=.079, height_m=.059, run_id="ac52",
                           image=view.session.image_path, image_size=(80, 60))
    panel.crop_separate.click()
    assert store["choice"] == crop.CROP_SEPARATE_AREA and "ac52" in store["areas"]
    panel.mark_crop.click()
    assert panel._mode == "crop" and view.canvas.place_mode
    assert "Finish placing" in view.measured_edges.note.text()
    for row, col in ((5, 5), (5, 70), (50, 72), (52, 6)):
        panel.canvas.image_point_selected.emit(col / 79, row / 59)
    assert panel._mode == "" and not view.canvas.place_mode
    assert "Finish placing" not in view.measured_edges.note.text()
    assert store["areas"]["ac52"].corners["top_right"] == pytest.approx((5., 70.))
    assert "x 6.0 to 70.0 mm and y 9.0 to 54.0 mm" in panel.crop_status.text()
