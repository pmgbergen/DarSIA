"""Setup offers the strict image-file policies (only RAW, only JPEG/TIFF/PNG)."""

from darsia.fluidflower import image_formats as formats
from darsia.fluidflower.pages.setup.steps.image_sources import ImageSourcesPage


def test_the_image_sources_step_offers_and_keeps_the_strict_policies(qtbot):
    page = ImageSourcesPage()
    qtbot.addWidget(page)
    box = page.image_file_policy
    offered = [box.itemData(index) for index in range(box.count())]
    assert offered[0] == formats.IMAGE_FILE_POLICY_PROCESSED
    assert formats.IMAGE_FILE_POLICY_RAW_ONLY in offered
    assert formats.IMAGE_FILE_POLICY_PROCESSED_ONLY in offered
    assert page.image_file_policy_value() == formats.DEFAULT_IMAGE_FILE_POLICY
    for policy in (formats.IMAGE_FILE_POLICY_RAW_ONLY, formats.IMAGE_FILE_POLICY_PROCESSED_ONLY):
        page.set_image_file_policy(policy)
        assert page.image_file_policy_value() == policy
    assert "never fall back" in box.toolTip()
