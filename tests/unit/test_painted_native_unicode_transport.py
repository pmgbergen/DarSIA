"""New native metadata survives the existing platform-specific TOML reader."""
import json

import pytest

from darsia.fluidflower import corrected_setup
from darsia.fluidflower import painted_setup_apply
from darsia.presets.workflows.config.rig import RigConfig


@pytest.mark.parametrize("route", ["ordinary", "raw"])
def test_native_unicode_metadata_uses_ascii_transport_and_original_consumer(tmp_path, route):
    native = dict(format="darsia-painted-native-product-v1", run_id="one",
        analysis_exclusions=None, provenance={"name": "æøå", "path": "original/æ",
            "mixed": [None, [], {}, True, 1, 1.0]})
    before = (b'[rig]\nwidth = 0.9\nheight = 0.49\ndim = 2\npath = "old-rig"\n'
        b'[data]\nuse_cache = true\n[labeling]\nlabels = "old-labels"\nframe = "camera"\n'
        b'[facies]\npath = "old-facies"\nprops = "old-properties"\n')
    if route == "ordinary":
        replacements = dict(labeling=dict(labels="new-labels", frame="rig-corrected"),
            facies=dict(path="new-facies", props="new-properties"))
        encoded = corrected_setup.corrected_config_bytes(before, replacements,
            native_rig=dict(native_product=native))
    else:
        camera = dict(configuration=str(tmp_path / "camera.json"), sha256="1" * 64,
            profile_id="explicit", run_id="one")
        encoded = painted_setup_apply.raw_settings_bytes(before,
            dict(run_id="one", raw_camera=camera, rig_path=str(tmp_path / "rig"), native_product=native))
    assert encoded.isascii(), "Only the new JSON transport is escaped; no global reader change"
    path = tmp_path / "new-settings.toml"
    path.write_bytes(encoded)
    actual = RigConfig().load(path, results=tmp_path).native_product
    assert json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(native, sort_keys=True, allow_nan=False)
    assert actual["analysis_exclusions"] is None
