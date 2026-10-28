"""The annotation decoder keeps physical RGB units and explicit reduction."""
import numpy as np
from PIL import Image
from darsia.fluidflower.photograph_io import load_reduced


def test_reduced_photograph_retains_float32_rgb_units(tmp_path):
    path = tmp_path / "photograph.png"
    pixels = np.array([[[0, 128, 255], [255, 0, 128]]], dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    decoded = load_reduced(path, 1)
    assert decoded.dtype == np.float32
    assert decoded.shape == pixels.shape
    assert decoded[0, 0, 0] == np.float32(0)
    assert decoded[0, 0, 2] == np.float32(1)
    assert decoded[0, 0, 1] == np.float32(128) / np.float32(255)


def test_reduced_photograph_uses_exact_box_grid(tmp_path):
    path = tmp_path / "photograph.png"
    Image.new("RGB", (8, 6), (255, 0, 0)).save(path)
    decoded = load_reduced(path, 2)
    assert decoded.shape == (3, 4, 3)
    assert decoded.dtype == np.float32
    assert np.all(decoded[..., 0] == np.float32(1))
    assert np.all(decoded[..., 1:] == np.float32(0))
