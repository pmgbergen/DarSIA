"""Fixed FF_BM RAW-to-feature colour contract; no fitted photometry or JPEG."""
import numpy as np


def encode_training_rgb(linear_dn_hwc, valid_support, *, white_range_dn,
                        white_balance, exposure, matrix, transfer, chunk_rows=128):
    """Apply the supplied fixed camera-WB and sRGB transfer, retaining float32.

    This matches the existing camera-U display arithmetic before its final
    8-bit rounding. RAW DN are sampled geometrically before this transfer.
    It does not claim that camera RGB is colorimetrically calibrated sRGB.
    """
    source = np.asarray(linear_dn_hwc)
    support = np.asarray(valid_support)
    balance = np.asarray(white_balance, np.float64)
    if (source.ndim != 3 or source.shape[2] != 3 or support.dtype != bool
            or support.shape != source.shape[:2] or balance.shape != (3,)
            or not np.isfinite(balance).all() or (balance <= 0).any()
            or not np.isfinite(white_range_dn) or white_range_dn <= 0
            or not np.isfinite(exposure) or exposure <= 0 or matrix is not None
            or transfer != 'srgb' or chunk_rows <= 0):
        raise ValueError('An explicit fixed BM camera-WB/sRGB contract is required')
    output = np.full(source.shape, np.nan, np.float32)
    for y in range(0, len(source), chunk_rows):
        end = min(len(source), y+chunk_rows)
        valid = support[y:end]
        values = np.asarray(source[y:end], np.float64)
        if not np.isfinite(values[valid]).all():
            raise ValueError('Supported RAW samples must have finite camera DN')
        scaled = np.clip(values*(balance*float(exposure)/float(white_range_dn)), 0, 1)
        encoded = np.where(scaled <= .0031308, 12.92*scaled,
                           1.055*np.power(scaled, 1/2.4)-.055)
        encoded[~valid] = np.nan
        output[y:end] = encoded.astype(np.float32)
    return output
