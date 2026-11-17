"""Synchronized letterbox for RGB images and segmentation masks."""
import numpy as np
from PIL import Image


def letterbox(image, mask, size=(256, 256)):
    """Return RGB and uint8 mask; size=(width,height), padding labels ignored.

    Mask values must already be 0/background, 1/cat, 2/dog, 255/ignore.
    Coordinates are in the original file orientation; no isolated EXIF transpose.
    """
    if image.size != mask.size:
        raise ValueError('Image/mask size mismatch')
    if not set(np.unique(np.asarray(mask))).issubset({0, 1, 2, 255}):
        raise ValueError('Unexpected target mask labels')
    width, height = size
    if min(width, height) <= 0:
        raise ValueError('Positive target dimensions required')
    ratio = min(width / image.width, height / image.height)
    resized = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    offset = ((width - resized[0]) // 2, (height - resized[1]) // 2)
    rgb = Image.new('RGB', size, color=(0, 0, 0))
    target = Image.new('L', size, color=255)
    rgb.paste(image.convert('RGB').resize(resized, Image.Resampling.BILINEAR), offset)
    target.paste(mask.convert('L').resize(resized, Image.Resampling.NEAREST), offset)
    return rgb, target
