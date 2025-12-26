from __future__ import annotations

from typing import Tuple, List
from .image_processing import resize_image_bytes, make_collage


def load_and_resize_image(path: str, max_size: int, jpeg_quality: int) -> Tuple[bytes, int, int, str]:
    with open(path, "rb") as f:
        data = f.read()
    return resize_image_bytes(data, max_size, jpeg_quality)


def resize_image_from_bytes(data: bytes, max_size: int, jpeg_quality: int) -> Tuple[bytes, int, int, str]:
    return resize_image_bytes(data, max_size, jpeg_quality)


def make_collage_wrapper(named_images: List[tuple], max_size: int, quality: int):
    return make_collage(named_images, max_size, quality)
