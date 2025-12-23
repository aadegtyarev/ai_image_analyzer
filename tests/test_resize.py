import os
import tempfile
from PIL import Image
from ai_image_analyzer import load_and_resize_image


def make_temp_image(w=800, h=600, color=(123, 222, 64)):
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img = Image.new("RGB", (w, h), color=color)
    img.save(path, format="JPEG", quality=90)
    return path


def test_load_and_resize_image_basic():
    path = make_temp_image(800, 600)
    try:
        bts, w, h, orient = load_and_resize_image(path, max_size=512, jpeg_quality=85)
        assert isinstance(bts, (bytes, bytearray))
        assert len(bts) > 0
        assert max(w, h) <= 512
        assert orient in ("portrait", "landscape", "square")
    finally:
        os.remove(path)


def test_load_and_resize_image_no_resize():
    path = make_temp_image(200, 150)
    try:
        bts, w, h, orient = load_and_resize_image(path, max_size=1024, jpeg_quality=85)
        assert len(bts) > 0
        assert w == 200 and h == 150
    finally:
        os.remove(path)
