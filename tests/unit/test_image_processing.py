from ai_image_analyzer.image_processing import resize_image_bytes, make_collage
from PIL import Image
import io


def make_in_memory_jpeg(w=200, h=150, color=(100, 150, 200)):
    img = Image.new("RGB", (w, h), color=color)
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=90)
    return b.getvalue()


def test_resize_image_bytes_smaller():
    data = make_in_memory_jpeg(100, 80)
    out, w, h, orient = resize_image_bytes(data, max_size=200, quality=85)
    assert isinstance(out, (bytes, bytearray))
    assert w == 100 and h == 80
    assert orient in ("portrait", "landscape", "square")


def test_make_collage_basic():
    imgs = [(f"i{i}.jpg", make_in_memory_jpeg(64, 64)) for i in range(4)]
    out, names = make_collage(imgs, max_size=256, quality=80)
    assert isinstance(out, (bytes, bytearray))
    assert len(names) == 4
