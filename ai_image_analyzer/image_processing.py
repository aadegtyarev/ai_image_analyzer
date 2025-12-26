from __future__ import annotations

from typing import List, Tuple, Optional
import io
from PIL import Image, ImageOps


def resize_image_bytes(data: bytes, max_size: int, quality: int) -> Tuple[bytes, int, int, str]:
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        w, h = im.size
        if h > w:
            orientation = "portrait"
        elif w > h:
            orientation = "landscape"
        else:
            orientation = "square"
        scale = min(1.0, float(max_size) / max(w, h))
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            im = im.resize(new_size, Image.LANCZOS)
        final_w, final_h = im.size
        buf = io.BytesIO()
        try:
            im.save(buf, format="JPEG", quality=quality, optimize=True)
        except Exception:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=min(quality, 95), optimize=False)
        out = buf.getvalue()
        if not out:
            raise RuntimeError("resize resulted in empty bytes")
        return out, final_w, final_h, orientation


def make_collage(named_images: List[Tuple[str, bytes]], max_size: int, quality: int) -> Tuple[bytes, List[str]]:
    # simple square grid collage
    n = len(named_images)
    if n == 0:
        raise ValueError("no images")
    import math

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    # open images and resize to cell size
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for _, b in named_images]
    # compute cell size as max_size / max(cols, rows)
    cell = max(1, max_size // max(cols, rows))
    resized = [img.resize((cell, cell), Image.LANCZOS) for img in imgs]
    out_w = cols * cell
    out_h = rows * cell
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    for idx, img in enumerate(resized):
        x = (idx % cols) * cell
        y = (idx // cols) * cell
        canvas.paste(img, (x, y))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), [name for name, _ in named_images]


def build_collage_system_prompt(base_prompt: str, filenames: List[str]) -> str:
    files_str = ", ".join(filenames)
    prefix = (
        "Мы отправляем общий коллаж. "
        "Каждая ячейка коллажа соответствует отдельному исходному файлу. "
        f"Порядок файлов слева направо, сверху вниз: {files_str}.\n\n"
    )
    return prefix + (base_prompt or "")
