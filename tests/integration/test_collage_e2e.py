import io
from PIL import Image
import base64

from ai_image_analyzer import handle_json_request


def make_jpeg_bytes(color=(255, 0, 0), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


def test_collage_end_to_end(monkeypatch):
    img1 = make_jpeg_bytes((255, 0, 0))
    img2 = make_jpeg_bytes((0, 255, 0))

    b1 = base64.b64encode(img1).decode("ascii")
    b2 = base64.b64encode(img2).decode("ascii")

    captured = {}

    def fake_call(cfg, jpeg_bytes, system_prompt=None, user_text=None, quiet=False, image_meta=None):
        # Ensure jpeg_bytes is valid JPEG
        from PIL import Image
        import io

        im = Image.open(io.BytesIO(jpeg_bytes))
        assert im.format == "JPEG"
        captured["size"] = len(jpeg_bytes)
        return ("ok", {"total_cost": 0.7})

    monkeypatch.setattr("ai_image_analyzer.call_model_with_image", fake_call)

    req = {"action": "analyze", "images": [{"name": "a.jpg", "data_b64": b1}, {"name": "b.jpg", "data_b64": b2}], "per_image": False, "include_billing": True}
    resp = handle_json_request(req)
    assert resp["ok"]
    assert resp["mode"] == "collage"
    assert resp["usage"]["total_cost"] == 0.7
    assert captured.get("size", 0) > 0
