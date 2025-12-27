import base64
import tempfile

from ai_image_analyzer import handle_json_request


def encode_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def test_json_api_per_image(monkeypatch):
    # two small "images" as bytes; monkeypatch the model call
    calls = []

    def fake_call(cfg, jpeg_bytes, system_prompt=None, user_text=None, quiet=False, image_meta=None):
        calls.append((image_meta or {}).get("name"))
        return (f"ok for {image_meta.get('name')}", {"total_cost": 0.1})

    monkeypatch.setattr("ai_image_analyzer.call_model_with_image", fake_call)
    # stub image resize to avoid needing real image bytes
    monkeypatch.setattr("ai_image_analyzer.json_api.resize_image_from_bytes", lambda b, ms, q: (b, 1, 1, "square"))

    img1 = encode_bytes(b"abc")
    img2 = encode_bytes(b"def")
    req = {"action": "analyze", "images": [{"name": "a.jpg", "data_b64": img1}, {"name": "b.jpg", "data_b64": img2}], "per_image": True, "include_billing": True}
    resp = handle_json_request(req)
    assert resp["ok"]
    assert resp["mode"] == "per_image"
    assert isinstance(resp["results"], list) and len(resp["results"]) == 2
    assert resp["usage"]["total_cost"] == 0.2


def test_json_api_collage(monkeypatch):
    def fake_call(cfg, jpeg_bytes, system_prompt=None, user_text=None, quiet=False, image_meta=None):
        return ("collage result", {"total_cost": 0.5})

    monkeypatch.setattr("ai_image_analyzer.call_model_with_image", fake_call)
    monkeypatch.setattr("ai_image_analyzer.json_api.resize_image_from_bytes", lambda b, ms, q: (b, 1, 1, "square"))
    # avoid real PIL collage work: stub make_collage_wrapper
    monkeypatch.setattr("ai_image_analyzer.json_api.make_collage_wrapper", lambda imgs, ms, q: (b"collage", [n for n, _ in imgs]))

    img1 = encode_bytes(b"x")
    img2 = encode_bytes(b"y")
    req = {"action": "analyze", "images": [{"name": "a.jpg", "data_b64": img1}, {"name": "b.jpg", "data_b64": img2}], "per_image": False, "include_billing": True}
    resp = handle_json_request(req)
    assert resp["ok"]
    assert resp["mode"] == "collage"
    assert resp["results"] == "collage result"
    assert resp["usage"]["total_cost"] == 0.5


def test_json_api_invalid_base64():
    req = {"action": "analyze", "images": [{"name": "bad.jpg", "data_b64": "notbase64"}]}
    resp = handle_json_request(req)
    assert resp["ok"] is False
    assert any("invalid base64" in e for e in resp.get("errors", []))


def test_json_api_path(tmp_path, monkeypatch):
    # create a small binary file and ensure path-based processing works
    p = tmp_path / "img.jpg"
    p.write_bytes(b"xxx")

    def fake_call(cfg, jpeg_bytes, system_prompt=None, user_text=None, quiet=False, image_meta=None):
        return ("ok", None)

    monkeypatch.setattr("ai_image_analyzer.call_model_with_image", fake_call)
    monkeypatch.setattr("ai_image_analyzer.json_api.load_and_resize_image", lambda path, ms, q: (open(path, 'rb').read(), 1, 1, "square"))
    req = {"action": "analyze", "images": [{"name": "on_disk.jpg", "path": str(p)}], "per_image": True}
    resp = handle_json_request(req)
    assert resp["ok"]
    assert resp["mode"] == "per_image"
