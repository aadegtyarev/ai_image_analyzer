import base64
import json
import pytest

from ai_image_analyzer.providers.openai_provider import OpenAIProvider


class DummyCfg:
    api_key = "x"
    base_url = "https://api.example"
    model = "gpt-image-1"


def test_check_balance_http(monkeypatch):
    called = {}

    def fake_get(url, headers=None, timeout=None):
        called["url"] = url

        class R:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"balance": 42}

        return R()

    monkeypatch.setattr("requests.get", fake_get)
    p = OpenAIProvider()
    res = p.check_balance(DummyCfg())
    assert res["balance"] == 42
    assert called["url"].endswith("/balance")


def test_call_image_via_http(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        captured["url"] = url
        captured["files"] = files
        captured["data"] = data

        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {"text": "analysis result", "usage": {"total_cost": 0.5}}

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    p = OpenAIProvider()
    text, usage = p.call_image(DummyCfg(), b"\xFF\xD8\xFF", system_prompt="do it")
    assert text == "analysis result"
    assert usage["total_cost"] == 0.5
    assert captured["url"].endswith("/images")


def test_call_image_via_client(monkeypatch):
    class DummyResponses:
        def create(self, model=None, input=None):
            # emulate SDK returning dict-like response
            return {"output": [{"content": "ok from sdk"}], "usage": {"total_cost": 0.2}}

    class DummyClient:
        responses = DummyResponses()

    cfg = DummyCfg()
    p = OpenAIProvider(client=DummyClient())
    text, usage = p.call_image(cfg, b"xxx")
    assert text == "ok from sdk"
    assert usage["total_cost"] == 0.2
