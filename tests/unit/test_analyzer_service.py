import pytest

from ai_image_analyzer.services.analyzer import AnalyzerService


class DummyProvider:
    def call_text(self, cfg, text, system_prompt=None, quiet=False):
        return f"ok: {text}", {"total_cost": 0.1}

    def call_image(self, cfg, bytes_, system_prompt=None, user_text=None, quiet=False, image_meta=None):
        return "img-ok", {"total_cost": 0.2}

    def check_balance(self, cfg, quiet=False):
        return {"balance": 1.23}


def test_analyze_text_and_image_and_balance():
    service = AnalyzerService(DummyProvider())
    txt, usage = service.analyze_text(None, "hello")
    assert "ok: hello" in txt
    assert usage and usage.get("total_cost") == 0.1

    txt2, usage2 = service.analyze_image(None, b"x")
    assert txt2 == "img-ok"
    assert usage2 and usage2.get("total_cost") == 0.2

    bal = service.check_balance(None)
    assert bal["balance"] == 1.23
