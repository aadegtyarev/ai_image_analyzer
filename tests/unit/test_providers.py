import pytest

from ai_image_analyzer.providers.base import ModelProvider
from ai_image_analyzer.providers.openai_provider import OpenAIProvider


def test_openai_provider_check_balance_requires_base_url(tmp_path, monkeypatch):
    class C: api_key = "x"; base_url = None

    p = OpenAIProvider()
    with pytest.raises(RuntimeError):
        p.check_balance(C())


def test_openai_provider_raises_when_sdk_missing(monkeypatch):
    # If SDK is missing and no client is provided, operations raise
    cfg = type("C", (), {"api_key": "x", "base_url": "https://example.com"})
    p = OpenAIProvider()
    # call_text and call_image raise by design (unimplemented)
    with pytest.raises(RuntimeError):
        p.call_text(cfg, "hello")
    with pytest.raises(RuntimeError):
        p.call_image(cfg, b"abc")
