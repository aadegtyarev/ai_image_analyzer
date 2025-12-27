"""Minimal OpenAI provider implementation with graceful fallback for tests."""
from __future__ import annotations

from typing import Any, Optional, Tuple
import json

try:
    # Prefer the official OpenAI-compatible SDK when available
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - behavior exercised via tests
    OpenAI = None  # type: ignore

from ..providers.base import ModelProvider


class OpenAIProvider:
    """Provider that uses OpenAI-compatible SDK or falls back to HTTP check for balance.

    This is intentionally small: tests should mock out network calls.
    """

    def __init__(self, client: Optional[Any] = None):
        # client can be injected (OpenAI client) to ease testing
        self._client = client

    def _build_client(self, cfg: Any):
        if self._client:
            return self._client
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK is not installed and no client provided")
        kwargs = {"api_key": getattr(cfg, "api_key", None)}
        if getattr(cfg, "base_url", None):
            kwargs["base_url"] = cfg.base_url
        return OpenAI(**kwargs)

    def call_text(self, cfg: Any, text: str, system_prompt: Optional[str] = None, quiet: bool = False) -> Any:
        # Simple wrapper: tests should mock _build_client and behaviour
        client = self._build_client(cfg)
        # This is intentionally abstract: real implementation depends on provider API
        raise RuntimeError("OpenAIProvider.call_text not implemented; mock in tests")

    def call_image(self, cfg: Any, jpeg_bytes: bytes, system_prompt: Optional[str] = None, user_text: Optional[str] = None, quiet: bool = False, image_meta: Optional[dict] = None) -> Any:
        client = self._build_client(cfg)
        raise RuntimeError("OpenAIProvider.call_image not implemented; mock in tests")

    def check_balance(self, cfg: Any, quiet: bool = False) -> dict:
        # If the SDK exposes a balance endpoint, clients can implement it;
        # otherwise, fallback to HTTP GET to cfg.base_url + '/balance' via requests
        if getattr(cfg, "base_url", None) is None:
            raise RuntimeError("OPENAI_BASE_URL is not set; cannot check balance")
        import requests

        url = cfg.base_url.rstrip("/") + "/balance"
        headers = {"Authorization": f"Bearer {cfg.api_key}"}
        resp = requests.get(url, headers=headers, timeout=getattr(cfg, "timeout", 30))
        resp.raise_for_status()
        return resp.json()
