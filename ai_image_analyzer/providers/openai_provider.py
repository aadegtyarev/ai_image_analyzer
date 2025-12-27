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
        # Basic implementation: try to use injected SDK client if available.
        client = None
        try:
            client = self._build_client(cfg)
        except Exception:
            client = self._client

        if client is not None and hasattr(client, "responses") and callable(getattr(client.responses, "create", None)):
            # SDK-like path
            resp = client.responses.create(model=getattr(cfg, "model", "gpt-4"), input=(system_prompt or "") + "\n" + text)
            # resp may be dict-like or have .to_dict()
            if hasattr(resp, "to_dict"):
                resp = resp.to_dict()
            if isinstance(resp, dict):
                if "text" in resp:
                    return resp["text"]
                if "output" in resp and isinstance(resp["output"], list):
                    parts = [p.get("content") or p.get("text") for p in resp["output"] if isinstance(p, dict)]
                    return "\n".join([p for p in parts if p])
            raise RuntimeError("OpenAIProvider.call_text: unexpected SDK response shape")

        # Fallback: if base_url is configured, call HTTP responses endpoint
        if getattr(cfg, "base_url", None):
            import requests

            url = cfg.base_url.rstrip("/") + "/responses"
            headers = {"Authorization": f"Bearer {cfg.api_key}"}
            payload = {"input": (system_prompt or "") + "\n" + text}
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=getattr(cfg, "timeout", 30))
                resp.raise_for_status()
                data = resp.json()
                # Expect {'text': '...', 'usage': {...}}
                return data.get("text") if data is not None else None
            except Exception as e:  # pragma: no cover - network failure paths
                raise RuntimeError(f"OpenAIProvider.call_text HTTP fallback failed: {e}") from e

        raise RuntimeError("OpenAIProvider.call_text not implemented; provide client or mock in tests")

    def call_image(self, cfg: Any, jpeg_bytes: bytes, system_prompt: Optional[str] = None, user_text: Optional[str] = None, quiet: bool = False, image_meta: Optional[dict] = None) -> Any:
        # Prefer explicit injected client; otherwise prefer HTTP when base_url is set.
        client = self._client

        if client is None and getattr(cfg, "base_url", None):
            # Use HTTP fallback (below)
            client = None

        # If no base_url or an explicit client was provided, try SDK client
        if client is None and not getattr(cfg, "base_url", None):
            try:
                client = self._build_client(cfg)
            except Exception:
                client = None

        if client is not None and hasattr(client, "responses") and callable(getattr(client.responses, "create", None)):
            # Build SDK-compatible input: many SDKs accept structured inputs
            inp = []
            if system_prompt:
                inp.append({"role": "system", "content": system_prompt})
            if user_text:
                inp.append({"role": "user", "content": user_text})
            # For images, include raw bytes in a known key (tests will supply compatible client)
            inp.append({"image": jpeg_bytes})
            resp = client.responses.create(model=getattr(cfg, "model", "gpt-image-1"), input=inp)
            if hasattr(resp, "to_dict"):
                resp = resp.to_dict()
            # Parse SDK-like response shapes
            if isinstance(resp, dict):
                # Common: {'text': '...', 'usage': {...}}
                if "text" in resp:
                    return resp["text"], resp.get("usage")
                if "output" in resp and isinstance(resp["output"], list):
                    parts = [p.get("content") or p.get("text") for p in resp["output"] if isinstance(p, dict)]
                    return "\n".join([p for p in parts if p]), resp.get("usage")
            raise RuntimeError("OpenAIProvider.call_image: unexpected SDK response shape")

        # Fallback to HTTP multipart POST to cfg.base_url + '/images'
        if getattr(cfg, "base_url", None):
            import requests
            try:
                url = cfg.base_url.rstrip("/") + "/images"
                headers = {"Authorization": f"Bearer {cfg.api_key}"}
                files = {"image": ("image.jpg", jpeg_bytes, "image/jpeg")}
                data = {"prompt": system_prompt or ""}
                resp = requests.post(url, headers=headers, files=files, data=data, timeout=getattr(cfg, "timeout", 30))
                resp.raise_for_status()
                data = resp.json()
                # Expecting {'text': '...', 'usage': {...}} from the service
                text = data.get("text")
                usage = data.get("usage")
                return text, usage
            except Exception as e:  # pragma: no cover - network failure
                raise RuntimeError(f"OpenAIProvider.call_image HTTP fallback failed: {e}") from e

        raise RuntimeError("OpenAIProvider.call_image not available: no SDK client and no base_url configured")

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
