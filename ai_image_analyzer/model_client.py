from typing import Optional, Tuple, Any

def call_model_with_text_only(cfg, text, system_prompt=None, quiet=False) -> Any:
    """Placeholder for model text calls.

    Tests should monkeypatch this function or higher-level providers.
    """
    raise RuntimeError("Model provider not configured. Implement a provider or monkeypatch this function in tests.")


def call_model_with_image(cfg, jpeg_bytes, system_prompt=None, user_text=None, quiet=False, image_meta=None) -> Any:
    """Placeholder for model image calls.

    Return value should be either string or (string, usage_dict).
    """
    raise RuntimeError("Model provider not configured. Implement a provider or monkeypatch this function in tests.")


def check_balance(cfg, quiet: bool = False) -> dict:
    """Check billing endpoint (requires cfg.base_url and cfg.api_key).

    This helper will perform an HTTP GET to `<base_url>/balance` and return parsed json.
    """
    import requests

    if not getattr(cfg, "base_url", None):
        raise RuntimeError("OPENAI_BASE_URL is not set; cannot check balance.")
    url = cfg.base_url.rstrip("/") + "/balance"
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    resp = requests.get(url, headers=headers, timeout=getattr(cfg, "timeout", 30))
    resp.raise_for_status()
    return resp.json()
