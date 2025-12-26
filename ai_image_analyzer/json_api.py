from typing import Optional
import os


def handle_json_request(req: dict) -> dict:
    # If there's no override_text, ensure API key exists
    if req.get("override_text") is None and not os.environ.get("OPENAI_API_KEY"):
        return {"ok": False, "errors": ["OPENAI_API_KEY is not set"]}

    action = req.get("action")
    if action != "analyze":
        return {"ok": False, "errors": ["unsupported action"]}

    override_text = req.get("override_text")
    include_billing = bool(req.get("include_billing", False))

    if override_text is not None:
        try:
            # Import from package so tests can monkeypatch call_model_with_text_only
            from ai_image_analyzer import call_model_with_text_only, load_config

            result = call_model_with_text_only(load_config(), override_text, system_prompt="", quiet=True)
        except Exception as e:
            return {"ok": False, "errors": [str(e)]}
        resp = {"ok": True, "results": result}
        if include_billing:
            resp["usage"] = None
        return resp

    return {"ok": False, "errors": ["no override_text provided"]}
