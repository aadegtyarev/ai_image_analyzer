import os

from ai_image_analyzer import load_config_from_env


def test_context_capping():
    env = {
        "OPENAI_API_KEY": "x",
        "OPENAI_MODEL": "gpt-test",
        "OPENAI_MAX_TOKENS": "5000",
        "OPENAI_CONTEXT_SIZE": "2048",
    }
    cfg = load_config_from_env(env)
    assert cfg.context_size == 2048
    assert cfg.max_tokens == 2047  # capped to context_size - 1


def test_defaults():
    env = {"OPENAI_API_KEY": "x", "OPENAI_MODEL": "gpt-test"}
    cfg = load_config_from_env(env)
    assert cfg.context_size == 2048
    assert cfg.max_tokens == 1024


def test_analyze_json_missing_api_key():
    from ai_image_analyzer import handle_json_request
    import os
    # ensure no OPENAI_API_KEY in environment for this test
    os.environ.pop("OPENAI_API_KEY", None)
    req = {"action": "analyze", "images": []}
    resp = handle_json_request(req)
    assert resp["ok"] is False
    assert any("OPENAI_API_KEY is not set" in e for e in resp.get("errors", []))


def test_x_title_in_config():
    env = {
        "OPENAI_API_KEY": "x",
        "OPENAI_MODEL": "gpt-test",
        "OPENAI_X_TITLE": "my-script",
    }
    cfg = load_config_from_env(env)
    assert cfg.x_title == "my-script"


def test_analyze_json_include_billing(monkeypatch):
    from ai_image_analyzer import handle_json_request

    # monkeypatch the model call to return usage info
    def fake_text_only(cfg, text, system_prompt, quiet):
        return (
            "ok",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "total_cost": 0.001},
        )

    monkeypatch.setattr("ai_image_analyzer.call_model_with_text_only", fake_text_only)

    req = {"action": "analyze", "override_text": "x", "include_billing": True}
    resp = handle_json_request(req)
    assert resp["ok"] is True
    assert "usage" in resp and (resp.get("usage") or resp.get("results") is not None)
