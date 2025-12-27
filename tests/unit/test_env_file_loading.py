import os
import tempfile

from ai_image_analyzer import load_config


def test_load_config_reads_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=dot-env-key\nOPENAI_MODEL=test-model\nIMAGE_MAX_SIZE=512\n")

    # monkeypatch find_dotenv to return our tmp .env
    import ai_image_analyzer.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "find_dotenv", lambda *a, **k: str(env_file))

    cfg = load_config()
    assert cfg.api_key == "dot-env-key"
    assert cfg.model == "test-model"
    assert cfg.image_max_size == 512
