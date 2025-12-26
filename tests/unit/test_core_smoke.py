from ai_image_analyzer import (
    load_config_from_env,
    read_prompt_file,
    resize_image_bytes,
    make_collage,
    build_collage_system_prompt,
)
import os


def test_load_config_from_env_basic():
    cfg = load_config_from_env({"OPENAI_MAX_TOKENS": "1234"})
    assert cfg.max_tokens == 1234


def test_read_prompt_file_missing(tmp_path, monkeypatch):
    p = tmp_path / "x.txt"
    s = read_prompt_file(str(p))
    assert s == ""
