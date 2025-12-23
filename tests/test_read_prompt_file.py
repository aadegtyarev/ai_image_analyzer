from ai_image_analyzer import read_prompt_file
import os


def test_read_prompt_file_missing(tmp_path, capsys):
    p = tmp_path / "not_exists.txt"
    # ensure file does not exist
    if p.exists():
        p.unlink()
    res = read_prompt_file(str(p))
    assert res == ""
    captured = capsys.readouterr()
    assert "ERROR: failed to read PROMPT_FILE" in captured.err


def test_read_prompt_file_empty_path():
    assert read_prompt_file(None) == ""
