import os
import tempfile
import shutil

from bot import load_prompts


def test_load_prompts_and_reload(monkeypatch, tmp_path):
    # create temp prompts dir
    pdir = tmp_path / "prompts"
    pdir.mkdir()
    (pdir / "a.txt").write_text("First prompt\nline2")

    import os
    os.environ["PROMPTS_DIR"] = str(pdir)
    prompts = load_prompts()
    assert any(pi.filename == "a.txt" for pi in prompts.values())
    # add another file and reload
    (pdir / "b.txt").write_text("Second prompt")
    prompts2 = load_prompts(str(pdir))
    assert any(pi.filename == "b.txt" for pi in prompts2.values())


# Note: dump to disk removed; we only test prompt loading now.
