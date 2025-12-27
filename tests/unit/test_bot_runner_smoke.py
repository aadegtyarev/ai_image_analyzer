import subprocess
import sys
import os
from pathlib import Path


def test_bot_runner_smoke():
    # Run the root bot.py as a subprocess and expect it to exit with a friendly error
    env = os.environ.copy()
    env.pop("BOT_TOKEN", None)
    # Find repository root by walking up until we find bot.py
    p = Path(__file__).resolve()
    repo_root = None
    while p != p.parent:
        if (p / "bot.py").exists():
            repo_root = p
            break
        p = p.parent
    assert repo_root is not None, "could not locate repository root"
    res = subprocess.run([sys.executable, str(repo_root / "bot.py")], cwd=str(repo_root), env=env, capture_output=True, text=True)
    out = (res.stdout or "") + (res.stderr or "")
    assert res.returncode != 0
    assert "BOT_TOKEN is not set" in out
