import subprocess
import sys
import os


def test_bot_runner_smoke(tmp_path):
    # Run the root bot.py as a subprocess and expect it to exit with a friendly error
    env = os.environ.copy()
    env.pop("BOT_TOKEN", None)
    res = subprocess.run([sys.executable, "bot.py"], cwd=str(tmp_path.parent), env=env, capture_output=True, text=True)
    out = (res.stdout or "") + (res.stderr or "")
    assert res.returncode != 0
    assert "BOT_TOKEN is not set" in out
