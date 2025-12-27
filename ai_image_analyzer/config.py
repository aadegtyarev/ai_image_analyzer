from dataclasses import dataclass
from typing import Optional
import os

try:
    from dotenv import load_dotenv, find_dotenv
except Exception:  # pragma: no cover - dotenv not installed in minimal env
    def load_dotenv(*a, **k):
        return False
    def find_dotenv(*a, **k):
        return None


@dataclass
class Settings:
    api_key: str
    base_url: Optional[str]
    model: str
    timeout: int
    max_tokens: int
    image_max_size: int
    image_quality: int
    collage_max_size: int
    collage_quality: int
    prompt_file: Optional[str]
    balance_threshold: Optional[int] = None
    image_debug: bool = False


def load_config() -> Settings:
    # Try find_dotenv(); if it fails to locate a file, search parent directories
    _env = find_dotenv()
    if not _env:
        # Walk up from current working directory to look for .env
        p = os.path.abspath(os.getcwd())
        found = None
        while True:
            cand = os.path.join(p, ".env")
            if os.path.exists(cand):
                found = cand
                break
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
        if not found:
            # Also try relative to this package file (handles being imported from subfolders)
            p = os.path.abspath(os.path.dirname(__file__))
            while True:
                cand = os.path.join(p, ".env")
                if os.path.exists(cand):
                    found = cand
                    break
                parent = os.path.dirname(p)
                if parent == p:
                    break
                p = parent
        _env = found or ".env"
    load_dotenv(_env)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    def _int_env(name: str, default: int) -> int:
        v = os.environ.get(name)
        if not v:
            return default
        try:
            return int(v)
        except Exception:
            return default

    return Settings(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        model=os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini",
        timeout=_int_env("OPENAI_TIMEOUT", 120),
        max_tokens=_int_env("OPENAI_MAX_TOKENS", 1024),
        image_max_size=_int_env("IMAGE_MAX_SIZE", 1024),
        image_quality=_int_env("IMAGE_QUALITY", 90),
        collage_max_size=_int_env("COLLAGE_MAX_SIZE", 2048),
        collage_quality=_int_env("COLLAGE_QUALITY", 90),
        prompt_file=os.environ.get("PROMPT_FILE"),
        balance_threshold=_int_env("OPENAI_BALANCE_THRESHOLD", 0) or None,
        image_debug=str(os.environ.get("DEBUG", os.environ.get("IMAGE_DEBUG", ""))).lower() in ("1", "true", "yes"),
    )


def load_config_from_env(env: dict):
    """Backward-compatible helper used in tests."""
    class SimpleCfg:
        pass

    cfg = SimpleCfg()
    try:
        context_size = int(env.get("OPENAI_CONTEXT_SIZE", 2048))
    except Exception:
        context_size = 2048
    try:
        max_tokens = int(env.get("OPENAI_MAX_TOKENS", 1024))
    except Exception:
        max_tokens = 1024
    if max_tokens > max(0, context_size - 1):
        max_tokens = max(0, context_size - 1)
    cfg.context_size = context_size
    cfg.max_tokens = max_tokens
    cfg.x_title = env.get("OPENAI_X_TITLE")
    return cfg


def read_prompt_file(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        # Keep message expected by existing tests
        print(f"ERROR: failed to read PROMPT_FILE {path}: {e}", file=os.sys.stderr)
        return ""
