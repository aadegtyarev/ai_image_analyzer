import os

# Bridge config used by bot; these are intentionally simple and map to existing env vars
BOT_TOKEN = os.getenv("BOT_TOKEN")

def _int_from_env(name: str, default: int = 0) -> int:
    val = os.getenv(name, "")
    if val is None:
        return default
    try:
        val = val.split("#", 1)[0].strip()
        return int(val) if val != "" else default
    except Exception:
        return default


BOT_ADMIN_ID = _int_from_env("BOT_ADMIN_ID", 0)
BOT_ADMIN_USERNAME = os.getenv("BOT_ADMIN_USERNAME")

PROMPTS_DIR = os.getenv("PROMPTS_DIR", "prompts")
HOWTO_DIR = os.getenv("HOWTO_DIR", "howto")
USERS_FILE = os.getenv("USERS_FILE", "db/users.json")

PER_IMAGE_DEFAULT = os.getenv("PER_IMAGE_DEFAULT", "true").lower() in ("1", "true", "yes", "on")
