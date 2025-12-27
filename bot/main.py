import asyncio
from aiogram import Bot, Dispatcher, Router
from .config import BOT_TOKEN
import bot as pkg
from ai_image_analyzer import load_config


def setup_config() -> None:
    """Load configuration via `load_config()` and attach it to `bot` package.

    This makes the configuration available as `bot.cfg` to handlers and
    background tasks without having to call `load_config()` everywhere.
    """
    try:
        cfg = load_config()
        pkg.cfg = cfg
        # Ensure BOT_TOKEN from .env is visible on the package for runtime checks
        try:
            pkg.BOT_TOKEN = __import__('os').environ.get('BOT_TOKEN')
        except Exception:
            pkg.BOT_TOKEN = None
        # If BOT_TOKEN wasn't set via load_dotenv (malformed .env lines), try a tolerant parse of the .env file
        if not pkg.BOT_TOKEN:
            try:
                from ai_image_analyzer.config import find_dotenv
                import re

                env_path = find_dotenv() or ".env"
                try:
                    with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
                        data = f.read()
                        m = re.search(r"\bBOT_TOKEN\s*=\s*([\w:\-\.]+)", data)
                        if m:
                            t = m.group(1).strip()
                            __import__("os").environ["BOT_TOKEN"] = t
                            pkg.BOT_TOKEN = t
                except Exception:
                    pass
            except Exception:
                pass
        return cfg
    except Exception:
        # Do not raise here; callers (start) will handle and report errors.
        pkg.cfg = None
        return None


async def run() -> None:
    # Load and attach configuration for the rest of the bot modules
    setup_config()

    # Read BOT_TOKEN from package/env after loading .env
    bot_token = getattr(pkg, "BOT_TOKEN", None) or __import__("os").environ.get("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("BOT_TOKEN is not set")

    # Initialize real Bot instance and attach to package for compatibility
    b = Bot(bot_token)
    pkg.bot = b

    dp = Dispatcher()
    router = Router()
    dp.include_router(router)

    # register commands and notify admin
    try:
        await pkg.setup_bot_commands()
    except Exception:
        pass
    try:
        await pkg.notify_admin_startup()
    except Exception:
        pass

    # Start polling (blocking)
    await dp.start_polling(b)


def start() -> None:
    asyncio.run(run())
