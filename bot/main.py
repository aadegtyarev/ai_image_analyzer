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
        return cfg
    except Exception:
        # Do not raise here; callers (start) will handle and report errors.
        pkg.cfg = None
        return None


async def run() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not set")

    # Load and attach configuration for the rest of the bot modules
    setup_config()

    # Initialize real Bot instance and attach to package for compatibility
    b = Bot(BOT_TOKEN)
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
