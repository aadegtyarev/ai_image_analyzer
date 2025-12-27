import asyncio
from aiogram import Bot, Dispatcher, Router
from .config import BOT_TOKEN
import bot as pkg


async def run() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not set")

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
