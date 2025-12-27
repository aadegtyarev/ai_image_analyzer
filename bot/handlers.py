import os
import sys
import traceback
from typing import Optional


async def send_howto_list(msg) -> None:
    dir_to_use = os.getenv("HOWTO_DIR", "howto")
    if not os.path.isdir(dir_to_use):
        await __import__('bot').send_response(msg, "📚 Папка howto пуста или недоступна.")
        return
    files = [f[:-3] for f in os.listdir(dir_to_use) if f.lower().endswith(".md")]
    if not files:
        await __import__('bot').send_response(msg, "📚 Пока нет howto-заметок.")
        return
    lines = ["📚 Доступные howto:"]
    for name in sorted(files):
        lines.append(f"`/howto {name}`")
    await __import__('bot').send_response(msg, "\n".join(lines))


async def send_howto_item(msg, name: str) -> None:
    path = os.path.join(os.getenv("HOWTO_DIR", "howto"), f"{name}.md")
    if not os.path.exists(path):
        await __import__('bot').send_response(msg, "❌ Нет такого howto.")
        return
    with open(path, "r", encoding="utf-8") as f:
        body = f.read()
    if not body.strip():
        await __import__('bot').send_response(msg, "⚠ Файл howto пуст.")
        return
    await __import__('bot').send_response(msg, body, filename_prefix=f"howto_{name}")


async def setup_bot_commands() -> None:
    # Build command list and call Bot API via patched bot object
    cmds = []
    from aiogram.types import BotCommand
    cmds.append(BotCommand(command="help", description="Справка и список промтов"))
    cmds.append(BotCommand(command="howto", description="Список howto-заметок"))
    cmds.append(BotCommand(command="text", description="Текст вместо промта (для фото)"))
    cmds.append(BotCommand(command="stats", description="Статистика по запросам"))
    # dynamic prompts
    PROMPTS = getattr(__import__('bot'), 'PROMPTS', {})
    for pi in PROMPTS.values():
        try:
            cmds.append(BotCommand(command=pi.command, description=pi.description[:80]))
        except Exception:
            pass
    try:
        await __import__('bot').bot.set_my_commands(cmds[:100])
    except Exception:
        pass


async def main_handler(msg) -> None:
    try:
        user_id = getattr(getattr(msg, 'from_user', None), 'id', 0)
        # parse command and tail
        cmd_raw = None
        raw = (getattr(msg, 'text', None) or getattr(msg, 'caption', None) or "").strip()
        if raw.startswith('/'):
            first, *rest = raw.split(maxsplit=1)
            cmd_raw = first[1:]
            if '@' in cmd_raw:
                cmd_raw = cmd_raw.split('@', 1)[0]
        tail = rest[0].strip() if rest else ""

        from .utils import normalize_command
        cmd = normalize_command(cmd_raw)

        if not __import__('bot').is_allowed(user_id, __import__('bot').users_db, __import__('bot').BOT_ADMIN_ID):
            return

        if cmd == "help":
            await __import__('bot').handle_help(msg)
            return

        if cmd == "howto":
            if not tail:
                await send_howto_list(msg)
            else:
                await send_howto_item(msg, tail)
            return

        # implement other handlers minimally by delegating to existing package functions
        # For admin commands, rely on is_admin helper
        if cmd == "users":
            if not __import__('bot').is_admin(user_id, __import__('bot').BOT_ADMIN_ID):
                return
            enabled = __import__('bot').users_db.get("enabled", [])
            meta = __import__('bot').users_db.get("meta", {})
            if enabled:
                lines = ["👥 Разрешённые пользователи:"]
                for uid in enabled:
                    info = meta.get(str(uid), {})
                    username = info.get("username") or ""
                    desc = info.get("description") or ""
                    line = str(uid)
                    if username:
                        line += f" @{username}"
                    if desc:
                        line += f" — {desc}"
                    lines.append(line)
                text = "\n".join(lines)
            else:
                text = "👥 Список пуст."
            await __import__('bot').send_response(msg, text)
            return

        # Other commands are intentionally left to original bot module or future implementation

    except Exception as e:
        traceback.print_exc()
        try:
            await __import__('bot').send_response(msg, "❌ Произошла ошибка при обработке запроса.")
        except Exception:
            pass
