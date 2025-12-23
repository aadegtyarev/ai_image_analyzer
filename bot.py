from aiogram.exceptions import TelegramBadRequest
#!/usr/bin/env python3
import asyncio
import sys
import os
import json
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from uuid import uuid4

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, Router
from aiogram.types import Message, FSInputFile, BotCommand
from aiogram.enums import ChatType
from aiogram.client.default import DefaultBotProperties

from ai_image_analyzer import (
    load_config,
    call_model_with_image,
    call_model_with_text_only,
    build_collage_system_prompt,
    read_prompt_file,
    make_collage,
    check_balance,
)
from typing import Dict

load_dotenv()

# --- ENV / paths ---

BOT_TOKEN = os.getenv("BOT_TOKEN")
BOT_ADMIN_ID = int(os.getenv("BOT_ADMIN_ID", "0"))
BOT_ADMIN_USERNAME = os.getenv("BOT_ADMIN_USERNAME")

PROMPTS_DIR = os.getenv("PROMPTS_DIR", "prompts")
HOWTO_DIR = os.getenv("HOWTO_DIR", "howto")
USERS_FILE = os.getenv("USERS_FILE", "db/users.json")

PER_IMAGE_DEFAULT = os.getenv("PER_IMAGE_DEFAULT", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set in environment/.env")

# --- aiogram wiring ---

bot = Bot(BOT_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)

# --- users db & stats ---

def _ensure_users_file_dir() -> None:
    d = os.path.dirname(USERS_FILE)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def load_users() -> Dict[str, Any]:
    _ensure_users_file_dir()
    if not os.path.exists(USERS_FILE):
        data = {"enabled": [], "stats": {}, "meta": {}}
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("enabled", [])
    data.setdefault("stats", {})
    data.setdefault("meta", {})
    return data


def save_users(data: Dict[str, Any]) -> None:
    _ensure_users_file_dir()
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


users_db: Dict[str, Any] = load_users()


def is_admin(user_id: int) -> bool:
    return user_id == BOT_ADMIN_ID


def is_allowed(user_id: int) -> bool:
    if is_admin(user_id):
        return True
    return user_id in users_db.get("enabled", [])


def ensure_stats(uid: int) -> None:
    stats = users_db.setdefault("stats", {})
    key = str(uid)
    if key not in stats:
        stats[key] = {
            "requests": 0,
            "images": 0,
            "megabytes": 0.0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }


def update_stats_after_call(
    uid: int,
    images: int,
    bytes_sent: int,
    usage: Optional[dict],
) -> None:
    ensure_stats(uid)
    s = users_db["stats"][str(uid)]
    s["requests"] += 1
    s["images"] += images
    s["megabytes"] += bytes_sent / (1024.0 * 1024.0)
    if usage:
        s["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        try:
            s["total_cost"] += float(usage.get("total_cost", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
    save_users(users_db)


def set_user_meta(uid: int, description: str, username: str = "", full_name: str = ""):
    meta = users_db.setdefault("meta", {})
    meta[str(uid)] = {
        "description": description,
        "username": username,
        "full_name": full_name,
    }
    save_users(users_db)



# --- Markdown → HTML для Telegram ---
import re
FORMAT_MODE = 'HTML'  # 'HTML' для теста, None — plain text

def simple_markdown_to_html(md: str) -> str:
    def esc(s):
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
        )

    # --- 1. Вырезаем code и ссылки, заменяя на плейсхолдеры, до любых замен ---
    code_placeholders = []
    link_placeholders = []

    def code_repl(m):
        # Сохраняем исходный текст кода (без экранирования и замен!)
        code_placeholders.append(m.group(1))
        return f"{{{{CODE{len(code_placeholders)-1}}}}}"

    def link_repl(m):
        # Сохраняем исходный текст ссылки (без экранирования и замен!)
        link_placeholders.append((m.group(1), m.group(2)))
        return f"{{{{LINK{len(link_placeholders)-1}}}}}"

    # Сначала вырезаем code и ссылки из исходного md
    md_wo_code = re.sub(r"`([^`]+?)`", code_repl, md)
    md_wo_code_links = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_repl, md_wo_code)

    html = esc(md_wo_code_links)

    # --- 2. Применяем остальную разметку ---
    # Заголовки # ... => <b>...</b>
    html = re.sub(r"^# (.+)$", r"<b>\1</b>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<b>\1</b>", html, flags=re.MULTILINE)
    html = re.sub(r"^### (.+)$", r"<b>\1</b>", html, flags=re.MULTILINE)
    # Жирный **...** или __...__ — только не внутри слова
    html = re.sub(r"(?<!\w)\*\*(.+?)\*\*(?!\w)", r"<b>\1</b>", html)
    html = re.sub(r"(?<!\w)__(.+?)__(?!\w)", r"<b>\1</b>", html)
    # Курсив *...* или _..._ — только не внутри слова
    html = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<i>\1</i>", html)
    html = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", html)
    # Маркированные списки
    html = re.sub(r"^\* (.+)$", r"• \1", html, flags=re.MULTILINE)

    # Нормализуем переносы строк: ограничиваем серии переносов и удаляем ведущие переносы
    html = re.sub(r'\n{3,}', '\n\n', html)
    html = re.sub(r'^\n+', '', html)
    # Убираем переносы перед HTML-тегами (например, "\n<code>")
    html = re.sub(r'\n+(<)', r'\1', html)

    # --- 3. Возвращаем плейсхолдеры обратно ---
    for i, (text, url) in enumerate(link_placeholders):
        # Внутри ссылок Telegram разрешает только текст, без вложенных тегов
        safe_text = esc(text)
        html = html.replace(f"{{{{LINK{i}}}}}", f'<a href="{url}">{safe_text}</a>')
    for i, code in enumerate(code_placeholders):
        # Внутри <code>...</code> запрещены вложенные теги и <br>
        safe_code = esc(code).replace('<br>', '\n')
        html = html.replace(f"{{{{CODE{i}}}}}", f'<code>{safe_code}</code>')

    # Убираем все <br>, стоящие сразу после первой строки (Telegram не любит <br> в начале)
    html = re.sub(r'^(.*?)(?:<br>)+', r'\1', html, count=1)

    return html

# --- централизованная отправка сообщений ---
async def send_response(
    msg: Message,
    text: str = None,
    file_path: str = None,
    filename_prefix: str = "response",
) -> None:
    """
    Унифицированная отправка текста или файла в Telegram.
    text: строка для отправки (если есть)
    file_path: путь к файлу для отправки (если есть)
    FORMAT_MODE: None — plain text, 'HTML' — markdown→html
    """
    if text:
        if FORMAT_MODE == 'HTML':
            html = simple_markdown_to_html(text)
            if len(html) <= 3800:
                try:
                    await msg.answer(html, parse_mode='HTML')
                    return
                except TelegramBadRequest as e:
                    if "can't parse entities" in str(e):
                        # Логируем ошибку парсера Telegram и показываем контекст байтов вокруг проблемы
                        err_text = str(e)
                        print(f"[TelegramBadRequest] can't parse entities: {err_text}", flush=True)
                        print(f"Исходный текст: {repr(text)}", flush=True)
                        print(f"HTML: {repr(html)}", flush=True)
                        m = re.search(r'byte offset (\d+)', err_text)
                        if m:
                            off = int(m.group(1))
                            b = html.encode('utf-8')
                            start = max(0, off - 40)
                            end = min(len(b), off + 40)
                            context_bytes = b[start:end]
                            print(f"Problem byte offset: {off}", flush=True)
                            print(f"Context bytes (hex): {context_bytes.hex()}", flush=True)
                            print(f"Context (utf-8, replace): {context_bytes.decode('utf-8', 'replace')}", flush=True)
                        await msg.answer("❗ Ошибка парсера Telegram: не удалось отобразить форматированный текст. Попробуйте упростить оформление или отправить как plain text.")
                        return
                    # Логируем другие ошибки TelegramBadRequest
                    print(f"[TelegramBadRequest] {e}\nИсходный текст: {repr(text)}\nHTML: {repr(html)}", flush=True)
                    raise
            tmp_path = f"/tmp/{filename_prefix}_{msg.message_id}.html.txt"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(html)
            await msg.answer_document(FSInputFile(tmp_path))
            return
        else:
            if len(text) <= 3800:
                await msg.answer(text)
                return
            tmp_path = f"/tmp/{filename_prefix}_{msg.message_id}.txt"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(text)
            await msg.answer_document(FSInputFile(tmp_path))
            return
    if file_path:
        await msg.answer_document(FSInputFile(file_path))
        return
    await msg.answer("⚠ Пустое сообщение.")

# --- prompts & commands ---

@dataclass
class PromptInfo:
    command: str
    filename: str
    path: str
    description: str


RESERVED_COMMANDS = {
    "start",
    "help",
    "howto",
    "stats",
    "stats_all",
    "users",
    "balance",
}


def sanitize_command_name(base: str, used: set, idx: int) -> str:
    name = base.lower()
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"
    name = "".join(c if c in allowed else "_" for c in name)
    name = name.strip("_")

    if not name:
        name = f"p_{idx}"

    if not name[0].isalpha():
        name = f"p_{name}"

    if len(name) > 32:
        name = name[:32]

    orig = name
    suffix = 1
    while name in used or name in RESERVED_COMMANDS:
        candidate = f"{orig}_{suffix}"
        if len(candidate) > 32:
            candidate = candidate[:32]
        name = candidate
        suffix += 1

    used.add(name)
    return name


def load_prompts(prompts_dir: Optional[str] = None) -> Dict[str, PromptInfo]:
    prompts: Dict[str, PromptInfo] = {}
    dir_to_use = prompts_dir or os.getenv("PROMPTS_DIR", PROMPTS_DIR)
    if not os.path.isdir(dir_to_use):
        return prompts

    used_commands: set = set()
    idx = 1

    for fname in sorted(os.listdir(dir_to_use)):
        if not fname.lower().endswith(".txt"):
            continue
        base = os.path.splitext(fname)[0]
        cmd = sanitize_command_name(base, used_commands, idx)
        idx += 1

        path = os.path.join(dir_to_use, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
        except OSError:
            first_line = ""
        desc = first_line[:80] if first_line else f"Prompt from {fname}"
        prompts[cmd] = PromptInfo(
            command=cmd,
            filename=fname,
            path=path,
            description=desc,
        )
    return prompts


PROMPTS: Dict[str, PromptInfo] = load_prompts()
import time

# Temporary cache for media group prompts: media_group_id -> {system_prompt, prompt_label, use_text_override, user_text, ts}
MEDIA_CONTEXTS: Dict[str, dict] = {}
MEDIA_CONTEXT_TTL = int(os.getenv("MEDIA_CONTEXT_TTL", "120"))


def _cleanup_media_contexts() -> None:
    now = time.time()
    to_delete = [k for k, v in MEDIA_CONTEXTS.items() if now - v.get("ts", 0) > MEDIA_CONTEXT_TTL]
    for k in to_delete:
        MEDIA_CONTEXTS.pop(k, None)
        try:
            env_dbg = os.environ.get("DEBUG", None)
            if env_dbg is None:
                env_dbg = os.environ.get("IMAGE_DEBUG", "")
            media_debug = str(env_dbg).lower() in ("1", "true", "yes")
        except Exception:
            media_debug = False
        if media_debug:
            print(f"[MEDIA_DEBUG] expired media context {k}", file=sys.stderr)


def get_media_context(media_group_id: Optional[str]) -> Optional[dict]:
    if not media_group_id:
        return None
    _cleanup_media_contexts()
    return MEDIA_CONTEXTS.get(media_group_id)


def set_media_context(media_group_id: str, ctx: dict) -> None:
    ctx = dict(ctx)
    ctx["ts"] = time.time()
    MEDIA_CONTEXTS[media_group_id] = ctx
    try:
        env_dbg = os.environ.get("DEBUG", None)
        if env_dbg is None:
            env_dbg = os.environ.get("IMAGE_DEBUG", "")
        media_debug = str(env_dbg).lower() in ("1", "true", "yes")
    except Exception:
        media_debug = False
    if media_debug:
        print(f"[MEDIA_DEBUG] set media context {media_group_id}: prompt_label={ctx.get('prompt_label')}", file=sys.stderr)


async def setup_bot_commands() -> None:
    # очистить и зарегистрировать список команд
    await bot.set_my_commands([])

    cmds: List[BotCommand] = []

    cmds.append(BotCommand(command="text", description="Текст вместо промта (для фото)"))
        # note: 'group' flag can be passed as parameter to any command to request collage behaviour
    cmds.append(BotCommand(command="howto", description="Список howto-заметок"))
    cmds.append(BotCommand(command="stats", description="Статистика по запросам"))
    cmds.append(BotCommand(command="help", description="Справка и список промтов"))

    # Add dynamic prompt commands (from prompts/). Order follows filename sorting.
    for pi in PROMPTS.values():
        # ensure we don't exceed Telegram limits and avoid reserved names
        try:
            cmds.append(BotCommand(command=pi.command, description=pi.description[:80]))
        except Exception:
            # ignore any invalid commands
            pass

    await bot.set_my_commands(cmds[:100])


# dump_payload_to_file is implemented in debug_utils and reads DUMP_PAYLOADS/DUMP_DIR at runtime

# --- helpers ---

def extract_command_and_text(msg: Message) -> Tuple[Optional[str], str]:
    raw = (msg.text or msg.caption or "").strip()
    if not raw.startswith("/"):
        return None, raw
    first, *rest = raw.split(maxsplit=1)
    cmd = first[1:]
    if "@" in cmd:
        cmd = cmd.split("@", 1)[0]
    tail = rest[0].strip() if rest else ""
    return cmd, tail


def parse_tail_flags(t: str) -> Tuple[str, bool]:
    """Parse supported flags from the tail text.

    Currently supports the 'group' flag which requests collage mode.
    Returns (cleaned_tail, force_collage_bool)
    """
    parts = [p for p in (t or "").split() if p]
    flags = set(p.lower() for p in parts if p.lower() == "group")
    cleaned = " ".join(p for p in parts if p.lower() not in flags)
    return cleaned, ("group" in flags)


def normalize_command(cmd: Optional[str]) -> Optional[str]:
    if not cmd:
        return None
    c = cmd
    no_underscore = c.replace("_", "").lower()
    if no_underscore == "statsall":
        return "stats_all"
    if no_underscore == "statsreset":
        return "stats_reset"
    if no_underscore == "userdel":
        return "user_del"
    return c


def get_cfg():
    return load_config()





async def safe_error_reply(msg: Message, err: Exception) -> None:
    """
    Унифицированная обработка ошибок: логирует traceback, сообщает пользователю суть и совет.
    """
    traceback.print_exc()
    user_id = msg.from_user.id if msg.from_user else 0
    if is_admin(user_id):
        text = (
            f"❌ Внутренняя ошибка: {err}\n"
            f"\n"
            f"Traceback см. в логах.\n"
            f"Если ошибка повторяется — проверьте конфиг, переменные окружения, логи сервера."
        )
    else:
        text = (
            "❌ Произошла ошибка при обработке запроса. "
            "Пожалуйста, попробуйте ещё раз позже. "
            "Если ошибка повторяется — напишите администратору."
        )
    try:
        await send_response(msg, text)
    except Exception:
        traceback.print_exc()


async def send_howto_list(msg: Message) -> None:
    if not os.path.isdir(HOWTO_DIR):
        await send_response(msg, "📚 Папка howto пуста или недоступна.")
        return
    files = [
        f[:-3] for f in os.listdir(HOWTO_DIR) if f.lower().endswith(".md")
    ]
    if not files:
        await send_response(msg, "📚 Пока нет howto-заметок.")
        return
    lines = ["📚 Доступные howto:"]
    for name in sorted(files):
        lines.append(f"`/howto {name}`")
    await send_response(msg, "\n".join(lines))


async def send_howto_item(msg: Message, name: str) -> None:
    path = os.path.join(HOWTO_DIR, f"{name}.md")
    if not os.path.exists(path):
        await send_response(msg, "❌ Нет такого howto.")
        return
    with open(path, "r", encoding="utf-8") as f:
        body = f.read()
    if not body.strip():
        await send_response(msg, "⚠ Файл howto пуст.")
        return
    await send_response(msg, body, filename_prefix=f"howto_{name}")


async def handle_help(msg: Message) -> None:
    if not is_allowed(msg.from_user.id):
        return

    lines: List[str] = [
        "AI Photo Assistant",
        "",
        "📷 Отправь фото — получишь разбор.",
        "✍ Если вместе с фото написать текст, он заменит системный промт.",
        "",
        "🎯 Промты (файлы из папки prompts):",
    ]

    if PROMPTS:
        for cmd, p in sorted(PROMPTS.items()):
            desc = p.description or ""
            line = f"`/{cmd}` - {desc}"
            lines.append(line)
    else:
        lines.append("(папка PROMPTS_DIR пуста)")

    lines.extend(
        [
            "",
            "🛠 Режимы:",
            "`/text` – использовать текст сообщения как запрос (без системного промта).",
            "`/text_collage` – то же, но несколько фото собираются в коллаж.",
            "/howto – список howto-заметок.",
            "/stats – твоя личная статистика.",
            "/help – краткая справка.",
        ]
    )

    if is_admin(msg.from_user.id) and msg.chat.type == ChatType.PRIVATE:
        lines.extend(
            [
                "",
                "👑 Админ-команды (ручной ввод):",
                "/users – список разрешённых пользователей.",
                "`/user_add USER_ID` Описание – добавить пользователя.",
                "`/user_del USER_ID` – удалить пользователя.",
                "`/stats_reset USER_ID` – сброс статистики пользователя.",
                "/stats_all – общая статистика по всем.",
                "/balance – баланс API.",
            ]
        )

    await send_response(msg, "\n".join(lines))


async def extract_images_from_message(message: Message) -> List[bytes]:
    """Скачать все подходящие картинки из сообщения (photo, document image/*)."""
    res: List[bytes] = []
    if message.photo:
        largest = message.photo[-1]
        file = await bot.get_file(largest.file_id)
        b = await bot.download_file(file.file_path)
        res.append(b.read())
    if (
        message.document
        and message.document.mime_type
        and message.document.mime_type.startswith("image/")
    ):
        file = await bot.get_file(message.document.file_id)
        b = await bot.download_file(file.file_path)
        res.append(b.read())
    return res

# --- main handler ---

@router.message()
async def main_handler(msg: Message) -> None:
    try:
        user_id = msg.from_user.id if msg.from_user else 0
        cmd, tail = extract_command_and_text(msg)
        cmd = normalize_command(cmd)

        if not is_allowed(user_id):
            return

        # --- сервис / админ ---

        if cmd == "help":
            await handle_help(msg)
            return

        if cmd == "howto":
            if not tail:
                await send_howto_list(msg)
            else:
                await send_howto_item(msg, tail)
            return

        if cmd == "users":
            if not is_admin(user_id):
                return
            enabled = users_db.get("enabled", [])
            meta = users_db.get("meta", {})
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
            await send_response(msg, text)
            return

        if cmd == "user_add":
            if not is_admin(user_id):
                return
            if not tail:
                await msg.answer(
                    "Использование: `/user_add USER_ID` Описание",                    
                )
                return

            parts = tail.split(maxsplit=2)
            if len(parts) < 2:
                await msg.answer(
                    "Нужно указать и ID, и описание.\n"
                    "Пример: `/user_add 7045549272 мой коллега`",                    
                )
                return

            first = parts[0]
            if first.startswith("@"):
                await msg.answer(
                    "По @username бот не может получить ID пользователя. "
                    "Нужен числовой ID.\n\n"
                    "Пример: `/user_add 7045549272 мой коллега`",                    
                )
                return

            try:
                uid = int(first)
            except ValueError:
                await msg.answer(
                    "Укажи числовой ID пользователя.\n"
                    "Пример: `/user_add 7045549272 мой коллега`",                    
                )
                return

            if len(parts) == 2:
                description = parts[1]
            else:
                description = parts[1] + " " + parts[2]

            username = ""
            full_name = ""
            try:
                chat = await bot.get_chat(uid)
                username = chat.username or ""
                full_name = " ".join(
                    [p for p in [chat.first_name, chat.last_name] if p]
                )
            except Exception:
                pass

            enabled = users_db.setdefault("enabled", [])
            if uid not in enabled:
                enabled.append(uid)
            set_user_meta(uid, description=description, username=username, full_name=full_name)
            await send_response(msg, f"✅ Пользователь {uid} добавлен в список.")
            return

        if cmd == "user_del":
            if not is_admin(user_id):
                return
            if not tail:
                await msg.answer(
                    "Использование: `/user_del USER_ID`",                    
                )
                return
            try:
                uid = int(tail.split()[0])
            except ValueError:
                await msg.answer("Укажи числовой ID пользователя.", parse_mode=None)
                return
            enabled = users_db.setdefault("enabled", [])
            if uid in enabled:
                enabled.remove(uid)
            users_db.setdefault("meta", {}).pop(str(uid), None)
            save_users(users_db)
            await send_response(msg, f"❌ Пользователь {uid} удалён из списка.")
            return

        if cmd == "reload_prompts":
            if not is_admin(user_id):
                return
            try:
                # reload prompts and re-register commands
                global PROMPTS
                PROMPTS = load_prompts()
                await setup_bot_commands()
                await send_response(msg, "✅ Prompts reloaded.")
                print("[ADMIN] PROMPTS reloaded", file=sys.stderr)
            except Exception as e:
                await send_response(msg, f"Failed to reload prompts: {e}")
            return

        if cmd == "stats":
            ensure_stats(user_id)
            s = users_db["stats"][str(user_id)]
            txt = (
                "📊 *Твоя статистика*\n\n"
                f"Запросов: *{s['requests']}*\n"
                f"Файлов: *{s['images']}*\n"
                f"Объём: *{s['megabytes']:.2f} MB*\n"
                f"Токены: *{s['total_tokens']}*\n"
                f"Стоимость: *{s['total_cost']:.3f}* у.е.\n"
            )
            await send_response(msg, txt)
            return

        if cmd == "stats_reset":
            if not is_admin(user_id):
                return
            if not tail:
                await msg.answer(
                    "Использование: `/stats_reset USER_ID`",                    
                )
                return
            uid = tail.split()[0]
            ensure_stats(int(uid))
            users_db["stats"][uid] = {
                "requests": 0,
                "images": 0,
                "megabytes": 0.0,
                "total_tokens": 0,
                "total_cost": 0.0,
            }
            save_users(users_db)
            await send_response(msg, f"🧹 Статистика пользователя {uid} сброшена.")
            return

        if cmd == "stats_all":
            if not is_admin(user_id):
                return
            stats = users_db.get("stats", {})
            meta = users_db.get("meta", {})
            if not stats:
                await send_response(msg, "📊 Статистика пока пуста.")
                return
            total_req = total_img = total_tok = 0
            total_mb = total_cost = 0.0
            lines = ["📊 Общая статистика", ""]
            for uid, s in stats.items():
                r = s.get("requests", 0)
                i = s.get("images", 0)
                mb = s.get("megabytes", 0.0)
                tok = s.get("total_tokens", 0)
                cost = s.get("total_cost", 0.0)
                meta_info = meta.get(uid, {})
                desc = meta_info.get("description") or ""
                label = uid
                if desc:
                    label = f"{uid} ({desc})"
                total_req += r
                total_img += i
                total_mb += mb
                total_tok += tok
                total_cost += cost
                lines.append(
                    f"{label}: запросы {r}, файлы {i}, токены {tok}, "
                    f"объём {mb:.2f} MB, стоимость {cost:.3f} у.е."
                )
            lines.append(
                f"Итого: запросы {total_req}, файлы {total_img}, "
                f"объём {total_mb:.2f} MB, токены {total_tok}, "
                f"стоимость {total_cost:.3f} у.е."
            )
            await send_response(msg, "\n\n".join(lines))
            return

        if cmd == "balance":
            if not is_admin(user_id):
                return
            try:
                cfg = get_cfg()
                data = check_balance(cfg, quiet=True) or {}
                d = data.get("data", {})
                try:
                    credits = float(d.get("credits", 0.0))
                except (TypeError, ValueError):
                    credits = 0.0
                sub_status = d.get("subscription_status", "")
                sub_end = d.get("subscription_end", "")
                user_status_text = d.get("user_status_text", "")

                text = (
                    "💳 Баланс API\n\n"
                    f"Кредиты: {credits:.3f}\n"
                    f"Статус подписки: {sub_status}\n"
                    f"Подписка до: {sub_end}\n"
                )
                if user_status_text:
                    text += f"Комментарий: {user_status_text}\n"
                await msg.answer(text, parse_mode=None)
            except Exception as e:
                await safe_error_reply(msg, e)
            return

        # --- аналитика ---

        cfg = get_cfg()

        # собираем изображения: сначала из самого сообщения,
        # если нет — из reply_to_message
        images_bytes: List[bytes] = []
        images_bytes.extend(await extract_images_from_message(msg))
        if not images_bytes and msg.reply_to_message:
            images_bytes.extend(await extract_images_from_message(msg.reply_to_message))

        text_after_cmd = tail
        # user_text holds override text when provided; initialize to avoid UnboundLocalError
        user_text = None
        # parse flags from tail (e.g., 'group' to force collage)
        text_after_cmd, flag_group = parse_tail_flags(text_after_cmd or "")
        # per-request short ID to correlate logs
        request_id = uuid4().hex[:8]
        # Prompt debug flag: unified DEBUG, fallback to IMAGE_DEBUG
        try:
            env_dbg = os.environ.get("DEBUG", None)
            if env_dbg is None:
                env_dbg = os.environ.get("IMAGE_DEBUG", "")
            prompt_debug = str(env_dbg).lower() in ("1", "true", "yes")
        except Exception:
            prompt_debug = False

        use_text_override = False
        force_collage = False
        prompt_path: Optional[str] = None

        if cmd == "text":
            use_text_override = True
            # force_collage if 'group' flag passed
            if flag_group:
                force_collage = True
        elif cmd in PROMPTS:
            prompt_path = PROMPTS[cmd].path
            # if there is remaining text, use it to override prompt; also check for group flag
            if text_after_cmd:
                use_text_override = True
            if flag_group:
                force_collage = True
        else:
            if text_after_cmd:
                use_text_override = True

        # только текст
        if not images_bytes:
            if not text_after_cmd:
                await send_response(msg, "Нет ни изображений, ни текста.")
                return
            await send_response(msg, "💭 Думаю над текстом...")
            resp = call_model_with_text_only(
                cfg,
                text_after_cmd,
                system_prompt="",
                quiet=True,
            )
            if isinstance(resp, tuple):
                text_result, usage = resp
            else:
                text_result, usage = resp, None
            update_stats_after_call(
                user_id,
                images=0,
                bytes_sent=0,
                usage=usage,
            )
            total_cost = 0.0
            if usage:
                try:
                    total_cost = float(usage.get("total_cost", 0.0) or 0.0)
                except (TypeError, ValueError):
                    total_cost = 0.0
            final = text_result
            if total_cost > 0:
                final += f"\n\n💎 {total_cost:.3f} у.е."
            await send_response(msg, final, filename_prefix="text")
            return

        # есть изображения — сначала ресайз
        from PIL import Image, ImageOps
        import io as _io

        def resize_bytes(data: bytes, max_size: int, quality: int) -> tuple[bytes, int, int, str]:
            # Учитываем EXIF-ориентацию и возвращаем байты + финальные размеры + ориентацию
            with Image.open(_io.BytesIO(data)) as im:
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                w, h = im.size
                if h > w:
                    orientation = "portrait"
                elif w > h:
                    orientation = "landscape"
                else:
                    orientation = "square"
                scale = min(1.0, float(max_size) / max(w, h))
                if scale < 1.0:
                    new_size = (int(w * scale), int(h * scale))
                    im = im.resize(new_size, Image.LANCZOS)
                final_w, final_h = im.size
                # Первый вариант: попытка сохранить с optimize=True
                buf = _io.BytesIO()
                try:
                    im.save(buf, format="JPEG", quality=quality, optimize=True)
                except Exception:
                    # Некоторые комбинации параметров/образов могут провоцировать ошибки оптимизации;
                    # попробуем без optimize
                    buf = _io.BytesIO()
                    im.save(buf, format="JPEG", quality=min(quality, 95), optimize=False)
                out = buf.getvalue()
                # Защитный fallback: если по какой-то причине байтов нет — попробуем ещё раз без optimize
                if not out:
                    buf = _io.BytesIO()
                    im.save(buf, format="JPEG", quality=min(quality, 85), optimize=False)
                    out = buf.getvalue()
                if not out:
                    raise RuntimeError("resize_bytes: resulted in empty JPEG bytes")
                return out, final_w, final_h, orientation

        resized: List[tuple[bytes, int, int, str]] = [
            resize_bytes(b, cfg.image_max_size, cfg.image_quality)
            for b in images_bytes
        ]

        # Debug output about resized images
        try:
            env_dbg = os.environ.get("DEBUG", None)
            if env_dbg is None:
                env_dbg = os.environ.get("IMAGE_DEBUG", "")
            image_debug = str(env_dbg).lower() in ("1", "true", "yes")
        except Exception:
            image_debug = False
        if image_debug:
            for idx, (b, w, h, o) in enumerate(resized, start=1):
                print(f"[IMAGE_DEBUG] image #{idx}: orientation={o}, size={w}x{h}, bytes={len(b)}", file=sys.stderr)

        # выясняем системный промт / его имя
        # Check for media group context first: if message is part of a media group
        mgid = getattr(msg, "media_group_id", None)
        cached = get_media_context(mgid) if mgid else None
        if cached:
            # reuse cached prompt info for this media group
            system_prompt = cached.get("system_prompt", "")
            prompt_label = cached.get("prompt_label", "без промта")
            use_text_override = cached.get("use_text_override", False)
            user_text = cached.get("user_text")
            if prompt_debug:
                print(f"[PROMPT_DEBUG][{request_id}] using cached media prompt for media_group_id={mgid}: prompt_label={prompt_label}", file=sys.stderr)
        else:
            if use_text_override:
                user_text = text_after_cmd
                system_prompt = ""
                prompt_label = "текст из сообщения"
                if prompt_debug:
                    print(f"[PROMPT_DEBUG][{request_id}] use_text_override -> user_text={user_text!r}", file=sys.stderr)
            elif prompt_path:
                system_prompt = read_prompt_file(prompt_path)
                prompt_label = os.path.splitext(os.path.basename(prompt_path))[0]
                if prompt_debug:
                    print(f"[PROMPT_DEBUG][{request_id}] prompt_path={prompt_path!r}, prompt_label={prompt_label!r}, system_snip={system_prompt[:200]!r}", file=sys.stderr)
                # If this message is part of a media group, cache the prompt info so subsequent messages reuse it
                if mgid:
                    set_media_context(mgid, {"system_prompt": system_prompt, "prompt_label": prompt_label, "use_text_override": use_text_override, "user_text": user_text})
            else:
                if not cfg.prompt_file:
                    system_prompt = ""
                    prompt_label = "без промта"
                    if prompt_debug:
                        print(f"[PROMPT_DEBUG][{request_id}] no system prompt configured", file=sys.stderr)
                else:
                    system_prompt = read_prompt_file(cfg.prompt_file)
                    prompt_label = os.path.splitext(os.path.basename(cfg.prompt_file))[0]
                    if prompt_debug:
                        print(f"[PROMPT_DEBUG][{request_id}] cfg.prompt_file={cfg.prompt_file!r}, prompt_label={prompt_label!r}, system_snip={system_prompt[:200]!r}", file=sys.stderr)
                    if mgid:
                        set_media_context(mgid, {"system_prompt": system_prompt, "prompt_label": prompt_label, "use_text_override": use_text_override, "user_text": user_text})

                # If debug enabled, print a short sanitized payload summary to stderr for easier tracing
                try:
                    if prompt_debug:
                        sp = {
                            "request_id": request_id,
                            "prompt_label": prompt_label,
                            "system_snip": (system_prompt or "")[:200],
                            "user_snip": (text_after_cmd or "")[:200],
                            "images": [{"idx": i + 1, "len": len(b)} for i, b in enumerate(images_bytes)],
                        }
                        print(f"[PROMPT_DEBUG][{request_id}] payload_summary: {sp}", file=sys.stderr)
                except Exception:
                    print(f"[PROMPT_DEBUG][{request_id}] failed to build payload summary", file=sys.stderr)

        # статусное сообщение (без Markdown, чтобы промт с _ не ломал парсер)
        def fmt_mb(n_bytes: int) -> str:
            return f"{n_bytes / (1024.0 * 1024.0):.2f} MB"

        orig_total = sum(len(b) for b in images_bytes)
        # resized is list of tuples (bytes, w, h, orientation)
        resized_total = sum(len(b) for b, *_ in resized)
        # If resized_total unexpectedly zero while orig_total > 0, log diagnostics
        if resized_total == 0 and orig_total > 0:
            print("[IMAGE_DEBUG] Warning: resized_total==0 while orig_total>0", file=sys.stderr)
            for idx, item in enumerate(resized, start=1):
                try:
                    b = item[0]
                    print(f"[IMAGE_DEBUG] resized #{idx}: type={type(b)!r}, len={len(b)}", file=sys.stderr)
                except Exception as e:
                    print(f"[IMAGE_DEBUG] resized #{idx}: failed to inspect: {e}", file=sys.stderr)
        n_files = len(resized)
        files_word = "файл" if n_files == 1 else "файла" if n_files < 5 else "файлов"

        status_suffix = ""
        if resized_total == 0 and orig_total > 0:
            status_suffix = " ⚠️ (ошибка при ресайзе; смотрите логи)"

        await send_response(
            msg,
            f"📷 Взял в работу {n_files} {files_word}. "
            f"Размер {fmt_mb(orig_total)} → {fmt_mb(resized_total)},{status_suffix} "
            f"промт {prompt_label}."
        )

        multiple = len(resized) > 1
        per_image = PER_IMAGE_DEFAULT
        if force_collage:
            per_image = False

        total_bytes = resized_total
        aggregated_texts: List[str] = []
        total_cost_request = 0.0

        if multiple and not per_image:
            named = [(f"image_{i+1}.jpg", b) for i, (b, w, h, o) in enumerate(resized)]
            collage_bytes, file_names = make_collage(
                named,
                cfg.collage_max_size,
                cfg.collage_quality,
            )
            if use_text_override:
                collage_system_prompt = ""
                user_text_for_call = user_text
            else:
                collage_system_prompt = build_collage_system_prompt(
                    system_prompt,
                    file_names,
                )
                user_text_for_call = None

            collage_meta = {"mode": "collage", "orientations": {f: o for f, o in zip(file_names, [orient for _, _, orient, _ in resized])}}
            if image_debug:
                print(f"[IMAGE_DEBUG] calling model for collage; files={file_names}; meta={collage_meta}; collage_bytes={len(collage_bytes)}", file=sys.stderr)
            if prompt_debug:
                print(f"[PROMPT_DEBUG][{request_id}] calling model for COLLAGE; prompt_label={prompt_label!r}, system_snip={collage_system_prompt[:200]!r}, user_text_snip={user_text_for_call[:200]!r}", file=sys.stderr)
            resp = call_model_with_image(
                cfg,
                collage_bytes,
                system_prompt=collage_system_prompt,
                user_text=user_text_for_call,
                quiet=True,
                image_meta=collage_meta,
            )
            if isinstance(resp, tuple):
                text_result, usage = resp
            else:
                text_result, usage = resp, None

            update_stats_after_call(
                user_id,
                images=len(resized),
                bytes_sent=total_bytes,
                usage=usage,
            )

            if usage:
                try:
                    total_cost_request += float(usage.get("total_cost", 0.0) or 0.0)
                except (TypeError, ValueError):
                    pass

            header = f"Коллаж — промт: {prompt_label}\n"
            aggregated_texts.append(header + text_result)
        else:
            for i, (jpeg, final_w, final_h, orientation) in enumerate(resized, start=1):
                if use_text_override:
                    system_prompt_for_call = ""
                    user_text_for_call = user_text
                else:
                    system_prompt_for_call = system_prompt
                    user_text_for_call = None
                image_meta = {"orientation": orientation, "width": final_w, "height": final_h}
                if image_debug:
                    print(f"[IMAGE_DEBUG] calling model for image #{i}; meta={image_meta}; bytes={len(jpeg)}", file=sys.stderr)
                if prompt_debug:
                    print(f"[PROMPT_DEBUG][{request_id}] calling model for image #{i}; prompt_label={prompt_label!r}, system_snip={(system_prompt_for_call or '')[:200]!r}, user_text_snip={str(user_text_for_call or '')[:200]!r}", file=sys.stderr)
                resp = call_model_with_image(
                    cfg,
                    jpeg,
                    system_prompt=system_prompt_for_call,
                    user_text=user_text_for_call,
                    quiet=True,
                    image_meta=image_meta,
                )
                if isinstance(resp, tuple):
                    text_result, usage = resp
                else:
                    text_result, usage = resp, None

                update_stats_after_call(
                    user_id,
                    images=1,
                    bytes_sent=len(jpeg),
                    usage=usage,
                )

                if usage:
                    try:
                        total_cost_request += float(
                            usage.get("total_cost", 0.0) or 0.0
                        )
                    except (TypeError, ValueError):
                        pass

                header = f"Изображение #{i} — промт: {prompt_label}\n"
                aggregated_texts.append(header + text_result)

        final_text = "\n\n".join(aggregated_texts)
        if total_cost_request > 0:
            final_text += f"\n\n💎 {total_cost_request:.3f} у.е."

        await send_response(msg, final_text, filename_prefix="images")

    except Exception as e:
        await safe_error_reply(msg, e)


async def main() -> None:
    await setup_bot_commands()
    print("Bot is running...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
