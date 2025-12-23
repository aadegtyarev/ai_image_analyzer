#!/usr/bin/env python3
import asyncio
import os
import json
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router
from aiogram.types import Message, FSInputFile, BotCommand
from aiogram.enums import ParseMode, ChatType
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

bot = Bot(
    BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
)
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

# --- markdown escape ---

def escape_md(text: str) -> str:
    """Простейшее экранирование под Telegram Markdown (v1)."""
    for ch in ("\\", "_", "*", "[", "]", "(", ")"):
        text = text.replace(ch, "\\" + ch)
    return text


def escape_md_smart(text: str) -> str:
    """Экранирование Markdown, пропуская части внутри `...`."""
    parts = text.split('`')
    for i in range(len(parts)):
        if i % 2 == 0:  # вне backticks
            parts[i] = escape_md(parts[i])
    return '`'.join(parts)

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


def load_prompts() -> Dict[str, PromptInfo]:
    prompts: Dict[str, PromptInfo] = {}
    if not os.path.isdir(PROMPTS_DIR):
        return prompts

    used_commands: set = set()
    idx = 1

    for fname in sorted(os.listdir(PROMPTS_DIR)):
        if not fname.lower().endswith(".txt"):
            continue
        base = os.path.splitext(fname)[0]
        cmd = sanitize_command_name(base, used_commands, idx)
        idx += 1

        path = os.path.join(PROMPTS_DIR, fname)
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


async def setup_bot_commands() -> None:
    # очистить и зарегистрировать список команд
    await bot.set_my_commands([])

    cmds: List[BotCommand] = []

    cmds.append(BotCommand(command="text", description="Текст вместо промта (для фото)"))
    cmds.append(
        BotCommand(
            command="text_collage",
            description="Текст + коллаж для нескольких фото",
        )
    )
    cmds.append(BotCommand(command="howto", description="Список howto-заметок"))
    cmds.append(BotCommand(command="stats", description="Статистика по запросам"))
    cmds.append(BotCommand(command="help", description="Справка и список промтов"))

    await bot.set_my_commands(cmds[:100])

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


async def send_text_or_file(
    msg: Message,
    text: str,
    filename_prefix: str = "response",
) -> None:
    if not text:
        await msg.answer("⚠ Пустое сообщение.", parse_mode=None)
        return
    if len(text) <= 3800:
        await msg.answer(text)
    else:
        tmp_path = f"/tmp/{filename_prefix}_{msg.message_id}.txt"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
        await msg.answer_document(FSInputFile(tmp_path))


async def safe_error_reply(msg: Message, err: Exception) -> None:
    traceback.print_exc()
    if msg.from_user and is_admin(msg.from_user.id):
        text = f"❌ Внутренняя ошибка: {err}"
    else:
        text = "❌ Что-то пошло не так при обработке запроса."
    try:
        await msg.answer(text, parse_mode=None)
    except Exception:
        traceback.print_exc()


async def send_howto_list(msg: Message) -> None:
    if not os.path.isdir(HOWTO_DIR):
        await msg.answer("📚 Папка howto пуста или недоступна.", parse_mode=None)
        return
    files = [
        f[:-3] for f in os.listdir(HOWTO_DIR) if f.lower().endswith(".md")
    ]
    if not files:
        await msg.answer("📚 Пока нет howto-заметок.", parse_mode=None)
        return
    lines = ["📚 Доступные howto:"]
    for name in sorted(files):
        lines.append(f"`/howto {name}`")
    await msg.answer("\n".join(lines))


async def send_howto_item(msg: Message, name: str) -> None:
    path = os.path.join(HOWTO_DIR, f"{name}.md")
    if not os.path.exists(path):
        await msg.answer("❌ Нет такого howto.", parse_mode=None)
        return
    with open(path, "r", encoding="utf-8") as f:
        body = f.read()
    if not body.strip():
        await msg.answer("⚠ Файл howto пуст.", parse_mode=None)
        return
    # howto — это наш markdown из файла, отдаём как есть
    await send_text_or_file(msg, body, filename_prefix=f"howto_{name}")


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
            line = f"`/{cmd}` - {escape_md(desc)}"
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

    safe_lines = [escape_md_smart(line) for line in lines]
    await msg.answer("\n".join(safe_lines))


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
            await msg.answer(text, parse_mode=None)
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
            await msg.answer(
                f"✅ Пользователь `{uid}` добавлен в список.",                
            )
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
            await msg.answer(
                f"❌ Пользователь `{uid}` удалён из списка.",                
            )
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
            await msg.answer(txt)
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
            await msg.answer(
                f"🧹 Статистика пользователя `{uid}` сброшена.",                
            )
            return

        if cmd == "stats_all":
            if not is_admin(user_id):
                return
            stats = users_db.get("stats", {})
            meta = users_db.get("meta", {})
            if not stats:
                await msg.answer("📊 Статистика пока пуста.", parse_mode=None)
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
            await msg.answer("\n\n".join(lines), parse_mode=None)
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

        use_text_override = False
        force_collage = False
        prompt_path: Optional[str] = None

        if cmd in ("text", "text_collage"):
            use_text_override = True
            force_collage = cmd == "text_collage"
        elif cmd in PROMPTS:
            prompt_path = PROMPTS[cmd].path
            if text_after_cmd:
                use_text_override = True
        else:
            if text_after_cmd:
                use_text_override = True

        # только текст
        if not images_bytes:
            if not text_after_cmd:
                await msg.answer("Нет ни изображений, ни текста.", parse_mode=None)
                return
            await msg.answer("💭 Думаю над текстом...", parse_mode=None)
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
            safe_body = escape_md(text_result)
            final = safe_body
            if total_cost > 0:
                final += f"\n\n*💎 {total_cost:.3f} у.е.*"
            await send_text_or_file(msg, final, filename_prefix="text")
            return

        # есть изображения — сначала ресайз
        from PIL import Image
        import io as _io

        def resize_bytes(data: bytes, max_size: int, quality: int) -> bytes:
            with Image.open(_io.BytesIO(data)) as im:
                im = im.convert("RGB")
                w, h = im.size
                scale = min(1.0, float(max_size) / max(w, h))
                if scale < 1.0:
                    new_size = (int(w * scale), int(h * scale))
                    im = im.resize(new_size, Image.LANCZOS)
                buf = _io.BytesIO()
                im.save(buf, format="JPEG", quality=quality, optimize=True)
                return buf.getvalue()

        resized: List[bytes] = [
            resize_bytes(b, cfg.image_max_size, cfg.image_quality)
            for b in images_bytes
        ]

        # выясняем системный промт / его имя
        if use_text_override:
            user_text = text_after_cmd
            system_prompt = ""
            prompt_label = "текст из сообщения"
        else:
            if prompt_path:
                system_prompt = read_prompt_file(prompt_path)
                prompt_label = os.path.splitext(os.path.basename(prompt_path))[0]
            else:
                if not cfg.prompt_file:
                    system_prompt = ""
                    prompt_label = "без промта"
                else:
                    system_prompt = read_prompt_file(cfg.prompt_file)
                    prompt_label = os.path.splitext(os.path.basename(cfg.prompt_file))[0]

        # статусное сообщение (без Markdown, чтобы промт с _ не ломал парсер)
        def fmt_mb(n_bytes: int) -> str:
            return f"{n_bytes / (1024.0 * 1024.0):.2f} MB"

        orig_total = sum(len(b) for b in images_bytes)
        resized_total = sum(len(b) for b in resized)
        n_files = len(resized)
        files_word = "файл" if n_files == 1 else "файла" if n_files < 5 else "файлов"

        await msg.answer(
            f"📷 Взял в работу {n_files} {files_word}. "
            f"Размер {fmt_mb(orig_total)} → {fmt_mb(resized_total)}, "
            f"промт {prompt_label}.",            
        )

        multiple = len(resized) > 1
        per_image = PER_IMAGE_DEFAULT
        if force_collage:
            per_image = False

        total_bytes = resized_total
        aggregated_texts: List[str] = []
        total_cost_request = 0.0

        if multiple and not per_image:
            named = [(f"image_{i+1}.jpg", b) for i, b in enumerate(resized)]
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

            resp = call_model_with_image(
                cfg,
                collage_bytes,
                system_prompt=collage_system_prompt,
                user_text=user_text_for_call,
                quiet=True,
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

            aggregated_texts.append(escape_md(text_result))
        else:
            for i, jpeg in enumerate(resized, start=1):
                if use_text_override:
                    system_prompt_for_call = ""
                    user_text_for_call = user_text
                else:
                    system_prompt_for_call = system_prompt
                    user_text_for_call = None
                resp = call_model_with_image(
                    cfg,
                    jpeg,
                    system_prompt=system_prompt_for_call,
                    user_text=user_text_for_call,
                    quiet=True,
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

                header = f"*Изображение #{i}*\n"
                aggregated_texts.append(header + escape_md(text_result))

        final_text = "\n\n".join(aggregated_texts)
        if total_cost_request > 0:
            final_text += f"\n\n*💎 {total_cost_request:.3f} у.е.*"

        await send_text_or_file(msg, final_text, filename_prefix="images")

    except Exception as e:
        await safe_error_reply(msg, e)


async def main() -> None:
    await setup_bot_commands()
    print("Bot is running...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
