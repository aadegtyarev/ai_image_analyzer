#!/usr/bin/env python3
"""
Telegram-bot helper for photographers, built on top of ai_image_analyzer.

- Работает с JSON-API ai_image_analyzer.handle_json_request.
- Динамически регистрирует команды по файлам в prompts/.
- Поддерживает howto-заметки из папки howto/.
- Ведёт список разрешённых пользователей и статистику в users.json.
"""

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    Message,
    BotCommand,
    BufferedInputFile,
)

from dotenv import load_dotenv

from ai_image_analyzer import handle_json_request

# ------------------ logging ------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("photo_helper_bot")


# ------------------ config ------------------ #

@dataclass
class BotConfig:
    token: str
    admin_id: Optional[int]
    admin_username: Optional[str]
    prompts_dir: str
    howto_dir: str
    users_file: str
    per_image_default: bool
    default_prompt_file: Optional[str]


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def load_config() -> BotConfig:
    load_dotenv()

    token = os.getenv("BOT_TOKEN")
    if not token:
        raise SystemExit("BOT_TOKEN is not set in .env")

    admin_id_str = os.getenv("BOT_ADMIN_ID")
    admin_id = int(admin_id_str) if admin_id_str else None
    admin_username = os.getenv("BOT_ADMIN_USERNAME")

    prompts_dir = os.getenv("PROMPTS_DIR", "prompts")
    howto_dir = os.getenv("HOWTO_DIR", "howto")
    users_file = os.getenv("USERS_FILE", "users.json")

    per_image_default = _env_bool("PER_IMAGE_DEFAULT", True)

    default_prompt_file = os.getenv("PROMPT_FILE")

    cfg = BotConfig(
        token=token,
        admin_id=admin_id,
        admin_username=admin_username.lstrip("@") if admin_username else None,
        prompts_dir=prompts_dir,
        howto_dir=howto_dir,
        users_file=users_file,
        per_image_default=per_image_default,
        default_prompt_file=default_prompt_file,
    )
    log.info("Config loaded: %s", cfg)
    return cfg


# ------------------ user store & stats ------------------ #

@dataclass
class UserStats:
    enabled: bool = False
    username: Optional[str] = None
    full_name: Optional[str] = None
    total_requests: int = 0
    total_images: int = 0
    total_megabytes: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    last_used: Optional[str] = None  # ISO8601


class UserStore:
    def __init__(self, path: str):
        self.path = path
        self.users: Dict[str, UserStats] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for uid, data in raw.get("users", {}).items():
                    self.users[uid] = UserStats(**data)
                log.info("UserStore loaded: %d users", len(self.users))
            except Exception as e:
                log.error("Failed to load users file %s: %s", self.path, e)
        else:
            log.info("UserStore: no existing file, starting empty")
        self._loaded = True

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        data = {"users": {uid: asdict(u) for uid, u in self.users.items()}}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
        log.info("UserStore saved: %s", self.path)

    def ensure_user(self, user_id: int, username: Optional[str], full_name: Optional[str]) -> UserStats:
        self.load()
        key = str(user_id)
        if key not in self.users:
            self.users[key] = UserStats(
                enabled=False,
                username=(username.lstrip("@") if username else None),
                full_name=full_name,
            )
            self.save()
        else:
            u = self.users[key]
            changed = False
            if username and u.username != username.lstrip("@"):
                u.username = username.lstrip("@")
                changed = True
            if full_name and u.full_name != full_name:
                u.full_name = full_name
                changed = True
            if changed:
                self.save()
        return self.users[key]

    def is_allowed(self, user_id: int, username: Optional[str], cfg: BotConfig) -> bool:
        # Админ всегда разрешён
        if cfg.admin_id is not None and user_id == cfg.admin_id:
            return True
        if cfg.admin_username and username and username.lstrip("@") == cfg.admin_username:
            return True

        user = self.ensure_user(user_id, username, None)
        return user.enabled

    def set_enabled_by_username(self, username: str, enabled: bool) -> bool:
        username = username.lstrip("@")
        self.load()
        found = False
        for u in self.users.values():
            if u.username == username:
                u.enabled = enabled
                found = True
        if found:
            self.save()
        return found

    def all_users(self) -> List[Tuple[str, UserStats]]:
        self.load()
        return sorted(self.users.items(), key=lambda kv: int(kv[0]))

    def add_stats(
        self,
        user_id: int,
        username: Optional[str],
        images_count: int,
        total_bytes: int,
        requests_count: int,
        tokens_used: int,
        total_cost: float,
    ) -> None:
        u = self.ensure_user(user_id, username, None)
        u.total_requests += requests_count
        u.total_images += images_count
        u.total_megabytes += float(total_bytes) / (1024 * 1024)
        u.total_tokens += tokens_used
        u.total_cost += float(total_cost)
        u.last_used = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self.save()

    def reset_stats_all(self) -> None:
        self.load()
        for u in self.users.values():
            u.total_requests = 0
            u.total_images = 0
            u.total_megabytes = 0.0
            u.total_tokens = 0
            u.total_cost = 0.0
        self.save()

    def reset_stats_username(self, username: str) -> bool:
        username = username.lstrip("@")
        self.load()
        found = False
        for u in self.users.values():
            if u.username == username:
                u.total_requests = 0
                u.total_images = 0
                u.total_megabytes = 0.0
                u.total_tokens = 0
                u.total_cost = 0.0
                found = True
        if found:
            self.save()
        return found


# ------------------ prompts & howto ------------------ #

def scan_prompts(prompts_dir: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not os.path.isdir(prompts_dir):
        log.warning("Prompts dir %r not found", prompts_dir)
        return mapping
    for name in sorted(os.listdir(prompts_dir)):
        path = os.path.join(prompts_dir, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith((".txt", ".md")):
            continue
        base = os.path.splitext(name)[0]
        mapping[base] = path
    log.info("Found %d prompts: %s", len(mapping), ", ".join(mapping.keys()))
    return mapping


def scan_howto(howto_dir: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not os.path.isdir(howto_dir):
        log.info("Howto dir %r not found, skipping", howto_dir)
        return mapping
    for name in sorted(os.listdir(howto_dir)):
        path = os.path.join(howto_dir, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith(".md"):
            continue
        base = os.path.splitext(name)[0]
        mapping[base] = path
    log.info("Found %d howto files: %s", len(mapping), ", ".join(mapping.keys()))
    return mapping


# ------------------ helpers ------------------ #

def parse_command_and_args(text: str, bot_username: Optional[str]) -> Tuple[Optional[str], str]:
    """
    Возвращает (command, args_text_without_command).

    command без '/', без @botname.
    """
    text = text.strip()
    if not text.startswith("/"):
        return None, text

    first, _, rest = text.partition(" ")
    cmd = first[1:]
    if "@" in cmd:
        cmd, _, at = cmd.partition("@")
        if bot_username and at.lower() != bot_username.lower():
            # команда не для этого бота
            return None, text

    return cmd, rest.strip()


def strip_collage_flag(args: str) -> Tuple[str, bool]:
    tokens = args.split()
    remaining: List[str] = []
    collage = False
    for t in tokens:
        tl = t.lower()
        if tl in ("collage", "коллаж"):
            collage = True
        else:
            remaining.append(t)
    return " ".join(remaining).strip(), collage


async def collect_images_from_message(message: Message) -> List[Tuple[str, bytes, int]]:
    """
    Собирает изображения из самого сообщения и, если это reply,
    из сообщения, на которое отвечаем.

    Возвращает список (name, bytes, size_bytes).
    """
    images: List[Tuple[str, bytes, int]] = []

    async def _extract_from(msg: Message) -> None:
        if msg.photo:
            photo = msg.photo[-1]
            file = await msg.bot.get_file(photo.file_id)
            b = await msg.bot.download_file(file.file_path)
            content = b.read()
            name = f"photo_{photo.file_unique_id}.jpg"
            images.append((name, content, len(content)))
        elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("image/"):
            doc = msg.document
            file = await msg.bot.get_file(doc.file_id)
            b = await msg.bot.download_file(file.file_path)
            content = b.read()
            name = doc.file_name or f"doc_{doc.file_unique_id}.jpg"
            images.append((name, content, len(content)))

    await _extract_from(message)
    if message.reply_to_message:
        await _extract_from(message.reply_to_message)

    return images


def build_analyzer_request(
    *,
    images: List[Tuple[str, bytes, int]],
    prompt_file: Optional[str],
    override_text: Optional[str],
    per_image: bool,
) -> dict:
    images_payload = [
        {
            "name": name,
            "data_b64": base64.b64encode(data).decode("ascii"),
        }
        for (name, data, _size) in images
    ]
    req = {
        "action": "analyze",
        "prompt_file": prompt_file,
        "override_text": override_text,
        "per_image": per_image,
        "quiet": True,
        "include_billing": True,
        "images": images_payload,
        "overrides": {},
    }
    return req


def summarize_usage(resp: dict) -> Tuple[int, int, float]:
    """
    Суммарная статистика по ответу анализатора:
    - requests_count: количество модельных вызовов
    - total_tokens: суммарное число токенов
    - total_cost: суммарная стоимость (если провайдер вернул total_cost/cost)
    """
    requests_count = 0
    total_tokens = 0
    total_cost = 0.0

    def _acc_usage(u: Any) -> None:
        nonlocal requests_count, total_tokens, total_cost
        if not isinstance(u, dict):
            return
        requests_count += 1
        try:
            total_tokens_local = int(u.get("total_tokens", 0) or 0)
        except Exception:
            total_tokens_local = 0
        total_tokens += total_tokens_local
        cost_val = u.get("total_cost") or u.get("cost")
        if cost_val is not None:
            try:
                total_cost += float(cost_val)
            except Exception:
                pass

    if "usage" in resp:
        _acc_usage(resp.get("usage"))

    for item in resp.get("results") or []:
        u = item.get("usage")
        if u:
            _acc_usage(u)

    return requests_count, total_tokens, total_cost


async def send_result_text_or_file(status_msg: Message, text: str, filename_prefix: str) -> None:
    """
    Для ответов модели: если текст помещается — редактируем статус этим текстом.
    Если нет — пишем короткое сообщение и прикладываем txt-файл с полным ответом.
    """
    MAX_LEN = 4000
    if len(text) <= MAX_LEN:
        await status_msg.edit_text(text or "Пустой ответ от модели.")
        return

    await status_msg.edit_text(
        "Ответ модели получен, он длиннее лимита Telegram.\n"
        "Прикрепляю полный текст в файле."
    )
    data = text.encode("utf-8")
    fname = f"{filename_prefix}.txt"
    file = BufferedInputFile(data, filename=fname)
    await status_msg.answer_document(
        document=file,
        caption="Полный ответ модели в txt-файле.",
    )


async def send_text_message_or_file(message: Message, text: str, filename_prefix: str) -> None:
    """
    Для howto и других «просто сообщений»:
    если текст помещается — отправляем как обычное сообщение;
    если нет — отправляем текст-файл.
    """
    MAX_LEN = 4000
    if len(text) <= MAX_LEN:
        await message.reply(text)
        return

    await message.reply(
        "Текст длиннее лимита Telegram.\n"
        "Прикрепляю полный текст в файле."
    )
    data = text.encode("utf-8")
    fname = f"{filename_prefix}.txt"
    file = BufferedInputFile(data, filename=fname)
    await message.answer_document(
        document=file,
        caption=f"{filename_prefix}.txt",
    )


# ------------------ main bot logic ------------------ #

cfg: BotConfig
user_store: UserStore
prompts: Dict[str, str]
howtos: Dict[str, str]
bot_username_cache: Optional[str] = None


def _ensure_admin(message: Message) -> bool:
    user = message.from_user
    if not user:
        return False
    uid = user.id
    uname = user.username
    if cfg.admin_id is not None and uid == cfg.admin_id:
        return True
    if cfg.admin_username and uname and uname.lstrip("@") == cfg.admin_username:
        return True
    return False


def _is_allowed_or_admin(message: Message) -> bool:
    user = message.from_user
    if not user:
        return False
    if _ensure_admin(message):
        return True
    return user_store.is_allowed(user.id, user.username, cfg)


async def cmd_help(message: Message):
    if not _is_allowed_or_admin(message):
        return
    await message.reply(
        "Я помогаю разбирать и обсуждать ваши фотографии.\n\n"
        "• Пришлите фото (или несколько) — я разберу их по умолчательному промту.\n"
        "• Добавьте команду с промтом, например: `/art_analysis`.\n"
        "• Добавьте слово `collage`, чтобы склеить несколько фото в один коллаж.\n"
        "• Используйте `/text` — чтобы задать свой текст вместо системного промта.\n"
        "• `/howto` — список howto-заметок, `/howto имя` — показать одну.\n",
        parse_mode="Markdown",
    )


async def cmd_howto(message: Message):
    if not _is_allowed_or_admin(message):
        return

    text = message.text or message.caption or ""
    _, args = text.split(maxsplit=1) if " " in text else (text, "")
    name = args.strip()

    if not name:
        if not howtos:
            await message.reply("Пока нет доступных howto-заметок.")
            return
        names = ", ".join(sorted(howtos.keys()))
        await message.reply(
            "Доступные howto:\n"
            f"{names}\n\n"
            "Используй `/howto имя`, например: `/howto posing`.",
            parse_mode="Markdown",
        )
        return

    key = name.strip()
    if key not in howtos:
        await message.reply(f"Не знаю howto `{key}`.", parse_mode="Markdown")
        return

    path = howtos[key]
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        await message.reply(f"Не удалось прочитать howto `{key}`: {e}")
        return

    await send_text_message_or_file(message, content, f"howto_{key}")


async def cmd_user_add(message: Message):
    if not _ensure_admin(message):
        return

    text = message.text or ""
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await message.reply("Использование: `user_add @username`", parse_mode="Markdown")
        return
    username = parts[1].strip().lstrip("@")
    if not username:
        await message.reply("Нужно указать username.", parse_mode="Markdown")
        return

    ok = user_store.set_enabled_by_username(username, True)
    if ok:
        await message.reply(f"Пользователь @{username} включён.")
    else:
        await message.reply(
            f"Не нашёл @{username} в базе. Человек должен сначала отправить хоть одно сообщение боту."
        )


async def cmd_user_del(message: Message):
    if not _ensure_admin(message):
        return

    text = message.text or ""
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await message.reply("Использование: `user_del @username`", parse_mode="Markdown")
        return
    username = parts[1].strip().lstrip("@")
    if not username:
        await message.reply("Нужно указать username.", parse_mode="Markdown")
        return

    ok = user_store.set_enabled_by_username(username, False)
    if ok:
        await message.reply(f"Пользователь @{username} выключен.")
    else:
        await message.reply(f"Не нашёл @{username} в базе.")


async def cmd_users(message: Message):
    if not _ensure_admin(message):
        return

    lines: List[str] = []
    for uid, u in user_store.all_users():
        mark = "✅" if u.enabled else "❌"
        name = f"@{u.username}" if u.username else "(нет username)"
        lines.append(
            f"{mark} {uid}: {name}, "
            f"req={u.total_requests}, img={u.total_images}, "
            f"MB={u.total_megabytes:.2f}, tokens={u.total_tokens}, "
            f"cost={u.total_cost:.4f}"
        )
    if not lines:
        await message.reply("Пока нет ни одного пользователя.")
    else:
        await message.reply("Пользователи:\n" + "\n".join(lines))


async def cmd_stats(message: Message):
    if not _ensure_admin(message):
        return

    lines: List[str] = []
    total_req = total_img = total_tokens = 0
    total_mb = 0.0
    total_cost = 0.0
    for uid, u in user_store.all_users():
        total_req += u.total_requests
        total_img += u.total_images
        total_mb += u.total_megabytes
        total_tokens += u.total_tokens
        total_cost += u.total_cost
        name = f"@{u.username}" if u.username else "(нет username)"
        lines.append(
            f"{uid} {name}: req={u.total_requests}, img={u.total_images}, "
            f"MB={u.total_megabytes:.2f}, tokens={u.total_tokens}, "
            f"cost={u.total_cost:.4f}"
        )
    header = (
        f"Суммарно: req={total_req}, img={total_img}, "
        f"MB={total_mb:.2f}, tokens={total_tokens}, cost={total_cost:.4f}\n"
    )
    if not lines:
        await message.reply(header + "Пока нет ни одного пользователя.")
    else:
        await message.reply(header + "\n".join(lines))


async def cmd_stats_reset(message: Message):
    if not _ensure_admin(message):
        return

    text = message.text or ""
    parts = text.split(maxsplit=1)
    if len(parts) == 1:
        user_store.reset_stats_all()
        await message.reply("Статистика всех пользователей сброшена.")
        return

    arg = parts[1].strip()
    if arg.lower() == "all":
        user_store.reset_stats_all()
        await message.reply("Статистика всех пользователей сброшена.")
        return

    username = arg.lstrip("@")
    ok = user_store.reset_stats_username(username)
    if ok:
        await message.reply(f"Статистика пользователя @{username} сброшена.")
    else:
        await message.reply(f"Не нашёл @{username} в базе.")


async def cmd_billing(message: Message):
    """Админ-команда: запрос баланса у провайдера через JSON-API анализатора."""
    if not _ensure_admin(message):
        return

    status = await message.reply("Запрашиваю баланс провайдера…")
    req = {"action": "analyze", "check_balance": True, "quiet": True}
    resp = handle_json_request(req)

    if not resp.get("ok"):
        errs = resp.get("errors") or []
        err_text = "; ".join(str(e) for e in errs) or str(resp.get("error") or "unknown error")
        await status.edit_text(f"Не удалось получить баланс: {err_text}")
        return

    balance = resp.get("balance")
    errs = resp.get("errors") or []

    lines: List[str] = []
    if isinstance(balance, dict):
        bal_val = balance.get("balance") or balance.get("available") or balance.get("value")
        currency = balance.get("currency") or balance.get("unit") or "$"
        if bal_val is not None:
            lines.append(f"Баланс: {bal_val} {currency}")
        else:
            lines.append("Баланс (сырые данные):")
            lines.append(json.dumps(balance, ensure_ascii=False, indent=2))
    else:
        lines.append("Баланс (сырые данные):")
        lines.append(repr(balance))

    if errs:
        lines.append("\nСообщения:")
        lines.extend(str(e) for e in errs)

    await send_text_message_or_file(message, "\n".join(lines), "billing")


async def handle_any(message: Message):
    """
    Главный обработчик: команды промтов, /text, "без команды" и т.п.
    """
    global bot_username_cache

    user = message.from_user
    if not user:
        return

    # Обновляем/создаём запись пользователя
    user_store.ensure_user(user.id, user.username, user.full_name)

    # Проверка доступа (неразрешённых тихо игнорируем)
    if not _is_allowed_or_admin(message):
        return

    # Определяем username бота, чтобы корректно резать /cmd@botname
    if not bot_username_cache:
        me = await message.bot.get_me()
        bot_username_cache = me.username

    text_full = (message.text or message.caption or "").strip()

    cmd, args = parse_command_and_args(text_full, bot_username_cache)
    args_without_collage, collage_flag = strip_collage_flag(args)

    # Собираем изображения
    images = await collect_images_from_message(message)
    images_count = len(images)

    # Определяем промт/override_text
    prompt_file: Optional[str] = None
    override_text: Optional[str] = None

    # Служебные команды здесь не обрабатываем
    if cmd in {"help", "howto", "user_add", "user_del", "users", "stats", "stats_reset", "billing", "start"}:
        return

    # Команда /text — всегда override_text
    if cmd == "text":
        override_text = args_without_collage or None
        prompt_file = None

    # Команда соответствует одному из промтов
    elif cmd and cmd in prompts:
        prompt_file = prompts[cmd]
        override_text = None

    else:
        # Нет явной промт-команды
        if args_without_collage:
            override_text = args_without_collage
            prompt_file = None
        else:
            prompt_file = cfg.default_prompt_file

    # Определяем режим per_image (поштучно / коллаж)
    if collage_flag:
        per_image = False
    else:
        per_image = cfg.per_image_default

    # Случай: только текст, без картинок
    if images_count == 0:
        if not override_text:
            await message.reply("Нужен либо текст, либо фото.")
            return

        status = await message.reply("Принял текст, отправляю запрос модели…")
        req = {
            "action": "analyze",
            "prompt_file": None,
            "override_text": override_text,
            "per_image": False,
            "quiet": True,
            "include_billing": True,
            "images": [],
            "overrides": {},
        }
        resp = handle_json_request(req)
        if not resp.get("ok"):
            await status.edit_text("Ошибка при обращении к модели.")
            return

        text_only = resp.get("text_only") or ""
        await send_result_text_or_file(status, text_only, "text_analysis")

        # учёт статистики
        requests_count, tokens_used, total_cost = summarize_usage(resp)
        user_store.add_stats(
            user_id=user.id,
            username=user.username,
            images_count=0,
            total_bytes=0,
            requests_count=requests_count,
            tokens_used=tokens_used,
            total_cost=total_cost,
        )
        return

    # Тут есть картинки
    if collage_flag:
        mode_desc = "коллаж"
    else:
        mode_desc = "поштучно" if per_image else "авто/коллаж"

    prompt_desc = prompt_file or ("override_text" if override_text else "(по умолчанию)")

    status = await message.reply(
        f"Принял {images_count} изображение(й).\n"
        f"Режим: {mode_desc}.\n"
        f"Промт: {prompt_desc}.\n"
        f"Отправляю в модель…"
    )

    req = build_analyzer_request(
        images=images,
        prompt_file=prompt_file,
        override_text=override_text,
        per_image=per_image,
    )

    # Запрос к анализатору
    resp = handle_json_request(req)
    if not resp.get("ok"):
        await status.edit_text(f"Ошибка при анализе: {resp.get('error') or 'unknown error'}")
        return

    mode = resp.get("mode")
    results = resp.get("results") or []

    # Формируем ответ пользователю
    if mode in ("single", "collage", "image_with_text"):
        text_out = results[0].get("text", "")
        await send_result_text_or_file(status, text_out, "analysis")
    elif mode == "per_image":
        lines: List[str] = []
        for idx, r in enumerate(results, start=1):
            name = r.get("image_name") or f"image_{idx}"
            text_out = r.get("text", "")
            lines.append(f"#{idx} ({name}):\n{text_out}\n")
        final = "\n".join(lines)
        await send_result_text_or_file(status, final, "analysis_per_image")
    elif mode == "text_only":
        text_out = resp.get("text_only") or ""
        await send_result_text_or_file(status, text_out, "analysis_text")
    else:
        await status.edit_text("Модель вернула неожиданный формат ответа.")
        return

    # Статистика
    total_bytes = sum(size for (_, _, size) in images)
    requests_count, tokens_used, total_cost = summarize_usage(resp)
    user_store.add_stats(
        user_id=user.id,
        username=user.username,
        images_count=images_count,
        total_bytes=total_bytes,
        requests_count=requests_count,
        tokens_used=tokens_used,
        total_cost=total_cost,
    )


# ------------------ startup / main ------------------ #

async def on_startup(bot: Bot):
    # Регистрируем команды
    commands: List[BotCommand] = [
        BotCommand(command="help", description="Как пользоваться ботом"),
        BotCommand(command="text", description="Анализ по тексту без промта"),
        BotCommand(command="howto", description="Список howto-заметок"),
        BotCommand(command="billing", description="Показать баланс провайдера (админ)"),
    ]
    for name in sorted(prompts.keys()):
        commands.append(BotCommand(command=name, description=f"Промт: {name}"))
    await bot.set_my_commands(commands)
    log.info("Bot commands registered: %s", [c.command for c in commands])


async def main():
    global cfg, user_store, prompts, howtos

    cfg = load_config()
    user_store = UserStore(cfg.users_file)
    prompts = scan_prompts(cfg.prompts_dir)
    howtos = scan_howto(cfg.howto_dir)

    bot = Bot(token=cfg.token)
    dp = Dispatcher()

    dp.startup.register(on_startup)

    dp.message.register(cmd_help, Command("help"))
    dp.message.register(cmd_howto, Command("howto"))

    dp.message.register(cmd_user_add, Command("user_add"))
    dp.message.register(cmd_user_del, Command("user_del"))
    dp.message.register(cmd_users, Command("users"))
    dp.message.register(cmd_stats, Command("stats"))
    dp.message.register(cmd_stats_reset, Command("stats_reset"))
    dp.message.register(cmd_billing, Command("billing"))

    # основной обработчик — в самом конце
    dp.message.register(handle_any, F)

    log.info("Starting bot polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
