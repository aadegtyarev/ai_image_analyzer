"""Bot package — thin adapter modules for Telegram bot implementation.

This package exposes a small compatibility surface so tests that import from
`bot` (top-level module previously) continue to work during refactor.
"""

from .utils import parse_tail_flags, normalize_command
from .formatting import simple_markdown_to_html, send_response
from .prompts import load_prompts
from .media_group import (
	set_media_context,
	get_media_context,
	MEDIA_CONTEXTS,
	MEDIA_CONTEXT_TTL,
	_cleanup_media_contexts,
	update_media_context_with_override,
)

__all__ = [
	"parse_tail_flags",
	"normalize_command",
	"simple_markdown_to_html",
	"send_response",
	"load_prompts",
	"set_media_context",
	"get_media_context",
	"MEDIA_CONTEXTS",
	"MEDIA_CONTEXT_TTL",
	"_cleanup_media_contexts",
	"update_media_context_with_override",
	"bot",
]

# Dynamic prompts loaded at import time
PROMPTS = load_prompts()

async def handle_help(msg):
	allowed = is_allowed(getattr(getattr(msg, 'from_user', None), 'id', 0), users_db, BOT_ADMIN_ID)
	lines = [
		"**AI Photo Assistant**",
		"",
		"📷 Отправь фото — получишь разбор.",
		"",
		"",
		"🛠 Команды:",
		"- /howto – список howto-заметок.",
		"- /stats – твоя личная статистика.",
		"- /help – краткая справка.",
	]
	if is_admin(getattr(getattr(msg, 'from_user', None), 'id', 0), BOT_ADMIN_ID) and getattr(getattr(msg, 'chat', None), 'type', None) == 'private':
		lines.extend(["", "", "👑 Админ-команды (ручной ввод):", "- /users – список разрешённых пользователей.", "- /balance – баланс API."])
		lines.append("")
		lines.append("🎯 Промты:")
		if PROMPTS:
			for cmd, p in sorted(PROMPTS.items()):
				desc = getattr(p, 'description', '') or ''
				lines.append(f"- `/{cmd}` - {desc}")
		else:
			lines.append("(папка PROMPTS_DIR пуста)")
		lines.extend(["", "", "📒 Дополнительно:", "- Если вместе с фото написать текст, он заменит системный промт.", "- Добавьте флаг `group` (например `/text group` или `/art group`), чтобы при нескольких фото собрать коллаж.", "- `/text` – использовать текст сообщения как запрос (без системного промта)."])
		if not allowed:
			lines.insert(2, "⚠️ Вы не авторизованы для использования бота. Попросите админа добавить ваш ID через /user_add.")
	await send_response(msg, "\n".join(lines))

__all__.append("handle_help")

# Placeholder 'bot' attribute (tests may monkeypatch this to a FakeBot)
bot = None

# Expose a few convenience names expected by existing tests and code.
from ai_image_analyzer import call_model_with_image, make_collage, call_model_with_text_only
from ai_image_analyzer import build_collage_system_prompt
import importlib
from . import config as _cfg
importlib.reload(_cfg)
BOT_ADMIN_ID = _cfg.BOT_ADMIN_ID
from .users_store import load_users, is_allowed, is_admin

users_db = load_users()

import socket


async def notify_admin_startup() -> None:
	try:
		if BOT_ADMIN_ID:
			host = socket.gethostname()
			admin_text = f"✅ Bot started and ready on {host}."
			if bot:
				await bot.send_message(BOT_ADMIN_ID, admin_text)
	except Exception as e:
		print(f"[STARTUP] failed to notify admin: {e}", file=sys.stderr)

__all__.extend(["call_model_with_image", "make_collage", "call_model_with_text_only", "users_db", "BOT_ADMIN_ID", "notify_admin_startup"]) 
__all__.append("build_collage_system_prompt")
from .media_group import _process_media_group
__all__.append("_process_media_group")
from .handlers import setup_bot_commands, main_handler, send_howto_list, send_howto_item
__all__.extend(["setup_bot_commands", "main_handler", "send_howto_list", "send_howto_item"])
