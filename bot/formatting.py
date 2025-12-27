import re
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message, FSInputFile
from typing import Optional

FORMAT_MODE = "HTML"


def simple_markdown_to_html(md: str) -> str:
    def esc(s):
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
        )

    code_placeholders = []
    link_placeholders = []

    def code_repl(m):
        code_placeholders.append(m.group(1))
        return f"{{{{CODE{len(code_placeholders)-1}}}}}"

    def link_repl(m):
        link_placeholders.append((m.group(1), m.group(2)))
        return f"{{{{LINK{len(link_placeholders)-1}}}}}"

    md_wo_code = re.sub(r"`([^`]+?)`", code_repl, md)
    md_wo_code_links = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_repl, md_wo_code)
    html = esc(md_wo_code_links)

    html = re.sub(r"^# (.+)$", r"<b>\1</b>", html, flags=re.MULTILINE)
    html = re.sub(r"(?<!\w)\*\*(.+?)\*\*(?!\w)", r"<b>\1</b>", html)
    html = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", html)
    html = re.sub(r"^\* (.+)$", r"• \1", html, flags=re.MULTILINE)
    html = re.sub(r"\n{3,}", "\n\n", html)
    html = re.sub(r"^\n+", "", html)

    for i, (text, url) in enumerate(link_placeholders):
        safe_text = esc(text)
        html = html.replace(f"{{{{LINK{i}}}}}", f'<a href="{url}">{safe_text}</a>')
    for i, code in enumerate(code_placeholders):
        safe_code = esc(code).replace('<br>', '\n')
        html = html.replace(f"{{{{CODE{i}}}}}", f'<code>{safe_code}</code>')

    return html


async def send_response(msg: Message, text: Optional[str] = None, file_path: Optional[str] = None, filename_prefix: str = "response") -> None:
    if text:
        if FORMAT_MODE == "HTML":
            html = simple_markdown_to_html(text)
            if len(html) <= 3800:
                try:
                    await msg.answer(html, parse_mode="HTML")
                    return
                except TelegramBadRequest:
                    await msg.answer("❗ Telegram parse error; try plain text.")
                    return
        else:
            if len(text) <= 3800:
                await msg.answer(text)
                return
    if file_path:
        await msg.answer_document(FSInputFile(file_path))
        return
    await msg.answer("⚠ Пустое сообщение.")
