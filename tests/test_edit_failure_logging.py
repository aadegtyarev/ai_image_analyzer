import time
from types import SimpleNamespace
import asyncio

import bot


class DummyMsgWithAnswer:
    def __init__(self):
        self.message_id = 99
        self.chat = type("C", (), {"id": 12345})()

    async def answer(self, text=None, parse_mode=None):
        # Return a message-like object
        return self


def test_edit_failure_logs_and_falls_back(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("DEBUG", "1")
    mgid = "mg_edit_fail"
    imgs = [(b"a", 100, 100, "square")]
    reply = DummyMsgWithAnswer()

    bot.MEDIA_CONTEXTS[mgid] = {
        "images": imgs,
        "system_prompt": "SP",
        "prompt_label": "p",
        "use_text_override": False,
        "user_text": None,
        "force_collage": True,
        "reply_msg": reply,
        "ts": time.time() - 10,
    }

    # make_collage stub
    def fake_make_collage(images, max_size, jpeg_quality):
        fnames = ["image_1.jpg"]
        return (b"collage", fnames)

    monkeypatch.setattr(bot, "make_collage", fake_make_collage)

    # edit_message_text will raise
    async def fake_edit_message_text(*args, **kwargs):
        raise RuntimeError("edit failed")

    # edit_message_text is a method on the Bot instance (bot.bot)
    monkeypatch.setattr(bot.bot, "edit_message_text", fake_edit_message_text)

    # stub call_model_with_image to avoid external calls
    def fake_call_model_with_image(cfg, data, system_prompt=None, user_text=None, quiet=False, image_meta=None):
        return ("RESULT", None)

    monkeypatch.setattr(bot, "call_model_with_image", fake_call_model_with_image)

    # capture whether send_response fallback was used
    calls = []

    async def fake_send_response(msg, text=None, file_path=None, filename_prefix="response"):
        calls.append(text)

    monkeypatch.setattr(bot, "send_response", fake_send_response)

    cfg = SimpleNamespace(collage_max_size=1024, collage_quality=85)
    bot.GROUP_WAIT = 0

    asyncio.run(bot._process_media_group(mgid, cfg))

    captured = capsys.readouterr()
    print("STDERR:\n", captured.err)
    print("CALLS:\n", calls)
    assert "failed to edit status message" in captured.err
    # fallback send_response should have been called with final text (collage or per-image)
    assert any(isinstance(t, str) and ("Коллаж — промт" in t or "Изображение #1 — промт" in t) for t in calls)
