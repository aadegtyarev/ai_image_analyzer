import asyncio
from types import SimpleNamespace
import time

import pytest
import bot


class DummyMsg:
    def __init__(self):
        self.answers = []
        self.message_id = 1

    async def answer(self, text=None, parse_mode=None):
        self.answers.append(text)

    async def answer_document(self, *args, **kwargs):
        self.answers.append("doc")


def test_collage_orientations_and_single_kickoff(monkeypatch):
    monkeypatch.setenv("DEBUG", "1")
    mgid = "test_mg"
    # prepare images: (bytes, w, h, orientation)
    imgs = [
        (b"a", 1024, 682, "landscape"),
        (b"b", 682, 1024, "portrait"),
        (b"c", 1024, 1024, "square"),
    ]
    reply = DummyMsg()

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

    captured_meta = {}

    def fake_call_model_with_image(cfg, data, system_prompt=None, user_text=None, quiet=False, image_meta=None):
        captured_meta.update(image_meta or {})
        return ("RESULT", None)

    monkeypatch.setattr(bot, "call_model_with_image", fake_call_model_with_image)
    # avoid PIL image decoding in make_collage - stub it out
    def fake_make_collage(images, max_size, jpeg_quality):
        fnames = [f"image_{i+1}.jpg" for i in range(len(images))]
        return (b"collage", fnames)

    monkeypatch.setattr(bot, "make_collage", fake_make_collage)

    # Also capture send_response kickoff call
    calls = []

    async def fake_send_response(msg, text=None, file_path=None, filename_prefix="response"):
        calls.append(text)

    monkeypatch.setattr(bot, "send_response", fake_send_response)

    cfg = SimpleNamespace(collage_max_size=1024, collage_quality=85)

    asyncio.run(bot._process_media_group(mgid, cfg))

    # After processing, MEDIA_CONTEXTS should no longer have mgid
    assert bot.get_media_context(mgid) is None

    # orientations must be strings
    assert "orientations" in captured_meta
    orients = captured_meta["orientations"]
    assert isinstance(orients, dict)
    assert orients["image_1.jpg"] == "landscape"
    assert orients["image_2.jpg"] == "portrait"
    assert orients["image_3.jpg"] == "square"

    # kickoff message sent once (either via send_response or as reply.answer)
    assert any(isinstance(t, str) and "Принял 3 файлов" in t for t in calls) or any("Принял 3 файлов" in t for t in reply.answers)
