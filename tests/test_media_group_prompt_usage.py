import time
from types import SimpleNamespace
import asyncio

import bot


class DummyMsg:
    def __init__(self):
        self.answers = []
        self.message_id = 1

    async def answer(self, text=None, parse_mode=None):
        self.answers.append(text)

    async def answer_document(self, *args, **kwargs):
        self.answers.append("doc")


def test_media_group_uses_prompt_path_on_process(monkeypatch, tmp_path):
    monkeypatch.setenv("DEBUG", "1")
    mgid = "mg_prompt_test"
    # create a prompt file
    p = tmp_path / "special_prompt.txt"
    p.write_text("SPECIAL PROMPT CONTENT\n", encoding="utf-8")

    # prepare images: (bytes, w, h, orientation)
    imgs = [
        (b"a", 1024, 682, "landscape"),
        (b"b", 682, 1024, "portrait"),
    ]
    reply = DummyMsg()

    bot.MEDIA_CONTEXTS[mgid] = {
        "images": imgs,
        "system_prompt": "",  # simulate missing system_prompt text
        "prompt_label": "art_create_series_plan",
        "prompt_path": str(p),
        "use_text_override": False,
        "user_text": None,
        "force_collage": True,
        "reply_msg": reply,
        "ts": time.time() - 10,
    }

    captured = {"system_prompt": None}

    def fake_call_model_with_image(cfg, data, system_prompt=None, user_text=None, quiet=False, image_meta=None):
        captured["system_prompt"] = system_prompt or ""
        return ("RESULT", None)

    monkeypatch.setattr(bot, "call_model_with_image", fake_call_model_with_image)
    # stub make_collage to avoid PIL decoding
    def fake_make_collage(images, max_size, jpeg_quality):
        fnames = [f"image_{i+1}.jpg" for i in range(len(images))]
        return (b"collage", fnames)

    monkeypatch.setattr(bot, "make_collage", fake_make_collage)

    calls = []

    async def fake_send_response(msg, text=None, file_path=None, filename_prefix="response"):
        calls.append(text)

    monkeypatch.setattr(bot, "send_response", fake_send_response)

    cfg = SimpleNamespace(collage_max_size=1024, collage_quality=85)

    # speed up GROUP_WAIT to avoid waiting in test
    bot.GROUP_WAIT = 0
    asyncio.run(bot._process_media_group(mgid, cfg))

    # MEDIA_CONTEXTS cleared
    assert bot.get_media_context(mgid) is None
    # system_prompt used by model must include the prompt file content
    assert "SPECIAL PROMPT CONTENT" in captured["system_prompt"]
    # kickoff message was sent (either via send_response or as reply.answer)
    assert any("Принял 2 файлов" in (t or "") for t in calls) or any("Принял 2 файлов" in t for t in reply.answers)
