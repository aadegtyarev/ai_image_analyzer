import time
import os
import sys
from typing import Optional
from .prompts import load_prompts

# Temporary cache for media group prompts: media_group_id -> {system_prompt, prompt_label, use_text_override, user_text, ts}
MEDIA_CONTEXTS: dict = {}
MEDIA_CONTEXT_TTL = int(os.getenv("MEDIA_CONTEXT_TTL", "120"))
GROUP_WAIT = float(os.getenv("GROUP_WAIT", "1.0"))


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


def update_media_context_with_override(media_group_id: str, prompt_path: Optional[str], text_override: Optional[str], prompt_debug: bool = False) -> None:
    if not media_group_id:
        return
    ctx = get_media_context(media_group_id)
    if not ctx:
        return
    changed = False
    try:
        if text_override:
            ctx["use_text_override"] = True
            ctx["user_text"] = text_override
            ctx["system_prompt"] = ""
            ctx["prompt_label"] = "текст из сообщения"
            changed = True
        if prompt_path:
            try:
                sp = ""
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        sp = f.read()
                except Exception:
                    sp = ""
            except Exception:
                sp = ""
            ctx["system_prompt"] = sp
            ctx["prompt_label"] = os.path.splitext(os.path.basename(prompt_path))[0]
            ctx["prompt_path"] = prompt_path
            changed = True
        if changed:
            set_media_context(media_group_id, ctx)
            if prompt_debug:
                print(f"[PROMPT_DEBUG] updated media context {media_group_id}: prompt_label={ctx.get('prompt_label')}", file=sys.stderr)
    except Exception:
        if prompt_debug:
            print(f"[PROMPT_DEBUG] failed to update media context {media_group_id}", file=sys.stderr)


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


async def _process_media_group(mgid: str, cfg=None) -> None:
    """Process cached media group: build collage or per-image calls and send results.

    If `cfg` is not provided, attempt to use `bot.cfg` (set by `bot.main.setup_config()`)
    or fall back to calling `ai_image_analyzer.load_config()`.
    """
    if cfg is None:
        try:
            cfg = getattr(__import__('bot'), 'cfg', None)
            if cfg is None:
                from ai_image_analyzer import load_config

                cfg = load_config()
        except Exception:
            cfg = None
    try:
        # If context is not ready, return
        ctx = get_media_context(mgid)
        if not ctx:
            return

        imgs = ctx.get("images", [])
        system_prompt = ctx.get("system_prompt", "")
        prompt_label = ctx.get("prompt_label", "без промта")
        use_text_override = ctx.get("use_text_override", False)
        user_text = ctx.get("user_text")
        force_collage = ctx.get("force_collage", False)
        reply_msg = ctx.get("reply_msg")
        if not imgs or not reply_msg:
            return

        multiple = len(imgs) > 1
        per_image = True
        if force_collage:
            per_image = False

        if multiple and not per_image:
            # Build named images
            named = [(f"image_{i+1}.jpg", b) for i, (b, _, _, _) in enumerate(imgs)]
            # at-call resolution of functions to allow tests to monkeypatch
            from bot import make_collage, call_model_with_image, build_collage_system_prompt, send_response

            collage_bytes, file_names = make_collage(named, cfg.collage_max_size, cfg.collage_quality)

            if use_text_override:
                collage_system_prompt = ""
                user_text_for_call = user_text
            else:
                # If a prompt_path is stored, prefer re-reading it at processing time
                # so that the exact prompt file content is used.
                if ctx.get("prompt_path"):
                    try:
                        sp = ""
                        try:
                            with open(ctx.get("prompt_path"), "r", encoding="utf-8") as f:
                                sp = f.read()
                        except Exception:
                            sp = ""
                        if sp:
                            system_prompt = sp
                            prompt_label = os.path.splitext(os.path.basename(ctx.get("prompt_path")))[0]
                    except Exception:
                        pass
                collage_system_prompt = build_collage_system_prompt(system_prompt, file_names)
                user_text_for_call = None

            orientations = {f: orient for f, (_, _, _, orient) in zip(file_names, imgs)}
            collage_meta = {"mode": "collage", "orientations": orientations}

            status = None
            try:
                sent = await reply_msg.answer(f"📷 Принял {len(imgs)} файлов, объединил в коллаж и отправил модели.")
                if sent and hasattr(sent, "message_id"):
                    chat_id = getattr(getattr(sent, "chat", None), "id", None)
                    mid = getattr(sent, "message_id", None)
                    if chat_id and mid:
                        status = {"chat_id": chat_id, "message_id": mid}
            except Exception:
                try:
                    await send_response(reply_msg, f"📷 Принял {len(imgs)} файлов, объединил в коллаж и отправил модели.")
                except Exception:
                    pass

            resp = call_model_with_image(cfg, collage_bytes, system_prompt=collage_system_prompt, user_text=user_text_for_call, quiet=True, image_meta=collage_meta)
            if isinstance(resp, tuple):
                text_result, usage = resp
            else:
                text_result, usage = resp, None

            final_text = f"Коллаж — промт: {prompt_label}\n\n" + text_result
            # Try to edit kickoff status message if we have one; fall back to sending a fresh response
            if status:
                try:
                    await send_response(reply_msg, final_text, filename_prefix="images")
                except Exception:
                    pass
                try:
                    await __import__('bot').bot.edit_message_text(final_text, chat_id=status["chat_id"], message_id=status["message_id"])
                except Exception as e:
                    print(f"[MEDIA_DEBUG] failed to edit status message for media_group {mgid}: {e}", file=sys.stderr)
                    try:
                        notify_on_edit = os.getenv("NOTIFY_ON_EDIT_FAILURE", "0")
                        notify_ok = str(notify_on_edit).lower() in ("1", "true", "yes")
                        if notify_ok and getattr(__import__('bot'), 'BOT_ADMIN_ID', None):
                            user_id = getattr(getattr(reply_msg, "from_user", None), "id", None)
                            chat_id = getattr(getattr(reply_msg, "chat", None), "id", None)
                            msg_id = getattr(reply_msg, "message_id", None)
                            admin_text = (
                                "⚠️ Ошибка при редактировании статуса альбома\n"
                                f"media_group_id: {mgid}\n"
                                f"prompt_label: {prompt_label}\n"
                                f"user_id: {user_id}\n"
                                f"chat_id: {chat_id}\n"
                                f"message_id: {msg_id}\n"
                                f"Ошибка: {e}\n"
                            )
                            try:
                                await __import__('bot').bot.send_message(__import__('bot').BOT_ADMIN_ID, admin_text)
                            except Exception:
                                pass
                    except Exception:
                        pass
            else:
                try:
                    await send_response(reply_msg, final_text, filename_prefix="images")
                except Exception:
                    pass

            MEDIA_CONTEXTS.pop(mgid, None)
            return
        # Per-image processing
        from bot import call_model_with_image, send_response
        # send kickoff status message so we can edit later
        status = None
        try:
            sent = await reply_msg.answer(f"📷 Принял {len(imgs)} файлов и отправляю их на поштучный анализ.")
            if sent and hasattr(sent, "message_id"):
                chat_id = getattr(getattr(sent, "chat", None), "id", None)
                mid = getattr(sent, "message_id", None)
                if chat_id and mid:
                    status = {"chat_id": chat_id, "message_id": mid}
        except Exception:
            try:
                await send_response(reply_msg, f"📷 Принял {len(imgs)} файлов и отправляю их на поштучный анализ.")
            except Exception:
                pass

        aggregated_texts = []
        for i, (jpeg, final_w, final_h, orientation) in enumerate(imgs, start=1):
            if use_text_override:
                system_prompt_for_call = ""
                user_text_for_call = user_text
            else:
                system_prompt_for_call = system_prompt
                user_text_for_call = None
            image_meta = {"orientation": orientation, "width": final_w, "height": final_h}
            try:
                resp = call_model_with_image(cfg, jpeg, system_prompt=system_prompt_for_call, user_text=user_text_for_call, quiet=True, image_meta=image_meta)
                if isinstance(resp, tuple):
                    text_result, usage = resp
                else:
                    text_result, usage = resp, None
                header = f"Изображение #{i} — промт: {prompt_label}\n"
                aggregated_texts.append(header + text_result)
            except Exception:
                pass

        final_text = "\n\n".join(aggregated_texts)
        # Try to edit kickoff status message if we have one; fall back to sending a fresh response
        if status:
            try:
                await send_response(reply_msg, final_text, filename_prefix="images")
                try:
                    await __import__('bot').bot.edit_message_text(final_text, chat_id=status["chat_id"], message_id=status["message_id"])
                except Exception as e:
                    print(f"[MEDIA_DEBUG] failed to edit status message for media_group {mgid}: {e}", file=sys.stderr)
                    try:
                        notify_on_edit = os.getenv("NOTIFY_ON_EDIT_FAILURE", "0")
                        notify_ok = str(notify_on_edit).lower() in ("1", "true", "yes")
                        if notify_ok and getattr(__import__('bot'), 'BOT_ADMIN_ID', None):
                            user_id = getattr(getattr(reply_msg, "from_user", None), "id", None)
                            chat_id = getattr(getattr(reply_msg, "chat", None), "id", None)
                            msg_id = getattr(reply_msg, "message_id", None)
                            admin_text = (
                                "⚠️ Ошибка при редактировании статуса альбома\n"
                                f"media_group_id: {mgid}\n"
                                f"prompt_label: {prompt_label}\n"
                                f"user_id: {user_id}\n"
                                f"chat_id: {chat_id}\n"
                                f"message_id: {msg_id}\n"
                                f"Ошибка: {e}\n"
                            )
                            try:
                                await __import__('bot').bot.send_message(__import__('bot').BOT_ADMIN_ID, admin_text)
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            try:
                await send_response(reply_msg, final_text, filename_prefix="images")
            except Exception:
                pass

        MEDIA_CONTEXTS.pop(mgid, None)

    except Exception:
        try:
            media_debug = str(os.environ.get("DEBUG", os.environ.get("IMAGE_DEBUG", ""))).lower() in ("1", "true", "yes")
        except Exception:
            media_debug = False
        if media_debug:
            print(f"[MEDIA_DEBUG] unexpected error while processing media_group {mgid}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
