from typing import Optional
import os
import base64
from .image_io import resize_image_from_bytes, load_and_resize_image, make_collage_wrapper
from .image_processing import build_collage_system_prompt
from .config import read_prompt_file


def handle_json_request(req: dict) -> dict:
    # If there's no override_text, ensure API key exists
    if req.get("override_text") is None and not os.environ.get("OPENAI_API_KEY"):
        return {"ok": False, "errors": ["OPENAI_API_KEY is not set"]}

    action = req.get("action")
    if action != "analyze":
        return {"ok": False, "errors": ["unsupported action"]}

    override_text = req.get("override_text")
    include_billing = bool(req.get("include_billing", False))

    if override_text is not None:
        try:
            # Import from package so tests can monkeypatch call_model_with_text_only
            from ai_image_analyzer import call_model_with_text_only, load_config

            result = call_model_with_text_only(load_config(), override_text, system_prompt="", quiet=True)
        except Exception as e:
            return {"ok": False, "errors": [str(e)]}
        resp = {"ok": True, "results": result}
        if include_billing:
            resp["usage"] = None
        return resp

    # Images handling
    images = req.get("images")
    if images is not None:
        if not isinstance(images, list):
            return {"ok": False, "errors": ["images must be a list"]}
        if len(images) == 0:
            return {"ok": False, "errors": ["no images provided"]}
        if len(images) > 10:
            return {"ok": False, "errors": ["too many images: max 10"]}

        per_image = bool(req.get("per_image", False))
        cfg = load_config()

        processed = []  # list of tuples (name, jpeg_bytes)
        for idx, itm in enumerate(images, start=1):
            name = itm.get("name") or f"image_{idx}.jpg"
            if itm.get("data_b64") is not None:
                try:
                    b = base64.b64decode(itm.get("data_b64"))
                except Exception as e:
                    return {"ok": False, "errors": [f"invalid base64 for image {name}: {e}"]}
                try:
                    jb, w, h, orient = resize_image_from_bytes(b, cfg.image_max_size, cfg.image_quality)
                except Exception as e:
                    return {"ok": False, "errors": [f"failed to process image {name}: {e}"]}
            elif itm.get("path") is not None:
                try:
                    jb, w, h, orient = load_and_resize_image(itm.get("path"), cfg.image_max_size, cfg.image_quality)
                except Exception as e:
                    return {"ok": False, "errors": [f"failed to load image {name} from path: {e}"]}
            else:
                return {"ok": False, "errors": [f"image {name} missing data_b64 or path"]}
            processed.append((name, jb))

        # per-image: call model for each
        results = []
        total_usage = {"total_cost": 0.0}
        if per_image:
            for name, jb in processed:
                try:
                    resp = None
                    from ai_image_analyzer import call_model_with_image

                    resp = call_model_with_image(load_config(), jb, system_prompt="", user_text=None, quiet=True, image_meta={"name": name})
                    if isinstance(resp, tuple):
                        text, usage = resp
                    else:
                        text, usage = resp, None
                except Exception as e:
                    return {"ok": False, "errors": [str(e)]}
                results.append({"name": name, "text": text})
                if include_billing and usage:
                    try:
                        total_usage["total_cost"] += float(usage.get("total_cost", 0.0) or 0.0)
                    except Exception:
                        pass
            resp = {"ok": True, "mode": "per_image", "results": results}
            if include_billing:
                resp["usage"] = total_usage
            return resp

        # collage mode: combine and send as single request
        try:
            collage_bytes, filenames = make_collage_wrapper(processed, cfg.collage_max_size, cfg.collage_quality)
        except Exception as e:
            return {"ok": False, "errors": [f"failed to make collage: {e}"]}

        # prepare system prompt based on prompt_file
        sys_prompt = ""
        if cfg.prompt_file:
            sys_prompt = read_prompt_file(cfg.prompt_file) or ""
            sys_prompt = build_collage_system_prompt(sys_prompt, filenames)
        try:
            from ai_image_analyzer import call_model_with_image

            resp = call_model_with_image(cfg, collage_bytes, system_prompt=sys_prompt, user_text=None, quiet=True, image_meta={"mode": "collage", "filenames": filenames})
            if isinstance(resp, tuple):
                text, usage = resp
            else:
                text, usage = resp, None
        except Exception as e:
            return {"ok": False, "errors": [str(e)]}

        out = {"ok": True, "mode": "collage", "results": text}
        if include_billing:
            out["usage"] = usage
        return out

    return {"ok": False, "errors": ["no override_text provided"]}
