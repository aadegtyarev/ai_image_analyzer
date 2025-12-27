#!/usr/bin/env python3
"""
ai_image_analyzer

Скрипт для анализа одиночных изображений или коллажей
через совместимую с OpenAI API модель (в т.ч. vsegpt).

Поддержка:
- .env конфиг (OPENAI_*, IMAGE_*, PROMPT_FILE, COLLAGE_*).
- Переопределение этих параметров через CLI.
- Режимы:
  * одна картинка → один запрос, результат в *_analyse.md;
  * несколько картинок (до 10) → коллаж, один запрос, group_..._analyse.md;
  * override-текст (-t/--text) вместо системного промта → печать только в stdout;
  * текст без картинок → текстовый запрос к модели, результат в stdout.
- Логи и статистика по usage (токены, стоимость).
- Запрос баланса провайдера (--check-balance).
"""

import argparse
import base64
import glob
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple

import requests
from PIL import Image, ImageOps  # pillow
try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # dotenv is optional
    def load_dotenv(*args, **kwargs):
        return False

try:
    from openai import OpenAI, BadRequestError
except ImportError:
    OpenAI = None  # type: ignore

    class BadRequestError(Exception):
        pass


# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------


@dataclass
class Config:
    api_key: str
    base_url: Optional[str]
    model: str
    timeout: int
    max_tokens: int
    image_max_size: int
    image_quality: int
    collage_max_size: int
    collage_quality: int
    prompt_file: Optional[str]
    image_debug: bool = False


def load_config() -> Config:
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY is not set")

    base_url = os.environ.get("OPENAI_BASE_URL") or None
    model = os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"

    def _int_env(name: str, default: int) -> int:
        v = os.environ.get(name)
        if not v:
            return default
        try:
            return int(v)
        except ValueError:
            print(f"WARNING: invalid {name}={v!r}, using default {default}", file=sys.stderr)
            return default

    timeout = _int_env("OPENAI_TIMEOUT", 120)
    max_tokens = _int_env("OPENAI_MAX_TOKENS", 1024)
    image_max_size = _int_env("IMAGE_MAX_SIZE", 1024)
    image_quality = _int_env("IMAGE_QUALITY", 90)
    collage_max_size = _int_env("COLLAGE_MAX_SIZE", 2048)
    collage_quality = _int_env("COLLAGE_QUALITY", 90)

    prompt_file = os.environ.get("PROMPT_FILE")
    # Unified debug flag: check DEBUG first, fall back to IMAGE_DEBUG for backward compatibility
    image_debug_env = os.environ.get("DEBUG", None)
    if image_debug_env is None:
        image_debug_env = os.environ.get("IMAGE_DEBUG", "")
    image_debug = str(image_debug_env).lower() in ("1", "true", "yes")

    return Config(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
        image_max_size=image_max_size,
        image_quality=image_quality,
        collage_max_size=collage_max_size,
        collage_quality=collage_quality,
        prompt_file=prompt_file,
        image_debug=image_debug,
    )


def read_prompt_file(path: Optional[str]) -> str:
    # Non-fatal: if no path is provided or the file cannot be read, return an
    # empty string and log a warning instead of exiting the process. The bot
    # will handle an empty system prompt gracefully.
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError as e:
        print(f"ERROR: failed to read PROMPT_FILE={path!r}: {e}", file=sys.stderr)
        return ""


def load_config_from_env(env: dict):
    """Backward-compatible loader used by tests.

    Returns a simple object with attributes `context_size`, `max_tokens`, and `x_title`.
    """
    class SimpleCfg:
        pass

    cfg = SimpleCfg()
    try:
        context_size = int(env.get("OPENAI_CONTEXT_SIZE", 2048))
    except Exception:
        context_size = 2048
    try:
        max_tokens = int(env.get("OPENAI_MAX_TOKENS", 1024))
    except Exception:
        max_tokens = 1024
    # cap max_tokens to context_size - 1
    if max_tokens > max(0, context_size - 1):
        max_tokens = max(0, context_size - 1)
    cfg.context_size = context_size
    cfg.max_tokens = max_tokens
    cfg.x_title = env.get("OPENAI_X_TITLE")
    return cfg


def handle_json_request(req: dict) -> dict:
    """Minimal JSON API handler for tests and integrations.

    Supports action 'analyze' with optional 'override_text' and 'include_billing'.
    """
    # basic validation: if there's no override_text we need an API key
    if req.get("override_text") is None and not os.environ.get("OPENAI_API_KEY"):
        return {"ok": False, "errors": ["OPENAI_API_KEY is not set"]}

    action = req.get("action")
    if action != "analyze":
        return {"ok": False, "errors": ["unsupported action"]}

    override_text = req.get("override_text")
    include_billing = bool(req.get("include_billing", False))

    if override_text is not None:
        # call text-only path
        try:
            result, usage = call_model_with_text_only(load_config(), override_text, system_prompt="", quiet=True)
        except Exception as e:
            return {"ok": False, "errors": [str(e)]}
        resp = {"ok": True, "results": result}
        if include_billing:
            resp["usage"] = usage
        return resp

    # For now, other modes are not implemented in tests
    return {"ok": False, "errors": ["no override_text provided"]}


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def log(msg: str, quiet: bool) -> None:
    """Простой префиксованный лог."""
    if not quiet:
        print(f"[ai_image_analyzer] {msg}", file=sys.stderr)


def expand_image_patterns(patterns: Sequence[str]) -> List[str]:
    """Разворачивает маски (glob)."""
    result: List[str] = []
    for p in patterns:
        expanded = glob.glob(p)
        if expanded:
            result.extend(sorted(expanded))
        else:
            result.append(p)
    # Убираем дубликаты, сохраняя порядок
    seen = set()
    uniq: List[str] = []
    for p in result:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def load_and_resize_image(path: str, max_size: int, jpeg_quality: int) -> tuple[bytes, int, int, str]:
    """
    Открыть картинку, привести по меньшей стороне к max_size,
    сохранить в JPEG с качеством jpeg_quality и вернуть байты.
    """
    try:
        with Image.open(path) as img:
            # Учитываем EXIF-ориентацию (поворачиваем изображение к корректному отображению)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            w, h = img.size
            # Ориентация
            if h > w:
                orientation = "portrait"
            elif w > h:
                orientation = "landscape"
            else:
                orientation = "square"
            scale = min(max_size / max(w, h), 1.0)
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            # Сохраняем итоговые размеры (после ресайза)
            final_w, final_h = img.size
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            return buf.getvalue(), final_w, final_h, orientation
    except Exception as e:
        raise RuntimeError(f"Failed to load/resize image {path!r}: {e}") from e


def image_bytes_to_data_url(jpeg_bytes: bytes) -> str:
    """Кодирует JPEG-байты в data: URL для передачи в vision-модель."""
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def make_collage(
    images: List[Tuple[str, bytes]],
    max_size: int,
    jpeg_quality: int,
) -> Tuple[bytes, List[str]]:
    """
    Делает квадратный коллаж из уже уменьшенных JPEG-байтов.
    Возвращает (jpeg_bytes_collage, filenames_in_order).
    """
    if not images:
        raise ValueError("make_collage: empty images list")

    filenames = [os.path.basename(p) for p, _ in images]

    decoded: List[Image.Image] = []
    for _, b in images:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        decoded.append(img)

    n = len(decoded)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    first_w, first_h = decoded[0].size
    tile = max(first_w, first_h)

    canvas_size = min(max_size, tile * max(cols, rows))
    tile_size = canvas_size // max(cols, rows)
    canvas = Image.new("RGB", (canvas_size, canvas_size), color=(0, 0, 0))

    for idx, img in enumerate(decoded):
        col = idx % cols
        row = idx // cols
        x = col * tile_size
        y = row * tile_size
        img_resized = img.resize((tile_size, tile_size), Image.LANCZOS)
        canvas.paste(img_resized, (x, y))

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return buf.getvalue(), filenames


def build_collage_system_prompt(base_prompt: str, filenames: List[str]) -> str:
    """
    Добавляет к системному промту описание того, что это коллаж,
    и перечисляет имена файлов в порядке слева-направо, сверху-вниз.
    Подходит и для одного файла, и для нескольких.
    """
    files_str = ", ".join(filenames)
    prefix = (
        "Мы отправляем общий коллаж. "
        "Каждая ячейка коллажа соответствует отдельному исходному файлу. "
        f"Порядок файлов слева направо, сверху вниз: {files_str}.\n\n"
    )
    return prefix + base_prompt


# ---------------------------------------------------------------------------
# Клиент модели + usage / баланс
# ---------------------------------------------------------------------------


def build_client(cfg: Config) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")

    kwargs: dict = {"api_key": cfg.api_key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return OpenAI(**kwargs)  # type: ignore[arg-type]


def normalize_usage(usage: Any) -> Optional[dict]:
    """
    Normalize usage object from OpenAI / vsegpt into a plain dict.
    Returns dict or None.
    """
    if usage is None:
        return None
    if isinstance(usage, dict):
        return dict(usage)
    # Pydantic-style objects (OpenAI 1.x)
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass
    if hasattr(usage, "to_dict"):
        try:
            return usage.to_dict()
        except Exception:
            pass
    return None


def normalize_usage_for_api(usage: Any) -> Optional[dict]:
    """
    Same as normalize_usage, но с округлением total_cost до 3 знаков.
    Удобно для передачи в бота.
    """
    raw = normalize_usage(usage)
    if raw is None:
        return None
    out = dict(raw)
    if "total_cost" in out:
        try:
            out["total_cost"] = round(float(out["total_cost"]), 3)
        except Exception:
            pass
    return out


def log_usage(usage: Optional[dict], quiet: bool) -> None:
    """Лаконичный лог по токенам/стоимости."""
    if not usage:
        return
    parts = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "total_cost"):
        if key in usage:
            parts.append(f"{key}={usage[key]}")
    if parts:
        log("Использование модели: " + ", ".join(parts), quiet)


def check_balance(cfg: Config, quiet: bool) -> dict:
    """
    Запрос баланса у провайдера (например, vsegpt: GET /balance).
    Требует установленного OPENAI_BASE_URL.
    """
    if not cfg.base_url:
        raise RuntimeError("OPENAI_BASE_URL is not set; cannot check balance.")
    url = cfg.base_url.rstrip("/") + "/balance"
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    log(f"Запрос баланса: GET {url}", quiet)
    resp = requests.get(url, headers=headers, timeout=cfg.timeout)
    resp.raise_for_status()
    data = resp.json()
    log("Баланс получен успешно", quiet)
    return data


def call_model_with_image(
    cfg: Config,
    jpeg_bytes: bytes,
    system_prompt: str,
    user_text: Optional[str],
    quiet: bool,
    image_meta: Optional[dict] = None,
) -> Tuple[str, Optional[dict]]:
    """
    Вызов модели с одной картинкой (или коллажем).
    Возвращает (text, usage_dict).
    """
    client = build_client(cfg)
    image_url = image_bytes_to_data_url(jpeg_bytes)

    content: List[dict] = []
    if user_text:
        content.append({"type": "text", "text": user_text})
    # Добавляем мета-информацию об изображении (ориентация, размеры и т.д.)
    if image_meta:
        try:
            if image_meta.get("mode") == "collage":
                # кратко опишем ориентации
                orients = image_meta.get("orientations", {})
                meta_parts = [f"{name} — {orient}" for name, orient in orients.items()]
                meta_text = (
                    "Примечание для модели: это коллаж. Ориентации по файлам: "
                    + ", ".join(meta_parts)
                    + ". Учитывайте ориентацию при анализе изображения."
                )
            else:
                meta_text = (
                    "Примечание для модели: ориентация изображения — "
                    f"{image_meta.get('orientation')}; размер: {image_meta.get('width')}×{image_meta.get('height')}. "
                    "Пожалуйста, учитывайте ориентацию при интерпретации сцены (например, объект может стоять, а не лежать)."
                )
        except Exception:
            meta_text = "Инфо об изображении: (unknown)"
        content.append({"type": "text", "text": meta_text})
        # debug printing controlled by env var IMAGE_DEBUG
        debug_on = getattr(cfg, "image_debug", False)
        if debug_on:
            print(f"[IMAGE_DEBUG] meta_text: {meta_text}", file=sys.stderr)
            # print a snippet of the image_url (length) to avoid huge dumps
            print(f"[IMAGE_DEBUG] image_url (len): {len(image_url)}", file=sys.stderr)
    content.append({"type": "image_url", "image_url": {"url": image_url}})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    # Prompt debug: respect unified DEBUG or fallback to IMAGE_DEBUG
    try:
        env_dbg = os.environ.get("DEBUG", None)
        if env_dbg is None:
            env_dbg = os.environ.get("IMAGE_DEBUG", "")
        prompt_debug = str(env_dbg).lower() in ("1", "true", "yes")
    except Exception:
        prompt_debug = False
    if prompt_debug:
        try:
            sys_snip = (system_prompt or "")[:200]
            usr_snip = (user_text or "")[:200]
            print(f"[PROMPT_DEBUG] system_snip={sys_snip!r} user_snip={usr_snip!r}", file=sys.stderr)
            # reuse existing sanitizer to show messages payload
            def _san(m):
                if isinstance(m.get("content"), list):
                    parts = []
                    for c in m["content"]:
                        t = c.get("type")
                        if t == "image_url":
                            url = c.get("image_url", {}).get("url", "")
                            parts.append({"type": "image_url", "len": len(url)})
                        else:
                            txt = c.get("text") or ""
                            parts.append({"type": t, "text_snip": txt[:200]})
                    return {"role": m.get("role"), "content": parts}
                return m

            s_msgs = [(_san(m)) for m in messages]
            print(f"[PROMPT_DEBUG] messages payload: {json.dumps(s_msgs, ensure_ascii=False)}", file=sys.stderr)
        except Exception:
            print("[PROMPT_DEBUG] failed to serialize messages payload", file=sys.stderr)

    log(
        f"Запрос к модели (image): model={cfg.model}, "
        f"system_len={len(system_prompt)}, has_user_text={bool(user_text)}",
        quiet,
    )

    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
        )
    except BadRequestError as e:
        msg = str(e)
        if "Exceeded soft user limit per query" in msg:
            explanation = (
                "Запрос отклонён провайдером: превышен мягкий лимит стоимости на один запрос.\n"
                "Сообщение сервиса:\n"
                f"{msg}\n\n"
                "Уменьшите размер запроса (меньше изображений, короче промт) "
                "или увеличьте лимит на стороне провайдера (в настройках аккаунта)."
            )
            log("Ошибка лимита стоимости запроса (soft user limit)", quiet)
            return explanation, None
        log(f"BadRequestError от модели: {msg}", quiet)
        raise

    # Доп. отладочная печать: показать сокращённый messages, если включён IMAGE_DEBUG
    debug_on = getattr(cfg, "image_debug", False)
    if debug_on:
        def sanitize_message(m):
            if isinstance(m.get("content"), list):
                parts = []
                for c in m["content"]:
                    t = c.get("type")
                    if t == "image_url":
                        url = c.get("image_url", {}).get("url", "")
                        parts.append({"type": "image_url", "len": len(url)})
                    else:
                        txt = c.get("text") or ""
                        parts.append({"type": t, "text_snip": txt[:200]})
                return {"role": m.get("role"), "content": parts}
            return m

        try:
            s_msgs = [sanitize_message(m) for m in messages]
            print(f"[IMAGE_DEBUG] messages payload: {json.dumps(s_msgs, ensure_ascii=False)}", file=sys.stderr)
        except Exception:
            print("[IMAGE_DEBUG] failed to serialize messages payload", file=sys.stderr)

    text = resp.choices[0].message.content or ""
    usage = normalize_usage_for_api(getattr(resp, "usage", None))
    log("Ответ модели (image) получен", quiet)
    log_usage(usage, quiet)
    return text, usage


def call_model_with_text_only(
    cfg: Config,
    text: str,
    system_prompt: str,
    quiet: bool,
) -> Tuple[str, Optional[dict]]:
    """
    Вызов модели без изображения.
    Возвращает (text, usage_dict).
    """
    client = build_client(cfg)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})

    log(
        f"Запрос к модели (text): model={cfg.model}, "
        f"system_len={len(system_prompt)}, text_len={len(text)}",
        quiet,
    )

    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
        )
    except BadRequestError as e:
        msg = str(e)
        if "Exceeded soft user limit per query" in msg:
            explanation = (
                "Запрос отклонён провайдером: превышен мягкий лимит стоимости на один запрос.\n"
                "Сообщение сервиса:\n"
                f"{msg}\n\n"
                "Уменьшите размер запроса (короче текст или промт) "
                "или увеличьте лимит на стороне провайдера (в настройках аккаунта)."
            )
            log("Ошибка лимита стоимости запроса (soft user limit)", quiet)
            return explanation, None
        log(f"BadRequestError от модели: {msg}", quiet)
        raise

    result = resp.choices[0].message.content or ""
    usage = normalize_usage_for_api(getattr(resp, "usage", None))
    log("Ответ модели (text) получен", quiet)
    log_usage(usage, quiet)
    return result, usage


# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------


def save_result_for_single_image(image_path: str, text: str) -> str:
    base, _ext = os.path.splitext(image_path)
    out_path = f"{base}_analyse.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def save_result_for_group(image_paths: List[str], text: str) -> str:
    """
    Сохраняет результат для группы картинок.
    Имя файла включает префикс group и timestamp.
    """
    folder = os.path.dirname(image_paths[0]) if image_paths else "."
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(folder, f"group_{ts}_analyse.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai_image_analyzer",
        description="Analyze images or collages with an OpenAI-compatible vision model.",
    )
    parser.add_argument(
        "images",
        nargs="*",
        help=(
            "Image file paths (supports glob masks like '*.jpg'). "
            "If omitted, the request will be text-only."
        ),
    )
    parser.add_argument(
        "-t",
        "--text",
        dest="text",
        help=(
            "Optional text. "
            "If used together with images, overrides PROMPT_FILE and is sent as user text "
            "along with the image/collage; result is printed only and not saved. "
            "If used without images, a pure text request is sent and printed."
        ),
    )
    parser.add_argument(
        "-p",
        "--prompt-file",
        dest="prompt_file",
        help=(
            "Override PROMPT_FILE from .env (path to system prompt file). "
            "Used only in modes where system prompt is required."
        ),
    )
    parser.add_argument(
        "--image-max-size",
        dest="image_max_size",
        type=int,
        help="Override IMAGE_MAX_SIZE from .env (max side in pixels for individual images).",
    )
    parser.add_argument(
        "--image-quality",
        dest="image_quality",
        type=int,
        help="Override IMAGE_QUALITY from .env (JPEG quality 1-100 for resized images).",
    )
    parser.add_argument(
        "--collage-max-size",
        dest="collage_max_size",
        type=int,
        help="Override COLLAGE_MAX_SIZE from .env (max side in pixels for collage).",
    )
    parser.add_argument(
        "--collage-quality",
        dest="collage_quality",
        type=int,
        help="Override COLLAGE_QUALITY from .env (JPEG quality 1-100 for collage).",
    )
    # OpenAI / provider overrides
    parser.add_argument(
        "-k",
        "--OPENAI_API_KEY",
        dest="openai_api_key",
        help="Override OPENAI_API_KEY from .env.",
    )
    parser.add_argument(
        "-u",
        "--OPENAI_BASE_URL",
        dest="openai_base_url",
        help="Override OPENAI_BASE_URL from .env.",
    )
    parser.add_argument(
        "-m",
        "--OPENAI_MODEL",
        dest="openai_model",
        help="Override OPENAI_MODEL from .env.",
    )
    parser.add_argument(
        "-T",
        "--OPENAI_TIMEOUT",
        dest="openai_timeout",
        type=int,
        help="Override OPENAI_TIMEOUT from .env (seconds).",
    )
    parser.add_argument(
        "-M",
        "--OPENAI_MAX_TOKENS",
        dest="openai_max_tokens",
        type=int,
        help="Override OPENAI_MAX_TOKENS from .env.",
    )
    parser.add_argument(
        "--check-balance",
        dest="check_balance",
        action="store_true",
        help="Check provider balance (if supported) and exit.",
    )
    parser.add_argument(
        "--image-debug",
        dest="image_debug",
        action="store_true",
        help="Enable image debug output to stderr (overrides IMAGE_DEBUG env).",
    )
    parser.add_argument(
        "--debug",
        dest="image_debug",
        action="store_true",
        help="Enable image debug output to stderr (alias for --image-debug).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode: suppress informational logs (старое поведение скрипта).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config()

    # CLI overrides for config
    if getattr(args, "openai_api_key", None):
        cfg.api_key = args.openai_api_key
    if getattr(args, "openai_base_url", None):
        cfg.base_url = args.openai_base_url
    if getattr(args, "openai_model", None):
        cfg.model = args.openai_model
    if getattr(args, "openai_timeout", None) is not None:
        cfg.timeout = args.openai_timeout
    if getattr(args, "openai_max_tokens", None) is not None:
        cfg.max_tokens = args.openai_max_tokens
    if getattr(args, "prompt_file", None):
        cfg.prompt_file = args.prompt_file
    if getattr(args, "image_max_size", None) is not None:
        cfg.image_max_size = args.image_max_size
    if getattr(args, "image_quality", None) is not None:
        cfg.image_quality = args.image_quality
    if getattr(args, "collage_max_size", None) is not None:
        cfg.collage_max_size = args.collage_max_size
    if getattr(args, "collage_quality", None) is not None:
        cfg.collage_quality = args.collage_quality
    if getattr(args, "image_debug", False):
        cfg.image_debug = True

    # init logs
    log("Инициализация...", args.quiet)
    log(f"Модель: {cfg.model}", args.quiet)
    log(f"BASE_URL: {cfg.base_url or 'default'}", args.quiet)
    log(f"MAX_TOKENS: {cfg.max_tokens}", args.quiet)
    log(f"TIMEOUT: {cfg.timeout}", args.quiet)
    log(
        f"IMAGE_MAX_SIZE={cfg.image_max_size}, IMAGE_QUALITY={cfg.image_quality}",
        args.quiet,
    )
    log(
        f"COLLAGE_MAX_SIZE={cfg.collage_max_size}, COLLAGE_QUALITY={cfg.collage_quality}",
        args.quiet,
    )
    if cfg.prompt_file:
        log(f"PROMPT_FILE: {cfg.prompt_file}", args.quiet)

    # Balance mode
    if getattr(args, "check_balance", False):
        try:
            data = check_balance(cfg, args.quiet)
        except Exception as e:  # pragma: no cover - сетевые ошибки
            print(f"Failed to get balance: {e}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    image_paths = expand_image_patterns(args.images)
    override_text: Optional[str] = args.text

    log(f"Получено файлов: {len(image_paths)}", args.quiet)
    if image_paths:
        for p in image_paths:
            log(f" → {p}", args.quiet)

    if not image_paths and not override_text:
        print("Nothing to do: no images and no text provided.", file=sys.stderr)
        sys.exit(1)

    # TEXT-ONLY MODE
    if not image_paths and override_text:
        result, usage = call_model_with_text_only(
            cfg,
            override_text,
            system_prompt="",
            quiet=args.quiet,
        )
        print(result)
        return

    # IMAGE MODES
    if len(image_paths) > 10:
        log(f"Ограничиваем количество файлов до 10 (из {len(image_paths)})", args.quiet)
        image_paths = image_paths[:10]

    resized: List[Tuple[str, bytes, str, tuple]] = []  # path, bytes, orientation, (w,h)
    for p in image_paths:
        try:
            jpeg_bytes, final_w, final_h, orientation = load_and_resize_image(
                p, cfg.image_max_size, cfg.image_quality
            )
        except RuntimeError as e:
            log(str(e), args.quiet)
            continue
        resized.append((p, jpeg_bytes, orientation, (final_w, final_h)))

    if not resized:
        print("No valid images after resize step.", file=sys.stderr)
        sys.exit(1)

    # Override-text mode (no system prompt)
    if override_text:
        if len(resized) == 1:
            _, jpeg_bytes, orientation, (final_w, final_h) = resized[0]
            log("Режим: одна картинка + override текст", args.quiet)
        else:
            log("Режим: несколько картинок + override текст → создаём коллаж", args.quiet)
            collage_bytes, _ = make_collage(
                [(p, b) for p, b, *_ in resized], cfg.collage_max_size, cfg.collage_quality
            )
            jpeg_bytes = collage_bytes
        # Передадим image_meta для лучшей интерпретации (ориентация/размеры)
        if len(resized) == 1:
            image_meta = {"orientation": orientation, "width": final_w, "height": final_h}
        else:
            # для коллажа соберём ориентации
            orientations = {os.path.basename(p): o for p, _, o, _ in resized}
            image_meta = {"mode": "collage", "orientations": orientations}

        result, usage = call_model_with_image(
            cfg,
            jpeg_bytes,
            system_prompt="",
            user_text=override_text,
            quiet=args.quiet,
            image_meta=image_meta,
        )
        print(result)
        return

    # Normal mode: PROMPT_FILE → system
    system_prompt = read_prompt_file(cfg.prompt_file)
    log(f"Системный промт загружен, длина={len(system_prompt)}", args.quiet)

    if len(resized) == 1:
        img_path, jpeg_bytes, orientation, (final_w, final_h) = resized[0]
        log("Режим: одна картинка, без override-текста", args.quiet)
        # Передаём метаданные изображения модели
        image_meta = {"orientation": orientation, "width": final_w, "height": final_h}
        result, usage = call_model_with_image(
            cfg,
            jpeg_bytes,
            system_prompt=system_prompt,
            user_text=None,
            quiet=args.quiet,
            image_meta=image_meta,
        )
        out_path = save_result_for_single_image(img_path, result)
        # Поведение stdout как раньше
        print(f"Saved analysis to {out_path}")
    else:
        log("Режим: несколько картинок → создаём коллаж", args.quiet)
        collage_bytes, filenames = make_collage(
            [(p, b) for p, b, *_ in resized],
            cfg.collage_max_size,
            cfg.collage_quality,
        )
        collage_system_prompt = build_collage_system_prompt(system_prompt, filenames)
        log(
            f"Коллаж готов, файлов в описании: {len(filenames)}",
            args.quiet,
        )
        # собираем мета-информацию по ориентации для всех изображений
        orientations = {os.path.basename(p): o for p, _, o, _ in resized}
        image_meta = {"mode": "collage", "orientations": orientations}
        result, usage = call_model_with_image(
            cfg,
            collage_bytes,
            system_prompt=collage_system_prompt,
            user_text=None,
            quiet=args.quiet,
            image_meta=image_meta,
        )
        original_paths = [p for p, _ in resized]
        out_path = save_result_for_group(original_paths, result)
        print(f"Saved group analysis to {out_path}")


if __name__ == "__main__":
    main()
