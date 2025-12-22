#!/usr/bin/env python3
"""
ai_image_analyzer: analyze single images or collages with an OpenAI-compatible vision model.

Config via .env (может быть переопределён CLI-параметрами):
- OPENAI_API_KEY      (required)       / --OPENAI_API_KEY, -k
- OPENAI_BASE_URL     (optional)       / --OPENAI_BASE_URL, -u
- OPENAI_MODEL        (required)       / --OPENAI_MODEL, -m
- OPENAI_TIMEOUT      (optional,int)   / --OPENAI_TIMEOUT, -T
- OPENAI_MAX_TOKENS   (optional,int)   / --OPENAI_MAX_TOKENS, -M
- IMAGE_MAX_SIZE      (optional,int)   / --IMAGE_MAX_SIZE, -s
- IMAGE_QUALITY       (optional,int)   / --IMAGE_QUALITY, -Q
- COLLAGE_MAX_SIZE    (optional,int)   / --COLLAGE_MAX_SIZE, -S
- COLLAGE_QUALITY     (optional,int)   / --COLLAGE_QUALITY, -G
- PROMPT_FILE         (required в обычном режиме) / --PROMPT_FILE, -p
"""

import argparse
import base64
import glob
import io
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from PIL import Image  # pillow
try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # dotenv is optional
    def load_dotenv(*args, **kwargs):
        return False

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


# ----------------- logging helpers ----------------- #

def log(msg: str, quiet: bool) -> None:
    """Simple stderr logger, disabled when quiet=True."""
    if not quiet:
        print(f"[ai_image_analyzer] {msg}", file=sys.stderr)


def human_size(n: int) -> str:
    """Human readable size in KB/MB."""
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    return f"{n / 1024:.1f} KB"


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


def load_config() -> Config:
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY is not set")

    model = os.environ.get("OPENAI_MODEL")
    if not model:
        sys.exit("ERROR: OPENAI_MODEL is not set")

    def _int_env(name: str, default: int) -> int:
        val = os.environ.get(name)
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            sys.exit(f"ERROR: {name} must be an integer, got {val!r}")

    timeout = _int_env("OPENAI_TIMEOUT", 60)
    max_tokens = _int_env("OPENAI_MAX_TOKENS", 1024)
    image_max_size = _int_env("IMAGE_MAX_SIZE", 1024)
    image_quality = _int_env("IMAGE_QUALITY", 85)

    collage_max_size = _int_env("COLLAGE_MAX_SIZE", image_max_size)
    collage_quality = _int_env("COLLAGE_QUALITY", image_quality)

    prompt_file = os.environ.get("PROMPT_FILE")
    base_url = os.environ.get("OPENAI_BASE_URL") or None

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
    )


def read_prompt_file(path: Optional[str]) -> str:
    if not path:
        sys.exit("ERROR: PROMPT_FILE is not set but required in this mode")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError as e:
        sys.exit(f"ERROR: failed to read PROMPT_FILE %r: %s" % (path, e))


def expand_image_patterns(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and os.path.isfile(pattern):
            matches.append(pattern)
        paths.extend(matches)
    # unique preserving order
    seen = set()
    unique_paths: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths


def load_and_resize_image(
    path: str,
    max_size: int,
    jpeg_quality: int,
) -> Tuple[bytes, int, int, int, int, int, int]:
    """
    Open image, resize so longest side <= max_size,
    return (JPEG bytes, size_before, size_after, orig_w, orig_h, new_w, new_h).
    """
    size_before = os.path.getsize(path)

    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        orig_w, orig_h = w, h

        scale = min(1.0, float(max_size) / max(w, h))
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
            w, h = img.size  # обновляем под новые размеры

        new_w, new_h = w, h

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        data = buf.getvalue()

    size_after = len(data)
    return data, size_before, size_after, orig_w, orig_h, new_w, new_h


def make_collage(
    images: List[Tuple[str, bytes]],
    max_size: int,
    jpeg_quality: int,
    quiet: bool,
) -> Tuple[bytes, List[str]]:
    """
    Build a simple square-ish collage from pre-resized JPEG bytes.

    Returns (jpeg_bytes, ordered_filenames).
    """
    if not images:
        raise ValueError("No images for collage")

    log(
        f"Коллаж: входных изображений={len(images)}, "
        f"max_size={max_size}, jpeg_quality={jpeg_quality}",
        quiet,
    )

    pil_images: List[Image.Image] = []
    filenames: List[str] = []
    for idx, (name, data) in enumerate(images, start=1):
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        w, h = img.size
        fname = os.path.basename(name)
        log(f"Коллаж: изображение #{idx} {fname} {w}x{h}", quiet)
        pil_images.append(img)
        filenames.append(fname)

    n = len(pil_images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    tile_size = max_size // max(cols, rows)
    tile_size = max(tile_size, 1)

    collage_width = cols * tile_size
    collage_height = rows * tile_size
    log(
        f"Коллаж: сетка {cols}x{rows}, tile_size={tile_size}, "
        f"итоговый размер полотна {collage_width}x{collage_height}",
        quiet,
    )

    collage = Image.new("RGB", (collage_width, collage_height), color=(0, 0, 0))

    for idx, img in enumerate(pil_images):
        w, h = img.size
        scale = min(float(tile_size) / w, float(tile_size) / h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resized = img.resize(new_size, Image.LANCZOS)

        row = idx // cols
        col = idx % cols
        x0 = col * tile_size + (tile_size - resized.size[0]) // 2
        y0 = row * tile_size + (tile_size - resized.size[1]) // 2
        collage.paste(resized, (x0, y0))

    buf = io.BytesIO()
    collage.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    data = buf.getvalue()
    log(
        f"Коллаж: JPEG размер {human_size(len(data))} "
        f"({len(data)} bytes)",
        quiet,
    )
    return data, filenames


def image_bytes_to_data_url(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_collage_system_prompt(base_prompt: str, filenames: Sequence[str]) -> str:
    if not filenames:
        return base_prompt
    files_str = ", ".join(filenames)
    prefix = (
        "Это один коллаж, в котором уже есть ВСЕ нужные изображения. "
        "Каждая ячейка коллажа соответствует ОТДЕЛЬНОМУ исходному файлу. "
        "Разбирай КАЖДЫЙ файл по отдельности, в одном ответе, без просьб прислать ещё. "
        f"Порядок файлов слева направо, сверху вниз: {files_str}.\n\n"
    )
    return prefix + base_prompt


def build_client(cfg: Config) -> "OpenAI":
    if OpenAI is None:
        sys.exit("ERROR: openai package is not installed. Install via 'pip install openai'.")
    kwargs = {"api_key": cfg.api_key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return OpenAI(**kwargs)  # type: ignore[arg-type]


def call_model_with_image(
    cfg: Config,
    jpeg_bytes: bytes,
    system_prompt: str,
    user_text: Optional[str],
    quiet: bool,
) -> str:
    client = build_client(cfg)
    image_url = image_bytes_to_data_url(jpeg_bytes)

    content: List[dict] = []
    if user_text:
        content.append({"type": "text", "text": user_text})
    content.append({"type": "image_url", "image_url": {"url": image_url}})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    log(
        f"Запрос к модели (image): model={cfg.model}, "
        f"system_len={len(system_prompt)}, has_user_text={bool(user_text)}, "
        f"image_bytes={len(jpeg_bytes)} ({human_size(len(jpeg_bytes))})",
        quiet,
    )

    resp = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
    )
    text = resp.choices[0].message.content or ""
    log("Ответ модели (image) получен", quiet)
    return text


def call_model_with_text_only(
    cfg: Config,
    text: str,
    system_prompt: str,
    quiet: bool,
) -> str:
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

    resp = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
    )
    result = resp.choices[0].message.content or ""
    log("Ответ модели (text) получен", quiet)
    return result


def save_result_for_single_image(image_path: str, text: str) -> str:
    base, _ = os.path.splitext(image_path)
    out_path = f"{base}_analyse.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def save_result_for_group(images: Sequence[str], text: str) -> str:
    if not images:
        raise ValueError("No images to save group result")
    first_dir = os.path.dirname(images[0]) or "."
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"group_{ts}_analyse.txt"
    out_path = os.path.join(first_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai_image_analyzer",
        description="Analyze images or collages with an OpenAI-compatible vision model.",
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Image file paths (supports glob masks like '*.jpg'). "
             "If omitted, the request will be text-only.",
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
    # CLI overrides for .env vars
    parser.add_argument(
        "--OPENAI_API_KEY",
        "-k",
        dest="OPENAI_API_KEY",
        help="Override OPENAI_API_KEY from .env",
    )
    parser.add_argument(
        "--OPENAI_BASE_URL",
        "-u",
        dest="OPENAI_BASE_URL",
        help="Override OPENAI_BASE_URL from .env",
    )
    parser.add_argument(
        "--OPENAI_MODEL",
        "-m",
        dest="OPENAI_MODEL",
        help="Override OPENAI_MODEL from .env",
    )
    parser.add_argument(
        "--OPENAI_TIMEOUT",
        "-T",
        dest="OPENAI_TIMEOUT",
        type=int,
        help="Override OPENAI_TIMEOUT (seconds) from .env",
    )
    parser.add_argument(
        "--OPENAI_MAX_TOKENS",
        "-M",
        dest="OPENAI_MAX_TOKENS",
        type=int,
        help="Override OPENAI_MAX_TOKENS from .env",
    )
    parser.add_argument(
        "--IMAGE_MAX_SIZE",
        "-s",
        dest="IMAGE_MAX_SIZE",
        type=int,
        help="Override IMAGE_MAX_SIZE (px) from .env",
    )
    parser.add_argument(
        "--IMAGE_QUALITY",
        "-Q",
        dest="IMAGE_QUALITY",
        type=int,
        help="Override IMAGE_QUALITY (JPEG quality 1-100) from .env",
    )
    parser.add_argument(
        "--COLLAGE_MAX_SIZE",
        "-S",
        dest="COLLAGE_MAX_SIZE",
        type=int,
        help="Override COLLAGE_MAX_SIZE (px) from .env",
    )
    parser.add_argument(
        "--COLLAGE_QUALITY",
        "-G",
        dest="COLLAGE_QUALITY",
        type=int,
        help="Override COLLAGE_QUALITY (JPEG quality 1-100) from .env",
    )
    parser.add_argument(
        "--PROMPT_FILE",
        "-p",
        dest="PROMPT_FILE",
        help="Override PROMPT_FILE from .env",
    )
    parser.add_argument(
        "--per-image",
        action="store_true",
        help="Analyze each image in a separate request (no collage, no group file).",
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

    # CLI overrides -> env (чтобы load_config увидел финальные значения)
    override_env_names = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_TIMEOUT",
        "OPENAI_MAX_TOKENS",
        "IMAGE_MAX_SIZE",
        "IMAGE_QUALITY",
        "COLLAGE_MAX_SIZE",
        "COLLAGE_QUALITY",
        "PROMPT_FILE",
    ]
    for name in override_env_names:
        val = getattr(args, name, None)
        if val is not None:
            os.environ[name] = str(val)

    cfg = load_config()

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
        result = call_model_with_text_only(cfg, override_text, system_prompt="", quiet=args.quiet)
        print(result)
        return

    # IMAGE MODES
    if len(image_paths) > 10:
        log(f"Ограничиваем количество файлов до 10 (из {len(image_paths)})", args.quiet)
        image_paths = image_paths[:10]

    resized: List[Tuple[str, bytes]] = []
    for p in image_paths:
        try:
            data, before, after, ow, oh, nw, nh = load_and_resize_image(
                p,
                cfg.image_max_size,
                cfg.image_quality,
            )
            log(
                f"Resize: {p} {human_size(before)} → {human_size(after)} "
                f"({before} → {after} bytes); {ow}x{oh} → {nw}x{nh}",
                args.quiet,
            )
        except OSError as e:
            print(f"WARNING: failed to open image {p!r}: {e}", file=sys.stderr)
            continue
        resized.append((p, data))

    if not resized:
        print("ERROR: no valid images after loading/resizing.", file=sys.stderr)
        sys.exit(1)

    # PER-IMAGE MODE: каждый файл отдельным запросом
    if args.per_image:
        log("Режим: per-image — отдельный запрос для каждого файла", args.quiet)
        # Если есть override_text — system пустой, user_text = override_text
        # Если override_text нет — используем PROMPT_FILE как system, user_text=None
        system_prompt = "" if override_text else read_prompt_file(cfg.prompt_file)

        for img_path, jpeg_bytes in resized:
            name = os.path.basename(img_path)
            log(f"Отправляем отдельно: {name}", args.quiet)

            result = call_model_with_image(
                cfg,
                jpeg_bytes,
                system_prompt=system_prompt,
                user_text=override_text,
                quiet=args.quiet,
            )

            if override_text:
                # Режим с override_text: выводим в консоль, не сохраняем в файл
                print(f"\n===== {name} =====")
                print(result)
            else:
                out_path = save_result_for_single_image(img_path, result)
                print(f"Saved analysis to {out_path}")

        return

    # Обычный режим: PROMPT_FILE → system, без per-image
    # override_text: игнорируем PROMPT_FILE, system пустой, не сохраняем в файл
    if override_text:
        if len(resized) == 1:
            log("Режим: одна картинка + override текст", args.quiet)
            _, jpeg_bytes = resized[0]
        else:
            log("Режим: несколько картинок + override текст → создаём коллаж", args.quiet)
            collage_bytes, _ = make_collage(
                resized,
                cfg.collage_max_size,
                cfg.collage_quality,
                args.quiet,
            )
            jpeg_bytes = collage_bytes
        result = call_model_with_image(
            cfg,
            jpeg_bytes,
            system_prompt="",
            user_text=override_text,
            quiet=args.quiet,
        )
        print(result)
        return

    # Обычный режим: PROMPT_FILE → system, без override_text
    system_prompt = read_prompt_file(cfg.prompt_file)
    log(f"Системный промт загружен, длина={len(system_prompt)}", args.quiet)

    if len(resized) == 1:
        img_path, jpeg_bytes = resized[0]
        log("Режим: одна картинка, без override-текста", args.quiet)
        result = call_model_with_image(
            cfg,
            jpeg_bytes,
            system_prompt=system_prompt,
            user_text=None,
            quiet=args.quiet,
        )
        out_path = save_result_for_single_image(img_path, result)
        print(f"Saved analysis to {out_path}")
    else:
        log("Режим: несколько картинок → создаём коллаж", args.quiet)
        collage_bytes, filenames = make_collage(
            resized,
            cfg.collage_max_size,
            cfg.collage_quality,
            args.quiet,
        )
        collage_system_prompt = build_collage_system_prompt(system_prompt, filenames)
        log(
            f"Коллаж готов, файлов в описании: {len(filenames)}",
            args.quiet,
        )
        result = call_model_with_image(
            cfg,
            collage_bytes,
            system_prompt=collage_system_prompt,
            user_text=None,
            quiet=args.quiet,
        )
        original_paths = [p for p, _ in resized]
        out_path = save_result_for_group(original_paths, result)
        print(f"Saved group analysis to {out_path}")


if __name__ == "__main__":
    main()
