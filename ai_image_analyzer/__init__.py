"""ai_image_analyzer package facade (core functions).

Exports small set of helper functions used by the bot and tests.
"""
from importlib.metadata import version

try:
    __version__ = version("ai_image_analyzer")
except Exception:
    __version__ = "0.0.0"

from .config import (
    Settings,
    load_config,
    load_config_from_env,
    read_prompt_file,
)
from .image_processing import (
    resize_image_bytes,
    make_collage,
    build_collage_system_prompt,
)
from .image_io import load_and_resize_image, resize_image_from_bytes
from .model_client import (
    call_model_with_image,
    call_model_with_text_only,
    check_balance,
)
from .json_api import handle_json_request

__all__ = [
    "Settings",
    "load_config",
    "load_config_from_env",
    "read_prompt_file",
    "resize_image_bytes",
	"load_and_resize_image",
	"resize_image_from_bytes",
    "make_collage",
    "build_collage_system_prompt",
    "call_model_with_image",
    "call_model_with_text_only",
    "check_balance",
    "handle_json_request",
    "__version__",
]

"""Package public API (lazy imports to avoid circular imports during package
initialization).
"""

__all__ = [
	"load_config",
	"check_balance",
	"load_config_from_env",
	"read_prompt_file",
	"load_and_resize_image",
	"handle_json_request",
	"call_model_with_image",
	"call_model_with_text_only",
]


def __getattr__(name: str):
	if name in ("load_config", "load_config_from_env", "check_balance"):
		from .config import (
			load_config,
			load_config_from_env,
			check_balance,
		)
		return locals()[name]
	if name == "handle_json_request":
		from .api import handle_json_request
		return handle_json_request
	if name == "read_prompt_file":
		from .prompts import read_prompt_file
		return read_prompt_file
	if name == "load_and_resize_image":
		from .image_io import load_and_resize_image
		return load_and_resize_image
	if name in ("call_model_with_image", "call_model_with_text_only", "make_collage", "build_collage_system_prompt"):
		from .core import call_model_with_image, call_model_with_text_only, make_collage, build_collage_system_prompt
		return locals()[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

