"""Public API for ai_image_analyzer.

Expose a small, explicit set of helpers used by the bot and tests.
Keep this module simple to avoid surprising lazy-imports.
"""
from importlib.metadata import version

try:
	__version__ = version("ai_image_analyzer")
except Exception:
	__version__ = "0.0.0"

from .config import Settings, load_config, load_config_from_env, read_prompt_file
from .image_io import load_and_resize_image, resize_image_from_bytes
from .image_processing import resize_image_bytes, make_collage, build_collage_system_prompt
from .model_client import call_model_with_image, call_model_with_text_only, check_balance
from .json_api import handle_json_request

__all__ = [
	"Settings",
	"load_config",
	"load_config_from_env",
	"read_prompt_file",
	"load_and_resize_image",
	"resize_image_from_bytes",
	"resize_image_bytes",
	"make_collage",
	"build_collage_system_prompt",
	"call_model_with_image",
	"call_model_with_text_only",
	"check_balance",
	"handle_json_request",
	"__version__",
]

