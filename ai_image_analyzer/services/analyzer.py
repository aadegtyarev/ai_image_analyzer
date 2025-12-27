"""AnalyzerService: orchestrates calls to ModelProvider and normalizes responses."""
from typing import Any, Optional, Tuple

from ..providers.base import ModelProvider


class AnalyzerService:
    def __init__(self, provider: ModelProvider):
        self.provider = provider

    def analyze_text(self, cfg: Any, text: str, system_prompt: Optional[str] = None, quiet: bool = False) -> Tuple[str, Optional[dict]]:
        res = self.provider.call_text(cfg, text, system_prompt=system_prompt, quiet=quiet)
        if isinstance(res, tuple):
            return res
        return res, None

    def analyze_image(self, cfg: Any, jpeg_bytes: bytes, system_prompt: Optional[str] = None, user_text: Optional[str] = None, quiet: bool = False, image_meta: Optional[dict] = None) -> Tuple[str, Optional[dict]]:
        res = self.provider.call_image(cfg, jpeg_bytes, system_prompt=system_prompt, user_text=user_text, quiet=quiet, image_meta=image_meta)
        if isinstance(res, tuple):
            return res
        return res, None

    def check_balance(self, cfg: Any, quiet: bool = False) -> dict:
        return self.provider.check_balance(cfg, quiet=quiet)
