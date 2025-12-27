from typing import Protocol, Any, Optional


class ModelProvider(Protocol):
    """Protocol describing a model provider implementation."""

    def call_text(self, cfg: Any, text: str, system_prompt: Optional[str] = None, quiet: bool = False) -> Any:
        ...

    def call_image(self, cfg: Any, jpeg_bytes: bytes, system_prompt: Optional[str] = None, user_text: Optional[str] = None, quiet: bool = False, image_meta: Optional[dict] = None) -> Any:
        ...

    def check_balance(self, cfg: Any, quiet: bool = False) -> dict:
        ...
