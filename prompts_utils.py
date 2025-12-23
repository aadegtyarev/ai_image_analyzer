import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class PromptInfo:
    command: str
    filename: str
    path: str
    description: str


def sanitize_command_name(base: str, used: set, idx: int) -> str:
    name = base.lower()
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"
    name = "".join(c if c in allowed else "_" for c in name)
    name = name.strip("_")

    if not name:
        name = f"p_{idx}"

    if not name[0].isalpha():
        name = f"p_{name}"

    if len(name) > 32:
        name = name[:32]

    orig = name
    suffix = 1
    while name in used:
        candidate = f"{orig}_{suffix}"
        if len(candidate) > 32:
            candidate = candidate[:32]
        name = candidate
        suffix += 1

    used.add(name)
    return name


def load_prompts(prompts_dir: str = None) -> Dict[str, PromptInfo]:
    raise RuntimeError("prompts_utils.load_prompts has been removed; use bot.load_prompts instead")
