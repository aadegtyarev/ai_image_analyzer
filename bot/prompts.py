import os
from typing import Dict
from dataclasses import dataclass


@dataclass
class PromptInfo:
    command: str
    filename: str
    path: str
    description: str


def load_prompts(prompts_dir: str | None = None) -> Dict[str, dict]:
    dir_to_use = prompts_dir or os.getenv("PROMPTS_DIR", "prompts")
    prompts = {}
    if not os.path.isdir(dir_to_use):
        return prompts
    files = [f for f in sorted(os.listdir(dir_to_use)) if f.lower().endswith(".txt")]
    idx = 1
    used = set()
    for fname in files:
        base = os.path.splitext(fname)[0]
        cmd = base.lower()
        if cmd in used:
            cmd = f"{cmd}_{idx}"
            idx += 1
        used.add(cmd)
        path = os.path.join(dir_to_use, fname)
        desc = ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                first = f.readline().strip()
                desc = first[:80] if first else f"Prompt from {fname}"
        except Exception:
            desc = f"Prompt from {fname}"
        prompts[cmd] = PromptInfo(command=cmd, filename=fname, path=path, description=desc)
    return prompts
