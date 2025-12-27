from typing import Tuple, Optional


def parse_tail_flags(t: str) -> Tuple[str, bool]:
    parts = [p for p in (t or "").split() if p]
    flags = set(p.lower() for p in parts if p.lower() == "group")
    cleaned = " ".join(p for p in parts if p.lower() not in flags)
    return cleaned, ("group" in flags)


def normalize_command(cmd: Optional[str]) -> Optional[str]:
    if not cmd:
        return None
    c = cmd
    no_underscore = c.replace("_", "").lower()
    if no_underscore == "statsall":
        return "stats_all"
    if no_underscore == "statsreset":
        return "stats_reset"
    if no_underscore == "userdel":
        return "user_del"
    return c
