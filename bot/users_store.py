import json
import os
from typing import Dict, Any

from .config import USERS_FILE


def _ensure_users_file_dir() -> None:
    d = os.path.dirname(USERS_FILE)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def load_users() -> Dict[str, Any]:
    _ensure_users_file_dir()
    if not os.path.exists(USERS_FILE):
        data = {"enabled": [], "stats": {}, "meta": {}}
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("enabled", [])
    data.setdefault("stats", {})
    data.setdefault("meta", {})
    return data


def save_users(data: Dict[str, Any]) -> None:
    _ensure_users_file_dir()
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_admin(user_id: int, admin_id: int) -> bool:
    return user_id == admin_id


def is_allowed(user_id: int, users_db: Dict[str, Any], admin_id: int) -> bool:
    if is_admin(user_id, admin_id):
        return True
    return user_id in users_db.get("enabled", [])


def ensure_stats(uid: int, users_db: Dict[str, Any]) -> None:
    stats = users_db.setdefault("stats", {})
    key = str(uid)
    if key not in stats:
        stats[key] = {
            "requests": 0,
            "images": 0,
            "megabytes": 0.0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }


def update_stats_after_call(users_db: Dict[str, Any], uid: int, images: int, bytes_sent: int, usage: dict) -> None:
    ensure_stats(uid, users_db)
    s = users_db["stats"][str(uid)]
    s["requests"] += 1
    s["images"] += images
    s["megabytes"] += bytes_sent / (1024.0 * 1024.0)
    if usage:
        s["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        try:
            s["total_cost"] += float(usage.get("total_cost", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
    save_users(users_db)


def set_user_meta(users_db: Dict[str, Any], uid: int, description: str, username: str = "", full_name: str = ""):
    meta = users_db.setdefault("meta", {})
    meta[str(uid)] = {
        "description": description,
        "username": username,
        "full_name": full_name,
    }
    save_users(users_db)
