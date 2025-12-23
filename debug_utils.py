"""
Debug dump utility has been removed. See `bot.py` for simplified debug logging
which uses `DEBUG` / `IMAGE_DEBUG` and prints details to stderr.

This stub remains for backward compatibility in case external imports still
reference `dump_payload_to_file`. It intentionally does nothing.
"""

def dump_payload_to_file(*args, **kwargs):
    # no-op: dumping to disk removed per project decision
    return None
