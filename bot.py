#!/usr/bin/env python3
"""Thin runner for the `bot` package — delegates to `bot.main.start()`.

This file is intentionally minimal to provide a convenient entrypoint
`python bot.py` for local development and compatibility with existing docs.
"""
import sys
from bot.main import start


def main() -> None:
    try:
        start()
    except Exception as e:
        print(f"Failed to start bot: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
