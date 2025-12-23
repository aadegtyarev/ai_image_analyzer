#!/bin/sh
set -eu

# Simple entrypoint for the container.
# - validate required envs
# - exec the given CMD so signals are forwarded correctly

if [ -z "${BOT_TOKEN:-}" ]; then
  echo "Error: required environment variable BOT_TOKEN is not set. Exiting." >&2
  exit 1
fi

echo "Starting ai_image_analyzer..."
exec "$@"
