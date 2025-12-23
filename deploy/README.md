# Deploy (Docker) — ai_image_analyzer

This folder contains instructions and examples to run the bot using Docker on an Ubuntu server.

## Quickstart (local)

1. Build the image:

   docker compose build --no-cache

2. Start the service:

   docker compose up -d

3. View logs:

   docker compose logs -f

4. Stop and remove containers:

   docker compose down

Notes:
- Provide a `.env` file at the repo root with at least `BOT_TOKEN` and `BOT_ADMIN_ID`.
- We mount `./prompts` and `./howto` read-only so you can edit prompts on the host and reload with `/reload_prompts`.

## Production (Ubuntu)

1. Install Docker on the server (Ubuntu 22.04+):

   sudo apt update && sudo apt install -y docker.io

   # make sure `docker compose` is available on the host (docker plugin / newer docker packages)

2. Copy repository to server (or `git clone`), place a secure `.env` with secrets.

3. Build and start:

   docker compose build
   docker compose up -d

4. To update: pull new code, rebuild and restart:

   git pull
   docker compose build
   docker compose up -d

## systemd Unit (optional)

You can create a systemd service to manage running Compose at boot:

Create `/etc/systemd/system/ai_image_analyzer.service` with:

```ini
[Unit]
Description=AI Image Analyzer (docker compose)
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/path/to/ai_image_analyzer
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down

[Install]
WantedBy=multi-user.target
```

Enable and start:

  sudo systemctl enable --now ai_image_analyzer.service

Note: the container image includes a small entrypoint that validates `BOT_TOKEN` and will exit with a non-zero code if it's not set. This helps surface missing secrets quickly when the service runs on boot.

## Security notes
- Do not commit `.env` to git (contains BOT_TOKEN).
- Consider using Docker secrets for production if running on Docker Swarm.
