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

3. Build and start (or use the install script below):

   docker compose build
   docker compose up -d

### Install as a service (automated)

If you'd like a systemd-managed service (recommended on single-server deployments), use the provided installer script which sets up a venv, installs dependencies, creates `.env` (if missing), and installs a systemd unit:

```bash
sudo ./deploy/install-as-service.sh --install-dir /home/$USER/applications/ai_image_analyzer --user $USER
```

After that, check status and logs:

```bash
sudo systemctl status ai_image_analyzer
sudo journalctl -u ai_image_analyzer -f
```

### Uninstall Docker (if you prefer to run the app directly)

If you decide to stop using Docker on the host and run the bot as a plain systemd service, remove Docker packages and clean up data:

```bash
sudo systemctl stop docker
sudo apt remove --purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
sudo apt autoremove -y
# Optional: remove leftover data (volumes/images) - be careful: this deletes images/containers/volumes
sudo rm -rf /var/lib/docker /etc/docker /var/run/docker.sock
```

Make sure you've backed up any volumes/host data you need before deleting.

### Run the bot directly (systemd) — recommended when not using Docker

1. Create a system user and group for running the bot (example):

```bash
sudo useradd -m --system aiimage
```

2. Clone repository and set up Python environment (recommended in the user's home):

```bash
sudo -u aiimage -H bash -c '
  cd ~
  git clone https://github.com/<your>/ai_image_analyzer.git
  cd ai_image_analyzer
  python3 -m venv venv
  ./venv/bin/pip install -r requirements.txt
'
# Create and fill .env with BOT_TOKEN and other variables in repo root
```

3. Copy and enable the example systemd unit file (edit paths first):

```bash
sudo cp deploy/ai_image_analyzer.service.example /etc/systemd/system/ai_image_analyzer.service
sudo systemctl daemon-reload
sudo systemctl enable --now ai_image_analyzer.service
sudo systemctl status ai_image_analyzer.service
```

4. Logs are available via journalctl:

```bash
sudo journalctl -u ai_image_analyzer -f
```

This approach starts `bot.py` directly under a system user, uses a Python virtualenv, and relies on systemd to restart the process on failure.

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
