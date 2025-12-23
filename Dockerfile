FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps required by Pillow and other native packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Run as non-root
RUN useradd -m bot && chown -R bot /app
USER bot

CMD ["python", "bot.py"]
