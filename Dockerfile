FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code
COPY . .

# run as non-root
RUN useradd -m bot && chown -R bot /app
USER bot

CMD ["python", "bot.py"]
