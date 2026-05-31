#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
fi

docker compose up -d

python - <<'PY'
import time

import psycopg

from rag.config import settings

last_error = None
for _ in range(30):
    try:
        conn = psycopg.connect(settings.database_url, connect_timeout=2)
        conn.close()
        print("Postgres is ready.")
        break
    except Exception as exc:
        last_error = exc
        time.sleep(1)
else:
    raise SystemExit(f"Postgres did not become ready: {last_error}")
PY

if [ -z "${PORT:-}" ]; then
  PORT="$(python - <<'PY'
import socket

for port in range(7860, 7900):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", port)) != 0:
            print(port)
            break
PY
)"
fi

echo "Starting Gradio on http://127.0.0.1:${PORT}"
python -c "import app; app.demo.launch(server_port=${PORT})"
