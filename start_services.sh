#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -d "venv" ]]; then
  echo "[ERROR] venv not found at $PROJECT_DIR/venv"
  exit 1
fi

source venv/bin/activate

mkdir -p logs database

if [[ ! -f "database/database.db" ]]; then
  echo "[INFO] Initializing database/database.db"
  python - <<'PY'
from database.db import init_db
init_db('database/database.db')
print('database/database.db initialized')
PY
fi

start_if_not_running() {
  local name="$1"
  local pattern="$2"
  local cmd="$3"
  local logfile="$4"

  if pgrep -af "$pattern" >/dev/null; then
    echo "[OK] $name already running"
    pgrep -af "$pattern" | sed 's/^/[PID] /'
    return 0
  fi

  echo "[INFO] Starting $name"
  nohup bash -lc "$cmd" > "$logfile" 2>&1 &
  local pid=$!
  sleep 2

  if pgrep -af "$pattern" >/dev/null; then
    echo "[OK] $name started (pid=$pid)"
  else
    echo "[ERROR] Failed to start $name. Check $logfile"
    tail -n 80 "$logfile" || true
    exit 1
  fi
}

start_if_not_running \
  "eye_server (port 5000)" \
  "python.*eye_server.py" \
  "cd '$PROJECT_DIR' && source venv/bin/activate && python eye_server.py" \
  "logs/server.log"

start_if_not_running \
  "kakao app (port 5001)" \
  "python.*database/app.py" \
  "cd '$PROJECT_DIR' && source venv/bin/activate && KAKAO_APP_PORT=5001 python database/app.py" \
  "logs/kakao_app.log"

wait_for_http() {
  local label="$1"
  local url="$2"

  for _ in {1..20}; do
    code=$(curl -s -o /dev/null -w "%{http_code}" "$url" || true)
    if [[ "$code" != "000" && "$code" != "404" ]]; then
      echo "[OK] $label reachable ($code)"
      return 0
    fi
    sleep 1
  done

  echo "[WARN] $label not confirmed yet: $url"
  return 1
}

wait_for_http "eye_server" "http://127.0.0.1:5000/capture" || true
wait_for_http "kakao_login" "http://127.0.0.1:5001/kakao/login?phone=01012341234" || true

echo "[DONE] Services startup routine finished."
