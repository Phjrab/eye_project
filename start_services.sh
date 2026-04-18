#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -d "venv" ]]; then
  echo "[ERROR] venv not found at $PROJECT_DIR/venv"
  exit 1
fi

source venv/bin/activate

# Detect host IP address
HOST_IP=$(hostname -I | awk '{print $1}')
if [[ -z "$HOST_IP" ]]; then
  HOST_IP="127.0.0.1"
fi

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
  "server.py (port 5000)" \
  "python.*[ /]server.py" \
  "cd '$PROJECT_DIR' && source venv/bin/activate && python server.py" \
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

wait_for_http "server.py" "http://127.0.0.1:5000/health" || true
wait_for_http "kakao_login" "http://127.0.0.1:5001/kakao/login?phone=01012341234" || true

echo ""
echo "=========================================="
echo "  eye_server  →  http://${HOST_IP}:5000"
echo "  kakao app   →  http://${HOST_IP}:5001"
echo "=========================================="

# ── Epiphany 브라우저 전체화면 자동 실행 ──
export DISPLAY=:0
BROWSER_URL="http://127.0.0.1:5000/health"

# 이미 실행 중이면 건너뛰기
if pgrep -x epiphany-browse >/dev/null 2>&1 || pgrep -af "epiphany.*browser" >/dev/null 2>&1; then
  echo "[OK] Epiphany browser already running"
else
  echo "[INFO] Launching Epiphany browser (fullscreen)..."
  nohup epiphany-browser "$BROWSER_URL" > logs/browser.log 2>&1 &
  BROWSER_PID=$!

  # 브라우저 창이 뜰 때까지 대기 후 F11 전체화면
  sleep 4
  if command -v xdotool >/dev/null 2>&1; then
    # epiphany 창 찾아서 전체화면 전환
    WID=$(xdotool search --name "Web" 2>/dev/null | head -1 || true)
    if [[ -z "$WID" ]]; then
      WID=$(xdotool search --pid "$BROWSER_PID" 2>/dev/null | head -1 || true)
    fi
    if [[ -n "$WID" ]]; then
      xdotool windowactivate "$WID" 2>/dev/null || true
      sleep 0.5
      xdotool key F11 2>/dev/null || true
      echo "[OK] Browser fullscreen activated (window=$WID)"
    else
      echo "[WARN] Could not find browser window for fullscreen"
    fi
  fi
fi

echo "[DONE] Services startup routine finished."
