#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR=""
if [[ -d "venv" ]]; then
  VENV_DIR="venv"
elif [[ -d ".venv" ]]; then
  VENV_DIR=".venv"
else
  echo "[ERROR] Python virtualenv not found at $PROJECT_DIR/venv or $PROJECT_DIR/.venv"
  exit 1
fi

source "$VENV_DIR/bin/activate"

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
  "eye_server.py (port 5000)" \
  "python.*eye_server.py" \
  "cd '$PROJECT_DIR' && source '$VENV_DIR/bin/activate' && python eye_server.py" \
  "logs/server.log"

start_if_not_running \
  "kakao app (port 5001)" \
  "python.*database/app.py" \
  "cd '$PROJECT_DIR' && source '$VENV_DIR/bin/activate' && KAKAO_APP_PORT=5001 python database/app.py" \
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

wait_for_http "eye_server.py" "http://127.0.0.1:5000/status" || true
wait_for_http "kakao_login" "http://127.0.0.1:5001/kakao/login?phone=01012341234" || true

echo ""
echo "=========================================="
echo "  eye_server  →  http://${HOST_IP}:5000"
echo "  kakao app   →  http://${HOST_IP}:5001"
echo "=========================================="

# ── Epiphany 브라우저 전체화면 자동 실행 ──
export DISPLAY="${DISPLAY:-:0}"

if [[ -z "${XAUTHORITY:-}" ]]; then
  XA_FROM_MUTTER="$(ls /run/user/"$(id -u)"/.mutter-Xwaylandauth.* 2>/dev/null | head -1 || true)"
  if [[ -n "$XA_FROM_MUTTER" ]]; then
    export XAUTHORITY="$XA_FROM_MUTTER"
  elif [[ -f "$HOME/.Xauthority" ]]; then
    export XAUTHORITY="$HOME/.Xauthority"
  fi
fi

BROWSER_URL="${BROWSER_URL:-http://127.0.0.1:5000/}"
RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
WAYLAND_SOCKET="${WAYLAND_DISPLAY:-wayland-0}"
WAYLAND_AVAILABLE=0
USE_WAYLAND_KIOSK="${USE_WAYLAND_KIOSK:-0}"

if [[ -S "$RUNTIME_DIR/$WAYLAND_SOCKET" ]]; then
  WAYLAND_AVAILABLE=1
fi

if [[ "$USE_WAYLAND_KIOSK" == "1" && "$WAYLAND_AVAILABLE" -eq 1 ]]; then
  export XDG_RUNTIME_DIR="$RUNTIME_DIR"
  export WAYLAND_DISPLAY="$WAYLAND_SOCKET"
  unset GDK_BACKEND

  echo "[INFO] Wayland session detected; launching native Epiphany for touch compatibility"

  # Remove stale X11 fallback window from previous runs.
  if pgrep -af "epiphany --private-instance --new-window" >/dev/null 2>&1; then
    pkill -f "epiphany --private-instance --new-window" || true
    sleep 1
  fi

  if pgrep -x epiphany >/dev/null 2>&1 || pgrep -x epiphany-browse >/dev/null 2>&1 || pgrep -af "epiphany.*browser" >/dev/null 2>&1; then
    echo "[OK] Epiphany browser already running (Wayland mode)"
  else
    echo "[INFO] Launching Epiphany browser (Wayland mode)..."
    nohup epiphany-browser --new-window "$BROWSER_URL" > logs/browser.log 2>&1 &
    sleep 2
    if pgrep -x epiphany >/dev/null 2>&1 || pgrep -x epiphany-browse >/dev/null 2>&1 || pgrep -af "epiphany.*browser" >/dev/null 2>&1; then
      echo "[OK] Epiphany browser started (Wayland mode)"
    else
      echo "[WARN] Epiphany browser did not start in Wayland mode; check logs/browser.log"
    fi
  fi

  echo "[DONE] Services startup routine finished."
  exit 0
fi

if [[ "$USE_WAYLAND_KIOSK" == "1" && "$WAYLAND_AVAILABLE" -eq 0 ]]; then
  echo "[WARN] USE_WAYLAND_KIOSK=1 but Wayland socket not found; falling back to X11 mode"
else
  echo "[INFO] Wayland kiosk mode disabled (USE_WAYLAND_KIOSK=$USE_WAYLAND_KIOSK); using X11 mode"
fi

activate_epiphany_fullscreen() {
  if ! command -v xdotool >/dev/null 2>&1; then
    echo "[WARN] xdotool is not installed; browser fullscreen step skipped"
    return 1
  fi

  local wid
  local epiphany_pid
  wid=$(xdotool search --onlyvisible --class "epiphany" 2>/dev/null | head -1 || true)
  if [[ -z "$wid" ]]; then
    wid=$(xdotool search --class "epiphany" 2>/dev/null | head -1 || true)
  fi
  if [[ -z "$wid" ]]; then
    epiphany_pid=$(pgrep -n epiphany || true)
    if [[ -n "$epiphany_pid" ]]; then
      wid=$(xdotool search --pid "$epiphany_pid" 2>/dev/null | head -1 || true)
    fi
  fi
  if [[ -n "$wid" ]]; then
    xdotool windowactivate "$wid" 2>/dev/null || true
    sleep 0.5
    xdotool key F11 2>/dev/null || true
    echo "[OK] Browser fullscreen activated (window=$wid)"
    return 0
  fi

  echo "[WARN] Could not find browser window for fullscreen (DISPLAY=$DISPLAY, XAUTHORITY=${XAUTHORITY:-unset})"
  return 1
}

# X11-only branch for environments without Wayland.
# 기존 프로세스가 있으면 우선 기존 창에 대해 전체화면 시도
fullscreen_done=0
if pgrep -x epiphany >/dev/null 2>&1 || pgrep -x epiphany-browse >/dev/null 2>&1 || pgrep -af "epiphany.*browser" >/dev/null 2>&1; then
  echo "[INFO] Epiphany browser already running; trying fullscreen on current window"
  if activate_epiphany_fullscreen; then
    fullscreen_done=1
  fi
else
  echo "[INFO] Launching Epiphany browser (fullscreen)..."
fi

# Wayland 세션에서 기존 인스턴스 제어가 안 되는 경우를 대비해 X11 private window로 fallback
if [[ "$fullscreen_done" -eq 0 ]]; then
  echo "[INFO] Launching dedicated kiosk window via X11 fallback"
  GDK_BACKEND=x11 nohup epiphany-browser --private-instance --new-window "$BROWSER_URL" > logs/browser.log 2>&1 &
  sleep 4
  if activate_epiphany_fullscreen; then
    fullscreen_done=1
  fi
fi

echo "[DONE] Services startup routine finished."
