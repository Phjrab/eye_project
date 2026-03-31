#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

stop_by_pattern() {
  local name="$1"
  local pattern="$2"

  local pids
  pids=$(pgrep -f "$pattern" || true)

  if [[ -z "$pids" ]]; then
    echo "[OK] $name not running"
    return 0
  fi

  echo "[INFO] Stopping $name: $pids"
  kill $pids || true

  for _ in {1..10}; do
    sleep 1
    if ! pgrep -f "$pattern" >/dev/null; then
      echo "[OK] $name stopped"
      return 0
    fi
  done

  echo "[WARN] $name still running, forcing stop"
  pkill -9 -f "$pattern" || true

  if pgrep -f "$pattern" >/dev/null; then
    echo "[ERROR] Failed to stop $name"
    return 1
  fi

  echo "[OK] $name force-stopped"
}

stop_by_pattern "kakao app" "python.*database/app.py"
stop_by_pattern "eye_server" "python.*eye_server.py"
stop_by_pattern "epiphany browser" "epiphany.*browser"

echo "[DONE] Services stop routine finished."
