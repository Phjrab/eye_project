#!/bin/bash
# Jetson Orin Nano 성능 최적화 및 메모리 정리 스크립트
# 사용법: sudo bash scripts/perf_setup.sh

set -e

echo "[1/3] nvpmodel → MAXN 최대 성능 모드 설정..."
nvpmodel -m 0

echo "[2/3] jetson_clocks → 클럭 최대 고정..."
jetson_clocks

echo "[3/3] 메모리 캐시 정리..."
sync
echo 3 > /proc/sys/vm/drop_caches

echo ""
echo "✅ 완료! 현재 상태:"
nvpmodel -q | head -5
free -h | grep Mem
cat /sys/devices/virtual/thermal/thermal_zone0/temp | awk '{printf "CPU 온도: %.1f°C\n", $1/1000}'
