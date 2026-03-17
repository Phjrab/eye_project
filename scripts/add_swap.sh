#!/bin/bash
# Jetson Orin Nano - 8GB 추가 스왑 파일 생성 스크립트
# 실행: sudo bash scripts/add_swap.sh

set -e

SWAP_FILE="/swapfile2"
SWAP_SIZE_GB=8

echo "[1/5] 기존 스왑 상태 확인..."
free -h
swapon --show

# 이미 존재하면 스킵
if swapon --show | grep -q "$SWAP_FILE"; then
    echo "✅ $SWAP_FILE 이미 활성화되어 있습니다."
    exit 0
fi

echo "[2/5] ${SWAP_SIZE_GB}GB 스왑 파일 생성 중... (시간이 걸릴 수 있습니다)"
fallocate -l ${SWAP_SIZE_GB}G $SWAP_FILE || dd if=/dev/zero of=$SWAP_FILE bs=1G count=$SWAP_SIZE_GB status=progress

echo "[3/5] 권한 설정..."
chmod 600 $SWAP_FILE

echo "[4/5] 스왑 포맷 및 활성화..."
mkswap $SWAP_FILE
swapon $SWAP_FILE

echo "[5/5] 부팅 시 자동 마운트 등록 (/etc/fstab)..."
if ! grep -q "$SWAP_FILE" /etc/fstab; then
    echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
    echo "fstab 등록 완료"
else
    echo "fstab 이미 등록됨"
fi

echo ""
echo "✅ 완료! 최종 스왑 상태:"
free -h
swapon --show
