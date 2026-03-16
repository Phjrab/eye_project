보안 실행 예시 (실토큰/실비밀값 하드코딩 금지)
$env:HASH_PEPPER="your-secret-pepper"
$env:KAKAO_ACCESS_TOKEN="your-kakao-access-token"
python eye_server.py


보고서 생성 + 카카오 전송 테스트 (메인 서버 통합 API)
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/report/share" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "user_id":"010-1234-5678",
    "send_kakao":true
  }'
