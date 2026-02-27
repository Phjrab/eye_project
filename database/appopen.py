서버 켜기 (토큰 포함)
$env:HASH_PEPPER="my-secret-pepper"
$env:KAKAO_ACCESS_TOKEN="09wZP9_vQHSJC2ArwoGhdMyn6raB_Ls-AAAAAQoNIdkAAAGcnYQR-yrXsvB0zxAC" 
python app.py


post 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/diagnosis" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "phone":"010-1234-5678",
    "display_name":"테스트",
    "ai_reading":{"label":"normal","score":0.92},
    "pixel_metrics":{"redness_area":1234,"vessel_density":0.18},
    "survey":{"dry_eye":true,"screen_time_hours":7},
    "impression":"전반적으로 정상 범위입니다.",
    "make_pdf":true
  }'
