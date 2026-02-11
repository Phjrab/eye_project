PRAGMA foreign_keys = ON;

-- 사용자: 전화번호 원문 저장 X, phone_hash를 고유 ID로 사용
CREATE TABLE IF NOT EXISTS users (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  phone_hash      TEXT NOT NULL UNIQUE,      -- 사용자 고유 식별자 역할(전화번호 해시)
  display_name    TEXT,                       -- UI 표시용(선택)
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_users_phone_hash ON users(phone_hash);

-- 진단 세션(= 진단 1회 = 4종 데이터를 한 쌍으로 묶는 단위)
CREATE TABLE IF NOT EXISTS diagnosis_sessions (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id            INTEGER NOT NULL,

  diagnosed_at       TEXT NOT NULL DEFAULT (datetime('now')),

  -- (1) AI 판독값: 모델 결과(분류/스코어/라벨 등) - 구조 유연하게 JSON
  ai_reading_json    TEXT NOT NULL,

  -- (2) 픽셀 분석 수치: 하준 팀장님 지표들(예: vessel_density, redness_area 등)
  pixel_metrics_json TEXT NOT NULL,

  -- (3) 설문 데이터: 문진/자가 체크/증상/생활 습관
  survey_json        TEXT NOT NULL,

  -- (4) FHIR Bundle: 표준 교환용 (Observation/DiagnosticReport/Media 등 묶음)
  fhir_bundle_json   TEXT,                    -- 나중에 생성해 넣어도 됨

  impression         TEXT,                    -- 최종 소견(자연어)
  status             TEXT NOT NULL DEFAULT 'done',  -- done/pending/failed

  created_at         TEXT NOT NULL DEFAULT (datetime('now')),

  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_diag_user_time ON diagnosis_sessions(user_id, diagnosed_at);

-- 리포트/이미지 파일 자산 (PDF, 원본 이미지, 분석 오버레이 등)
CREATE TABLE IF NOT EXISTS session_assets (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  INTEGER NOT NULL,

  asset_type  TEXT NOT NULL,        -- 'pdf_report' | 'image_raw' | 'image_annot' | ...
  file_path   TEXT NOT NULL,        -- Jetson 로컬 경로
  mime_type   TEXT,
  sha256      TEXT,                 -- 무결성(선택)
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),

  FOREIGN KEY(session_id) REFERENCES diagnosis_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_assets_session_type ON session_assets(session_id, asset_type);

-- (선택) 이벤트 로그: 진단 완료 -> QR 생성/카톡 전송 등 이벤트 추적
CREATE TABLE IF NOT EXISTS event_logs (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  INTEGER,
  event_type  TEXT NOT NULL,        -- 'DIAG_DONE' | 'QR_CREATED' | 'KAKAO_SENT' | ...
  payload_json TEXT,                -- 이벤트 상세(JSON)
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(session_id) REFERENCES diagnosis_sessions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_event_time ON event_logs(created_at);
