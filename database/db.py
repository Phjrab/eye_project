# db.py
import sqlite3
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# security_utils import (utils 폴더에 있음)
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from security_utils import phone_hash_id

SCHEMA_SQL = """
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
"""

def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db(db_path: str) -> None:
    conn = get_conn(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()

def upsert_user_by_phone(db_path: str, phone: str, display_name: Optional[str] = None) -> int:
    """
    phone_hash를 고유 ID로 사용하므로:
    - 이미 존재하면 id 반환
    - 없으면 생성 후 id 반환
    """
    ph = phone_hash_id(phone)

    conn = get_conn(db_path)
    try:
        row = conn.execute("SELECT id FROM users WHERE phone_hash = ?", (ph,)).fetchone()
        if row:
            # display_name 업데이트(선택)
            if display_name:
                conn.execute("UPDATE users SET display_name=? WHERE id=?", (display_name, row["id"]))
                conn.commit()
            return int(row["id"])

        cur = conn.execute(
            "INSERT INTO users (phone_hash, display_name) VALUES (?, ?)",
            (ph, display_name)
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()

def create_diagnosis_session(
    db_path: str,
    user_id: int,
    ai_reading: Dict[str, Any],
    pixel_metrics: Dict[str, Any],
    survey: Dict[str, Any],
    impression: Optional[str] = None,
    fhir_bundle: Optional[Dict[str, Any]] = None,
) -> int:
    conn = get_conn(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO diagnosis_sessions (
              user_id, ai_reading_json, pixel_metrics_json, survey_json,
              fhir_bundle_json, impression, status
            )
            VALUES (?, ?, ?, ?, ?, ?, 'done')
            """,
            (
                user_id,
                json.dumps(ai_reading, ensure_ascii=False),
                json.dumps(pixel_metrics, ensure_ascii=False),
                json.dumps(survey, ensure_ascii=False),
                json.dumps(fhir_bundle, ensure_ascii=False) if fhir_bundle else None,
                impression
            )
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()
