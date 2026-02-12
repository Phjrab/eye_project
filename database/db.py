# db.py
import sqlite3
import json
from typing import Optional, Dict, Any
from security_utils import phone_hash_id

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- 위 DDL 그대로 붙여넣으세요
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
