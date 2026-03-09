import os
import io
import json
import re
import hashlib
import socket
import sqlite3
import requests
import qrcode
from flask import Flask, request, jsonify, send_file
_PHONE_RE = re.compile(r"\D+")

def get_lan_ip() -> str:
    """
    폰/다른 기기에서 접근 가능한 '내 PC의 로컬(LAN) IP'를 잡는다.
    (172.x 같은 가상/WSL IP가 아니라 보통 192.168.x.x가 나옴)
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # 실제 전송 안 하고 로컬 IP만 얻음
        return s.getsockname()[0]
    finally:
        s.close()

def normalize_phone(phone: str) -> str:
    digits = _PHONE_RE.sub("", str(phone))
    if len(digits) < 9:
        raise ValueError(f"phone looks too short: {digits}")
    return digits

def phone_hash_id(phone: str) -> str:
    pepper = os.environ.get("HASH_PEPPER")
    if not pepper:
        raise RuntimeError("HASH_PEPPER 환경변수 필요! PowerShell: $env:HASH_PEPPER='secret'")
    norm = normalize_phone(phone)
    return hashlib.sha256(f"{norm}|{pepper}".encode("utf-8")).hexdigest()

def kakao_send_me(text: str, link_url: str):
    token = os.environ.get("KAKAO_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("KAKAO_ACCESS_TOKEN 환경변수가 필요합니다.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    template_object = {
        "object_type": "text",
        "text": text,
        "link": {
            "web_url": link_url,
            "mobile_web_url": link_url
        },
        "button_title": "리포트 열기",
    }

    data = {
        "template_object": json.dumps(template_object, ensure_ascii=False)
    }

    r = requests.post(
        "https://kapi.kakao.com/v2/api/talk/memo/default/send",
        headers=headers,
        data=data,
        timeout=10
    )

    if r.status_code != 200:
        raise RuntimeError(f"Kakao send failed: {r.status_code} {r.text}")

    return r.json()

DB_PATH = "db/database.db"
REPORT_DIR = "reports"

app = Flask(__name__)


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def upsert_user_by_phone(conn, phone: str, display_name: str | None = None) -> int:
    ph = phone_hash_id(phone)
    cur = conn.cursor()
    cur.execute("SELECT id, display_name FROM users WHERE phone_hash=?", (ph,))
    row = cur.fetchone()
    if row:
        if display_name and display_name != row["display_name"]:
            cur.execute("UPDATE users SET display_name=? WHERE id=?", (display_name, row["id"]))
        return int(row["id"])

    cur.execute("INSERT INTO users (phone_hash, display_name) VALUES (?, ?)", (ph, display_name))
    return int(cur.lastrowid)


def create_session(conn, user_id: int, ai_reading: dict, pixel_metrics: dict, survey: dict,
                   impression: str | None = None, fhir_bundle: dict | None = None,
                   status: str = "done") -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO diagnosis_sessions (
          user_id, ai_reading_json, pixel_metrics_json, survey_json,
          fhir_bundle_json, impression, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            json.dumps(ai_reading, ensure_ascii=False),
            json.dumps(pixel_metrics, ensure_ascii=False),
            json.dumps(survey, ensure_ascii=False),
            json.dumps(fhir_bundle, ensure_ascii=False) if fhir_bundle else None,
            impression,
            status,
        )
    )
    return int(cur.lastrowid)



def add_asset(conn, session_id: int, asset_type: str, file_path: str, mime_type: str | None = None):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO session_assets (session_id, asset_type, file_path, mime_type)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, asset_type, file_path, mime_type)
    )
    return int(cur.lastrowid)


def log_event(conn, session_id: int | None, event_type: str, payload: dict | None = None):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO event_logs (session_id, event_type, payload_json) VALUES (?, ?, ?)",
        (session_id, event_type, json.dumps(payload, ensure_ascii=False) if payload else None)
    )
    return int(cur.lastrowid)


def generate_pdf_report(session_id: int, ai_reading: dict, pixel_metrics: dict, survey: dict, impression: str | None):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    os.makedirs(REPORT_DIR, exist_ok=True)
    pdf_path = os.path.join(REPORT_DIR, f"report_{session_id}.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, f"Diagnosis Report (session_id={session_id})")
    y -= 40

    c.setFont("Helvetica", 11)
    c.drawString(50, y, "AI Reading:")
    y -= 18
    c.drawString(70, y, json.dumps(ai_reading, ensure_ascii=False))
    y -= 28

    c.drawString(50, y, "Pixel Metrics:")
    y -= 18
    c.drawString(70, y, json.dumps(pixel_metrics, ensure_ascii=False))
    y -= 28

    c.drawString(50, y, "Survey:")
    y -= 18
    c.drawString(70, y, json.dumps(survey, ensure_ascii=False))
    y -= 28

    if impression:
        c.drawString(50, y, "Impression:")
        y -= 18
        c.drawString(70, y, impression)
        y -= 28

    c.showPage()
    c.save()
    return pdf_path


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/diagnosis")
def post_diagnosis():
    data = request.get_json(force=True)

    phone = data.get("phone", "")
    display_name = data.get("display_name")

    ai_reading = data.get("ai_reading", {})
    pixel_metrics = data.get("pixel_metrics", {})
    survey = data.get("survey", {})
    impression = data.get("impression")
    make_pdf = bool(data.get("make_pdf", False))
    print("DEBUG make_pdf =", make_pdf)
    conn = get_conn()
    try:
        user_id = upsert_user_by_phone(conn, phone, display_name)
        session_id = create_session(conn, user_id, ai_reading, pixel_metrics, survey, impression=impression)
        log_event(conn, session_id, "DIAG_DONE", {"user_id": user_id})

        pdf_path = None
        if make_pdf:
            pdf_path = generate_pdf_report(session_id, ai_reading, pixel_metrics, survey, impression)
            add_asset(conn, session_id, "pdf_report", pdf_path, "application/pdf")
            log_event(conn, session_id, "PDF_CREATED", {"path": pdf_path})

            # (카톡) 일단 꺼두는 게 안전 -> 아래 3줄은 주석 처리해도 됨
            #server_ip = socket.gethostbyname(socket.gethostname())
            server_ip = get_lan_ip()   # <-- 이걸로 바꿔야 172.x 같은 거 안 잡힘
            open_url = f"http://{server_ip}:5000/open/{session_id}"

        
            try:    
                kakao_send_me(
                    text=f"진단 리포트가 생성되었습니다. (session_id={session_id})\n\n리포트 열기:\n{open_url}",
                    link_url=open_url
                )
                log_event(conn, session_id, "KAKAO_SENT", {"url": open_url})
            except Exception as e:
                print("DEBUG kakao error:", e)
                log_event(conn, session_id, "KAKAO_FAILED", {"error": str(e)})

        conn.commit()
        return jsonify({
            "session_id": session_id,
            "user_id": user_id,
            "pdf_path": pdf_path
        })

    finally:
        conn.close()


@app.get("/report/<int:session_id>")
def get_report(session_id: int):
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT file_path FROM session_assets WHERE session_id=? AND asset_type='pdf_report' ORDER BY created_at DESC LIMIT 1",
            (session_id,)
        ).fetchone()
        if not row:
            return jsonify({"error": "pdf not found"}), 404

        path = row["file_path"]
        if not os.path.exists(path):
            return jsonify({"error": "file missing on disk", "path": path}), 500
        # download=1 이면 첨부(다운로드), 아니면 inline(브라우저에서 열기)
        download = request.args.get("download") == "1"
        return send_file(path, mimetype="application/pdf", as_attachment=download, download_name=os.path.basename(path))
    finally:
        conn.close()

from flask import Response  # 상단 import에 추가

@app.get("/open/<int:session_id>")
def open_report(session_id: int):
    # 카톡 인앱브라우저에서 PDF 바로 렌더링이 불안정해서,
    # HTML 페이지에서 iframe으로 보여주는 방식
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Report {session_id}</title>
      <style>
        body, html {{ margin:0; padding:0; height:100%; }}
        .bar {{ padding:12px; font-family: sans-serif; }}
        iframe {{ width:100%; height: calc(100% - 52px); border:0; }}
        a {{ display:inline-block; padding:8px 12px; border:1px solid #ccc; border-radius:10px; text-decoration:none; }}
      </style>
    </head>
    <body>
      <div class="bar">
        <a href="/report/{session_id}" target="_blank">PDF 새탭으로 열기</a>
        &nbsp;
        <a href="/report/{session_id}?download=1">다운로드</a>
      </div>
      <iframe src="/report/{session_id}"></iframe>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")

@app.get("/qr/<int:session_id>")
def get_qr(session_id: int):
    # 서버 IP 구하기
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    # PDF 다운로드 링크
    url = f"http://{local_ip}:5000/report/{session_id}"

    # QR 코드 생성
    img = qrcode.make(url)

    # 메모리 버퍼에 이미지 저장
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
