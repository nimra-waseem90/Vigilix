"""
Vigilix backend.

Serves the frontend (frontend/index.html), authenticates the operator,
lets you register an RTSP camera and immediately starts fire + weapon
detection on it, streams the annotated feed back as MJPEG, and exposes
recent fused alerts as JSON for the dashboard to poll.

Run from the project root with:  python server/app.py
(see README.md for full setup)
"""

import os
import sys
import time
import uuid
import threading
from datetime import datetime

# Make project-root-relative imports/paths (camera/, model/, alert/, fusion/)
# work no matter where this file is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from flask import Flask, request, jsonify, session, send_from_directory, Response
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

from server.stream_worker import (  # noqa: E402
    StreamWorker, FIRE_CONF_THRESHOLD, WEAPON_CONF_THRESHOLD, REQUIRED_FRAMES,
)
from server import store  # noqa: E402

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
SCREENSHOT_DIR = os.path.join(PROJECT_ROOT, "alert", "screenshots")

app = Flask(__name__, static_folder=None)
app.secret_key = os.environ.get("VIGILIX_SECRET_KEY", "dev-secret-change-me")

ADMIN_USER = os.environ.get("VIGILIX_ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("VIGILIX_ADMIN_PASS", "vigilix123")

streams = {}            # camera_id -> StreamWorker
alerts = []              # newest-first in-memory alert feed (mirrors the DB)
alerts_lock = threading.Lock()
MAX_ALERTS = 200


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return jsonify({"error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


def record_alert(alert):
    alert["id"] = uuid.uuid4().hex[:8]
    alert["created_at"] = datetime.now().isoformat()
    with alerts_lock:
        alerts.insert(0, alert)
        del alerts[MAX_ALERTS:]
    store.save_alert(alert)


def start_camera(camera_id, name, source, persist=True):
    worker = StreamWorker(camera_id, name, source, on_alert=record_alert)
    streams[camera_id] = worker
    worker.start()
    if persist:
        store.save_camera(camera_id, name, source, datetime.now().isoformat())
    return worker


# Restore state saved on a previous run - the app starts with real history,
# not dummy data, but doesn't lose everything on every restart either.
with alerts_lock:
    alerts.extend(store.load_alerts(MAX_ALERTS))
for cam in store.load_cameras():
    start_camera(cam["id"], cam["name"], cam["source"], persist=False)


# ---------------------------------------------------------------- frontend
@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/alert/screenshots/<path:name>")
@login_required
def screenshots(name):
    return send_from_directory(SCREENSHOT_DIR, name)


@app.get("/alert/alarm.wav")
def alarm_sound():
    return send_from_directory(os.path.join(PROJECT_ROOT, "alert"), "alarm.wav")


# ---------------------------------------------------------------- config
@app.get("/api/config")
@login_required
def get_config():
    return jsonify({
        "fire_confidence": FIRE_CONF_THRESHOLD,
        "weapon_confidence": WEAPON_CONF_THRESHOLD,
        "required_frames": REQUIRED_FRAMES,
    })


# ---------------------------------------------------------------- auth
@app.post("/api/login")
def login():
    data = request.get_json(force=True, silent=True) or {}
    if data.get("username") == ADMIN_USER and data.get("password") == ADMIN_PASS:
        session["user"] = ADMIN_USER
        return jsonify({"ok": True, "user": ADMIN_USER})
    return jsonify({"ok": False, "error": "Invalid email or password"}), 401


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/session")
def get_session():
    user = session.get("user")
    return jsonify({"authenticated": bool(user), "user": user})


# ---------------------------------------------------------------- streams
@app.get("/api/streams")
@login_required
def list_streams():
    return jsonify([w.to_dict() for w in streams.values()])


@app.post("/api/streams")
@login_required
def add_stream():
    data = request.get_json(force=True, silent=True) or {}
    source = (data.get("rtsp_url") or "").strip()
    name = (data.get("name") or "Camera").strip()

    if not source:
        return jsonify({"error": "rtsp_url is required"}), 400

    camera_id = uuid.uuid4().hex[:8]
    worker = start_camera(camera_id, name, source)

    return jsonify(worker.to_dict()), 201


@app.delete("/api/streams/<camera_id>")
@login_required
def remove_stream(camera_id):
    worker = streams.pop(camera_id, None)
    if worker:
        worker.stop()
    store.delete_camera(camera_id)
    return jsonify({"ok": True})


@app.get("/api/streams/<camera_id>/mjpeg")
@login_required
def mjpeg(camera_id):
    worker = streams.get(camera_id)
    if worker is None:
        return jsonify({"error": "not found"}), 404

    def generate():
        boundary = b"--frame"
        worker.add_viewer()
        last_version = -1
        try:
            while camera_id in streams and not worker.is_stopped():
                frame, version = worker.wait_for_next_frame(last_version, timeout=0.5)
                if frame and version != last_version:
                    last_version = version
                    yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                           + frame + b"\r\n")
                elif frame is None and worker.status in ("error", "stopped"):
                    break
        except (GeneratorExit, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            worker.remove_viewer()

    return Response(generate(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------- alerts
@app.get("/api/alerts")
@login_required
def get_alerts():
    with alerts_lock:
        return jsonify(alerts[:100])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
