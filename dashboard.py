"""
YOLO-3D Industrial Dashboard - Flask web server for monitoring & control.

HTTP endpoints:
  GET  /                          -> Dashboard HTML (industrial UI)
  GET  /video/<vis|depth|isaac>   -> MJPEG stream (UI shows vis + isaac; depth optional)
  POST /api/push_isaac_frame      -> Isaac Sim pushes JPEG frames here
  POST /api/push_log              -> External processes push log lines here
  POST /api/control/<action>      -> Button click handler (pause/resume/...)
  GET  /api/status                -> SSE stream: telemetry + status + logs

Public Python API (backward compatible):
  start(host, port)
  push_frame(key, frame_bgr)
  push_detections(rows, fps, frame_id)
  push_udp_log(entry)
  set_isaac_connected(ok)

New Python API:
  push_log(env, msg, level='info')        - env: pipeline|camera|detector|udp|isaac|system
  set_status(env, ok, detail='')          - update environment status indicator
  register_control(action, callback)      - bind a UI button to a Python callback
  is_paused() -> bool                     - read pause flag (run loop can honor it)
  configure_isaac_commands(host, port)    - UDP destination for Isaac Sim control JSON (default :6001)
"""

import json
import socket
import threading
import time
from collections import deque
from typing import Callable, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

# ── App setup ─────────────────────────────────────────────────────────────

app = Flask(__name__)

# Suppress Flask's default request logging to keep terminal clean.
import logging as _logging
_logging.getLogger("werkzeug").setLevel(_logging.ERROR)


# ── Shared state ──────────────────────────────────────────────────────────

_LOG_ENVS = ("pipeline", "camera", "detector", "udp", "isaac", "system")
_LOG_LEVELS = ("info", "warn", "error", "ok")
_MAX_LOG_PER_ENV = 200

_lock = threading.Lock()
_state = {
    "frames":      {"vis": None, "depth": None, "isaac": None},
    "detections":  [],
    "fps":         0.0,
    "frame_id":    0,
    "started_at":  time.time(),
    "paused":      False,
    "isaac_ok":    False,
    "status": {
        "pipeline": {"ok": True,  "detail": "running"},
        "camera":   {"ok": False, "detail": "waiting"},
        "detector": {"ok": False, "detail": "waiting"},
        "udp":      {"ok": False, "detail": "waiting"},
        "isaac":    {"ok": False, "detail": "waiting"},
        "system":   {"ok": True,  "detail": "online"},
    },
    "logs": {env: deque(maxlen=_MAX_LOG_PER_ENV) for env in _LOG_ENVS},
    "metrics": {
        "objects":     0,
        "latency_ms":  0.0,
        "dropped":     0,
        "udp_packets": 0,
        "cube_signals": 0,
    },
}

_controls: dict[str, Callable[[Optional[dict]], dict]] = {}

# UDP for Isaac Sim environment commands (separate from pose stream on :6000)
_isaac_cmd_host = "127.0.0.1"
_isaac_cmd_port = 6001
_isaac_cmd_sock: Optional[socket.socket] = None

# UDP trigger for cube spawn / pick-place request. This is intentionally
# separate from Isaac command UDP (:6001). The cube trigger goes to :6000.
_cube_signal_port = 6000
_cube_signal_sock: Optional[socket.socket] = None


# ── Public API (called from run*.py) ──────────────────────────────────────

def configure_isaac_commands(host: str = "127.0.0.1", port: int = 6001):
    """
    Set UDP host/port for Isaac Sim control buttons (sim_play, sim_pause, ...).
    Pose packets stay on your existing cube UDP socket; this is only for commands.
    Isaac extension should parse JSON: {\"type\":\"isaac_command\",\"command\":\"sim_play\"}.
    """
    global _isaac_cmd_host, _isaac_cmd_port, _isaac_cmd_sock
    _isaac_cmd_host = str(host)
    _isaac_cmd_port = int(port)
    if _isaac_cmd_sock is not None:
        try:
            _isaac_cmd_sock.close()
        except Exception:
            pass
        _isaac_cmd_sock = None
    push_log("isaac", f"Isaac command UDP set to {_isaac_cmd_host}:{_isaac_cmd_port}", level="info")



def configure_cube_signal(host: str = "127.0.0.1", port: int = 6000):
    """
    Set UDP destination for the manual "Send Cube Signal" button.

    This button does NOT send YOLO pose. It only sends a trigger packet:
        {"type": "cube_signal", "command": "spawn_cube"}
    Isaac Sim listens on this port, spawns/resets the demo cube, then waits
    for the dashboard Play button before executing pick-and-place.
    """
    global _isaac_cmd_host, _cube_signal_port, _cube_signal_sock
    _isaac_cmd_host = str(host)
    _cube_signal_port = int(port)
    if _cube_signal_sock is not None:
        try:
            _cube_signal_sock.close()
        except Exception:
            pass
        _cube_signal_sock = None
    push_log("udp", f"Cube signal UDP set to {_isaac_cmd_host}:{_cube_signal_port}", level="info")


def push_frame(key: str, frame_bgr: np.ndarray):
    """Update a live MJPEG frame for keys: 'vis' | 'depth' | 'isaac'."""
    with _lock:
        _state["frames"][key] = frame_bgr.copy()
        if key == "vis":
            _state["status"]["camera"] = {"ok": True, "detail": "RGB overlay streaming"}


def push_detections(rows: list, fps: float, frame_id: int, latency_ms: float = 0.0):
    """Push detection rows + per-frame telemetry from the main loop."""
    slim = []
    for r in rows:
        slim.append({
            "class":   r.get("class_name", ""),
            "conf":    round(float(r.get("confidence", 0)), 2),
            "cx":      round(float(r.get("cx", 0)), 3),
            "cy":      round(float(r.get("cy", 0)), 3),
            "cz":      round(float(r.get("cz", 0)), 3),
            "l_cm":    round(float(r.get("length_m", 0)) * 100, 1),
            "w_cm":    round(float(r.get("width_m",  0)) * 100, 1),
            "h_cm":    round(float(r.get("height_m", 0)) * 100, 1),
            "roll":    round(float(r.get("roll_deg",  0)), 1),
            "pitch":   round(float(r.get("pitch_deg", 0)), 1),
            "yaw":     round(float(r.get("yaw_deg",   0)), 1),
            "label":   r.get("cube_size_label", ""),
        })
    with _lock:
        _state["detections"]          = slim
        _state["fps"]                 = round(fps, 1)
        _state["frame_id"]            = int(frame_id)
        _state["metrics"]["objects"]  = len(slim)
        _state["metrics"]["latency_ms"] = round(float(latency_ms), 1)
        _state["status"]["detector"]  = {
            "ok": True,
            "detail": f"{len(slim)} obj | {round(fps, 1)} FPS",
        }


def push_udp_log(entry: str):
    """Backward-compat: append a UDP message (also routes to env='udp')."""
    push_log("udp", entry, level="info")
    with _lock:
        _state["metrics"]["udp_packets"] += 1
        _state["status"]["udp"] = {"ok": True, "detail": f"{_state['metrics']['udp_packets']} pkts"}


def set_isaac_connected(ok: bool):
    """Backward-compat: mark Isaac Sim as connected/disconnected."""
    with _lock:
        _state["isaac_ok"] = bool(ok)
        _state["status"]["isaac"] = {
            "ok": bool(ok),
            "detail": "connected" if ok else "waiting",
        }


def push_log(env: str, msg: str, level: str = "info"):
    """Append a log line to a specific environment channel."""
    if env not in _LOG_ENVS:
        env = "system"
    if level not in _LOG_LEVELS:
        level = "info"
    entry = {"t": time.strftime("%H:%M:%S"), "level": level, "msg": str(msg)}
    with _lock:
        _state["logs"][env].appendleft(entry)


def set_status(env: str, ok: bool, detail: str = ""):
    """Update the status indicator for an environment."""
    if env not in _LOG_ENVS:
        return
    with _lock:
        _state["status"][env] = {"ok": bool(ok), "detail": str(detail)}


def register_control(action: str, callback: Callable[[Optional[dict]], dict]):
    """Bind a UI button (action id) to a Python callback returning a dict."""
    _controls[action] = callback


def is_paused() -> bool:
    with _lock:
        return bool(_state["paused"])


def _send_isaac_command(cmd: str) -> bool:
    """Send a one-line JSON command to Isaac Sim (Omniverse extension / UDP listener)."""
    global _isaac_cmd_sock
    payload = {"type": "isaac_command", "command": str(cmd), "t": time.time()}
    data = json.dumps(payload).encode("utf-8")
    try:
        if _isaac_cmd_sock is None:
            _isaac_cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _isaac_cmd_sock.sendto(data, (_isaac_cmd_host, _isaac_cmd_port))
        push_log("isaac", f"Command sent: {cmd} → {_isaac_cmd_host}:{_isaac_cmd_port}", level="ok")
        with _lock:
            _state["status"]["isaac"] = {
                "ok": True,
                "detail": f"cmd {cmd} @ {_isaac_cmd_port}",
            }
        return True
    except OSError as e:
        push_log("isaac", f"Command send failed ({cmd}): {e}", level="error")
        return False


def _has_cube_detection() -> bool:
    """Return True when the current dashboard detection table contains a cube."""
    with _lock:
        dets = list(_state.get("detections", []))
    for d in dets:
        cls = str(d.get("class", "")).lower()
        lbl = str(d.get("label", "")).lower()
        if "cube" in cls or "cube" in lbl:
            return True
    return False


def _send_cube_signal() -> bool:
    """Send manual cube trigger to Isaac Sim on UDP :6000."""
    global _cube_signal_sock
    payload = {
        "type": "cube_signal",
        "command": "spawn_cube",
        "source": "dashboard_button",
        "t": time.time(),
    }
    data = json.dumps(payload).encode("utf-8")
    try:
        if _cube_signal_sock is None:
            _cube_signal_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _cube_signal_sock.sendto(data, (_isaac_cmd_host, _cube_signal_port))
        push_log("udp", f"Cube signal sent → {_isaac_cmd_host}:{_cube_signal_port}", level="ok")
        with _lock:
            _state["metrics"]["cube_signals"] += 1
            _state["status"]["udp"] = {
                "ok": True,
                "detail": f"cube signal #{_state['metrics']['cube_signals']}",
            }
        return True
    except OSError as e:
        push_log("udp", f"Cube signal send failed: {e}", level="error")
        return False


# ── Built-in control handlers (used if no custom one is registered) ───────

def _ctrl_send_cube_signal(_payload):
    """
    Dashboard button: only send the cube trigger after YOLO detects a cube.
    This keeps the required flow:
        detect cube -> user presses Send Cube Signal -> Isaac spawns cube
    """
    if not _has_cube_detection():
        push_log("udp", "Send Cube Signal blocked: no cube detection in current frame.", level="warn")
        return {"sent": False, "reason": "no_cube_detection"}
    ok = _send_cube_signal()
    return {"sent": ok, "port": _cube_signal_port}


def _ctrl_pause(_payload):
    with _lock:
        _state["paused"] = True
        _state["status"]["pipeline"] = {"ok": False, "detail": "paused"}
    push_log("pipeline", "Pipeline paused via dashboard", level="warn")
    return {"paused": True}


def _ctrl_resume(_payload):
    with _lock:
        _state["paused"] = False
        _state["status"]["pipeline"] = {"ok": True, "detail": "running"}
    push_log("pipeline", "Pipeline resumed via dashboard", level="ok")
    return {"paused": False}


def _ctrl_clear_logs(_payload):
    with _lock:
        for env in _LOG_ENVS:
            _state["logs"][env].clear()
    push_log("system", "All logs cleared", level="info")
    return {"cleared": True}


def _ctrl_reset_counters(_payload):
    with _lock:
        _state["metrics"]["dropped"] = 0
        _state["metrics"]["udp_packets"] = 0
        _state["frame_id"] = 0
        _state["started_at"] = time.time()
    push_log("system", "Counters reset", level="info")
    return {"reset": True}


def _ctrl_test_udp(_payload):
    push_udp_log("test packet from dashboard")
    return {"sent": True}


def _ctrl_ping_isaac(_payload):
    with _lock:
        ok = _state["isaac_ok"]
    push_log("isaac", "Ping requested" + (" - link OK" if ok else " - no response yet"),
             level="ok" if ok else "warn")
    return {"isaac_ok": ok}


def _ctrl_isaac_play(_p):
    ok = _send_isaac_command("sim_play")
    return {"sent": ok, "command": "sim_play"}


def _ctrl_isaac_pause(_p):
    ok = _send_isaac_command("sim_pause")
    return {"sent": ok, "command": "sim_pause"}


def _ctrl_isaac_stop(_p):
    ok = _send_isaac_command("sim_stop")
    return {"sent": ok, "command": "sim_stop"}


def _ctrl_isaac_reset(_p):
    ok = _send_isaac_command("scene_reset")
    return {"sent": ok, "command": "scene_reset"}


def _ctrl_isaac_step(_p):
    ok = _send_isaac_command("sim_step")
    return {"sent": ok, "command": "sim_step"}


_DEFAULT_CONTROLS = {
    "pause":          _ctrl_pause,
    "resume":         _ctrl_resume,
    "clear-logs":     _ctrl_clear_logs,
    "reset-counters": _ctrl_reset_counters,
    "test-udp":       _ctrl_test_udp,
    "send-cube-signal": _ctrl_send_cube_signal,
    "ping-isaac":     _ctrl_ping_isaac,
    "isaac-play":     _ctrl_isaac_play,
    "isaac-pause":    _ctrl_isaac_pause,
    "isaac-stop":     _ctrl_isaac_stop,
    "isaac-reset":    _ctrl_isaac_reset,
    "isaac-step":     _ctrl_isaac_step,
}


# ── Internal helpers ──────────────────────────────────────────────────────

_NO_SIGNAL: Optional[np.ndarray] = None


def _no_signal():
    global _NO_SIGNAL
    if _NO_SIGNAL is None:
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(img, "NO SIGNAL", (60, 128), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (70, 70, 70), 2, cv2.LINE_AA)
        _NO_SIGNAL = img
    return _NO_SIGNAL


def _encode(frame: Optional[np.ndarray], quality: int = 72) -> bytes:
    if frame is None:
        frame = _no_signal()
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _snapshot_logs():
    """Return logs as plain lists (deques aren't JSON-serializable)."""
    return {env: list(_state["logs"][env]) for env in _LOG_ENVS}


# ── Flask routes ──────────────────────────────────────────────────────────

def _mjpeg_gen(key: str):
    while True:
        with _lock:
            frame = _state["frames"].get(key)
        data = _encode(frame)
        yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        time.sleep(0.033)


@app.route("/video/<key>")
def video_feed(key):
    if key not in ("vis", "depth", "isaac"):
        return "Not found", 404
    return Response(_mjpeg_gen(key), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/api/push_isaac_frame", methods=["POST"])
def push_isaac_frame():
    """Isaac Sim POSTs viewport frames (JPEG or PNG bytes)."""
    data = request.get_data()
    if not data:
        return "empty", 400
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        try:
            from io import BytesIO

            from PIL import Image

            im = Image.open(BytesIO(data))
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            frame = rgb[:, :, ::-1].copy()
        except Exception:
            return "bad image", 400
    with _lock:
        _state["frames"]["isaac"] = frame
        _state["isaac_ok"] = True
        _state["status"]["isaac"] = {"ok": True, "detail": "streaming"}
    return "ok", 200


@app.route("/api/push_log", methods=["POST"])
def push_log_endpoint():
    """External services POST logs here as JSON: {env, msg, level}."""
    data = request.get_json(silent=True) or {}
    msg = data.get("msg")
    if not msg:
        return "bad request", 400
    push_log(data.get("env", "system"), msg, data.get("level", "info"))
    return "ok", 200


@app.route("/api/control/<action>", methods=["POST"])
def control_endpoint(action):
    payload = request.get_json(silent=True)
    handler = _controls.get(action) or _DEFAULT_CONTROLS.get(action)
    if handler is None:
        return jsonify({"ok": False, "error": f"unknown action: {action}"}), 404
    try:
        result = handler(payload) or {}
        return jsonify({"ok": True, "action": action, "result": result})
    except Exception as e:
        push_log("system", f"control '{action}' failed: {e}", level="error")
        return jsonify({"ok": False, "action": action, "error": str(e)}), 500


@app.route("/api/status")
def status_sse():
    def _gen():
        while True:
            with _lock:
                payload = {
                    "fps":         _state["fps"],
                    "frame_id":    _state["frame_id"],
                    "uptime_s":    int(time.time() - _state["started_at"]),
                    "paused":      _state["paused"],
                    "isaac_ok":    _state["isaac_ok"],
                    "detections":  _state["detections"],
                    "status":      dict(_state["status"]),
                    "metrics":     dict(_state["metrics"]),
                    "logs":        _snapshot_logs(),
                }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.5)

    return Response(
        _gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/")
def index():
    return Response(_DASHBOARD_HTML, mimetype="text/html")


# ── Server bootstrap ──────────────────────────────────────────────────────

def start(host: str = "0.0.0.0", port: int = 5000):
    t = threading.Thread(
        target=lambda: app.run(
            host=host, port=port,
            threaded=True, debug=False, use_reloader=False,
        ),
        daemon=True,
        name="dashboard-server",
    )
    t.start()
    push_log("system", f"Dashboard server started on {host}:{port}", level="ok")
    print(f"[Dashboard] Industrial UI at  http://{host}:{port}")
    return t


# ── Embedded HTML (industrial dark theme) ─────────────────────────────────

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>YOLO-3D Industrial Dashboard</title>
<style>
  :root {
    --bg-0:#0b1220; --bg-1:#111a2c; --bg-2:#172339; --bg-3:#1f2d49;
    --line:#243557; --text:#e2e8f0; --muted:#7f8ea8; --dim:#475569;
    --ok:#22c55e; --warn:#eab308; --err:#ef4444; --info:#38bdf8;
    --accent:#22d3ee; --accent-2:#a78bfa;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html, body { height:100%; }
  body {
    background:
      radial-gradient(1200px 600px at 80% -10%, rgba(56,189,248,.06), transparent 60%),
      radial-gradient(900px 500px at -10% 110%, rgba(167,139,250,.05), transparent 60%),
      var(--bg-0);
    color:var(--text);
    font-family:'Inter','Segoe UI',system-ui,sans-serif;
    overflow:hidden;
    -webkit-font-smoothing:antialiased;
  }
  .card {
    background:linear-gradient(180deg, var(--bg-2), var(--bg-1));
    border:1px solid var(--line);
    border-radius:.75rem;
    box-shadow:0 1px 0 rgba(255,255,255,.03) inset, 0 6px 24px rgba(0,0,0,.25);
  }
  .lbl {
    font-size:.66rem; font-weight:700; color:var(--muted);
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px;
    display:flex; align-items:center; gap:6px;
  }
  .feed {
    width:100%; aspect-ratio:4/3; object-fit:contain;
    background:#000; border-radius:.4rem; display:block;
  }
  /* Large primary RGB stream (fills column — no depth panel) */
  .feed-rgb-main {
    flex:1; width:100%; min-height:0; height:100%;
    object-fit:contain; background:#000; border-radius:.4rem; display:block;
  }
  table { width:100%; border-collapse:collapse; font-size:.78rem; }
  th {
    background:var(--bg-3); color:var(--muted); padding:6px 8px; text-align:left;
    border-bottom:1px solid var(--line); white-space:nowrap;
    font-weight:600; font-size:.7rem; letter-spacing:.04em; text-transform:uppercase;
  }
  td { padding:5px 8px; border-bottom:1px solid var(--bg-3); white-space:nowrap; font-family:'JetBrains Mono','Consolas',monospace; }
  tr:hover td { background:rgba(34,211,238,.05); }
  .scroll { overflow-y:auto; }
  ::-webkit-scrollbar { width:6px; height:6px; }
  ::-webkit-scrollbar-thumb { background:var(--bg-3); border-radius:9999px; }
  ::-webkit-scrollbar-thumb:hover { background:var(--line); }

  .pill {
    display:inline-flex; align-items:center; gap:6px;
    padding:3px 9px; border-radius:9999px; font-size:.7rem; font-weight:600;
    background:var(--bg-3); border:1px solid var(--line);
  }
  .pill-dot { width:7px; height:7px; border-radius:50%; background:var(--dim); }
  .pill.ok   .pill-dot { background:var(--ok); box-shadow:0 0 8px var(--ok); }
  .pill.warn .pill-dot { background:var(--warn); }
  .pill.err  .pill-dot { background:var(--err); }

  .btn {
    appearance:none; border:1px solid var(--line); cursor:pointer;
    background:linear-gradient(180deg, var(--bg-3), var(--bg-2));
    color:var(--text); font-size:.78rem; font-weight:600;
    padding:7px 12px; border-radius:.45rem; display:inline-flex; align-items:center; gap:6px;
    transition:transform .05s ease, background .15s ease, border-color .15s ease;
  }
  .btn:hover { border-color:var(--accent); color:#fff; }
  .btn:active { transform:translateY(1px); }
  .btn.primary { background:linear-gradient(180deg, #0ea5e9, #0284c7); border-color:#0284c7; color:#fff; }
  .btn.primary:hover { background:linear-gradient(180deg, #38bdf8, #0ea5e9); }
  .btn.warn { background:linear-gradient(180deg, #f59e0b, #d97706); border-color:#d97706; color:#fff; }
  .btn.warn:hover { background:linear-gradient(180deg, #fbbf24, #f59e0b); }
  .btn.danger { background:linear-gradient(180deg, #ef4444, #b91c1c); border-color:#b91c1c; color:#fff; }
  .btn.danger:hover { background:linear-gradient(180deg, #f87171, #ef4444); }
  .btn[disabled] { opacity:.45; cursor:not-allowed; }

  .tag {
    display:inline-block; padding:1px 6px; border-radius:4px;
    font-size:.72rem; font-weight:700; background:#1d4ed8; color:#bfdbfe;
  }
  .stat-box { padding:10px 12px; border-radius:.55rem; background:var(--bg-1); border:1px solid var(--line); }
  .stat-num { font-size:1.6rem; font-weight:700; font-family:'JetBrains Mono',monospace; line-height:1; }
  .stat-lbl { font-size:.62rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-top:4px; }

  .tabs { display:flex; gap:4px; flex-wrap:wrap; }
  .tab {
    padding:5px 10px; font-size:.72rem; font-weight:600; border-radius:.4rem;
    cursor:pointer; color:var(--muted); border:1px solid transparent;
  }
  .tab:hover { color:var(--text); }
  .tab.active {
    background:var(--bg-3); color:var(--accent); border-color:var(--line);
  }

  .log-row {
    font-size:.74rem; padding:3px 6px; border-radius:.25rem;
    font-family:'JetBrains Mono','Consolas',monospace;
    display:flex; gap:8px; align-items:flex-start; line-height:1.4;
  }
  .log-row + .log-row { border-top:1px solid rgba(255,255,255,.03); }
  .log-row .lvl {
    font-size:.62rem; padding:1px 6px; border-radius:3px; flex-shrink:0;
    font-weight:700; letter-spacing:.04em; text-transform:uppercase;
  }
  .log-row .lvl.info  { background:#0c4a6e; color:#bae6fd; }
  .log-row .lvl.ok    { background:#14532d; color:#86efac; }
  .log-row .lvl.warn  { background:#78350f; color:#fde68a; }
  .log-row .lvl.error { background:#7f1d1d; color:#fecaca; }
  .log-row .t   { color:var(--dim); flex-shrink:0; }
  .log-row .msg { color:var(--text); white-space:pre-wrap; word-break:break-word; }

  .header {
    background:linear-gradient(180deg, var(--bg-2), var(--bg-1));
    border-bottom:1px solid var(--line);
    display:flex; align-items:center; justify-content:space-between;
    padding:8px 16px; height:48px;
  }
  .brand { display:flex; align-items:center; gap:10px; }
  .brand .logo {
    width:26px; height:26px; border-radius:6px;
    background:conic-gradient(from 210deg, #22d3ee, #a78bfa, #22d3ee);
    box-shadow:0 0 14px rgba(34,211,238,.45);
  }
  .brand .name { font-size:.95rem; font-weight:700; letter-spacing:.02em; }
  .brand .sub  { font-size:.66rem; color:var(--muted); margin-top:1px; }

  .toolbar {
    background:var(--bg-1); border-bottom:1px solid var(--line);
    display:flex; align-items:center; gap:8px; padding:8px 12px; flex-wrap:wrap;
  }
  .toolbar .group {
    display:flex; align-items:center; gap:6px; padding-right:10px;
    border-right:1px solid var(--line); margin-right:6px;
  }
  .toolbar .group:last-child { border-right:none; }
  .toolbar .group .glabel {
    font-size:.6rem; color:var(--muted); text-transform:uppercase;
    letter-spacing:.08em; margin-right:4px;
  }

  .grid {
    display:grid; grid-template-columns:1.5fr 0.95fr 1.05fr; gap:10px;
    padding:10px; height:calc(100vh - 48px - 50px); overflow:hidden;
  }
  .col { display:flex; flex-direction:column; gap:10px; min-height:0; }
  .col > .card.flex { flex:1; min-height:0; display:flex; flex-direction:column; }
  .pad-10 { padding:10px; }
  .pad-8  { padding:8px; }
  .row3 { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:8px; }
  .row2 { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:8px; }
  .row4 { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:8px; }

  .isaac-btn-row { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
  .isaac-hint {
    font-size:.58rem; color:var(--muted); margin-top:8px; line-height:1.4;
    border-top:1px solid var(--line); padding-top:8px;
  }
</style>
</head>
<body>

<!-- ===== Header ===== -->
<div class="header">
  <div class="brand">
    <div class="logo"></div>
    <div>
      <div class="name">YOLO-3D Industrial Console</div>
      <div class="sub">Edge perception &middot; pose estimation &middot; Isaac Sim bridge</div>
    </div>
  </div>
  <div style="display:flex; align-items:center; gap:8px;">
    <span id="pill-pipeline" class="pill ok"><span class="pill-dot"></span>Pipeline</span>
    <span id="pill-camera"   class="pill"><span class="pill-dot"></span>Camera</span>
    <span id="pill-detector" class="pill"><span class="pill-dot"></span>Detector</span>
    <span id="pill-udp"      class="pill"><span class="pill-dot"></span>UDP</span>
    <span id="pill-isaac"    class="pill"><span class="pill-dot"></span>Isaac Sim</span>
    <span id="pill-system"   class="pill ok"><span class="pill-dot"></span>System</span>
  </div>
</div>

<!-- ===== Toolbar ===== -->
<div class="toolbar">
  <div class="group">
    <span class="glabel">Pipeline</span>
    <button class="btn primary" data-action="resume">&#9654; Resume</button>
    <button class="btn warn"    data-action="pause">&#10073;&#10073; Pause</button>
    <button class="btn"         data-action="reset-counters">&#8635; Reset</button>
  </div>
  <div class="group">
    <span class="glabel">Bridge</span>
    <button class="btn primary" data-action="send-cube-signal">&#129482; Send Cube Signal</button>
    <button class="btn"         data-action="test-udp">&#128231; Test UDP</button>
    <button class="btn"         data-action="ping-isaac">&#128225; Ping Isaac</button>
  </div>
  <div class="group">
    <span class="glabel">Isaac Sim</span>
    <button class="btn primary" data-action="isaac-play">Play</button>
    <button class="btn warn"    data-action="isaac-pause">Pause</button>
    <button class="btn"         data-action="isaac-stop">Stop</button>
    <button class="btn"         data-action="isaac-reset">Reset scene</button>
    <button class="btn"         data-action="isaac-step">Step</button>
  </div>
  <div class="group">
    <span class="glabel">Logs</span>
    <button class="btn danger"  data-action="clear-logs">&#10007; Clear Logs</button>
  </div>
  <div style="margin-left:auto; display:flex; align-items:center; gap:14px; font-size:.74rem; color:var(--muted);">
    <span>Uptime: <span id="uptime" style="color:var(--text); font-family:monospace;">00:00:00</span></span>
    <span>Frame: <span id="frame-id" style="color:var(--text); font-family:monospace;">0</span></span>
    <span>FPS: <span id="fps-top" style="color:var(--accent); font-family:monospace; font-weight:700;">--</span></span>
    <span id="conn-dot" style="width:9px; height:9px; border-radius:50%; background:#eab308; display:inline-block;" title="SSE connection"></span>
  </div>
</div>

<!-- ===== Main grid ===== -->
<div class="grid">

  <!-- Col 1: Large RGB overlay only (depth not shown) -->
  <div class="col">
    <div class="card pad-8" style="flex:1; min-height:0; display:flex; flex-direction:column;">
      <div class="lbl">RGB &mdash; pose / size / GT overlay</div>
      <img class="feed-rgb-main" src="/video/vis" alt="RGB pose overlay"/>
    </div>
  </div>

  <!-- Col 2: Telemetry / detections -->
  <div class="col">

    <div class="card pad-10">
      <div class="row4">
        <div class="stat-box">
          <div id="m-fps"  class="stat-num" style="color:var(--accent);">--</div>
          <div class="stat-lbl">Pipeline FPS</div>
        </div>
        <div class="stat-box">
          <div id="m-obj"  class="stat-num" style="color:#4ade80;">0</div>
          <div class="stat-lbl">Objects</div>
        </div>
        <div class="stat-box">
          <div id="m-lat"  class="stat-num" style="color:#fde68a;">--</div>
          <div class="stat-lbl">Latency (ms)</div>
        </div>
        <div class="stat-box">
          <div id="m-udp"  class="stat-num" style="color:#a78bfa;">0</div>
          <div class="stat-lbl">UDP Packets</div>
        </div>
      </div>
    </div>

    <div class="card pad-10 flex">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <div class="lbl" style="margin:0;">&#128269; Detections</div>
        <span id="det-cnt" style="font-size:.7rem; color:var(--muted);">&mdash;</span>
      </div>
      <div class="scroll" style="flex:1; min-height:0;">
        <table>
          <thead><tr>
            <th>Class</th><th>Conf</th><th>Distance</th>
            <th>L &times; W &times; H (cm)</th><th>RPY (&deg;)</th>
          </tr></thead>
          <tbody id="det-body">
            <tr><td colspan="5" style="color:var(--dim); text-align:center; padding:18px;">
              Waiting for detections&hellip;
            </td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card pad-10">
      <div class="lbl">&#127760; Primary Object Pose (camera frame, m)</div>
      <div class="row3">
        <div class="stat-box">
          <div id="p-x" class="stat-num" style="color:var(--accent);">--</div>
          <div class="stat-lbl">X</div>
        </div>
        <div class="stat-box">
          <div id="p-y" class="stat-num" style="color:var(--accent);">--</div>
          <div class="stat-lbl">Y</div>
        </div>
        <div class="stat-box">
          <div id="p-z" class="stat-num" style="color:var(--accent);">--</div>
          <div class="stat-lbl">Z (depth)</div>
        </div>
      </div>
    </div>

  </div>

  <!-- Col 3: Isaac viewport + controls + Multi-tab logs -->
  <div class="col">
    <div class="card pad-8">
      <div class="lbl">&#129302; Isaac Sim viewport</div>
      <img class="feed" src="/video/isaac" alt="isaac"/>
    </div>

    <div class="card pad-8">
      <div class="lbl">Isaac Sim environment</div>
      <div class="isaac-btn-row">
        <button class="btn primary" data-action="isaac-play">Play simulation</button>
        <button class="btn warn"    data-action="isaac-pause">Pause simulation</button>
        <button class="btn"         data-action="isaac-stop">Stop simulation</button>
        <button class="btn"         data-action="isaac-reset">Reset scene</button>
        <button class="btn"         data-action="isaac-step">Step one frame</button>
      </div>
      <div class="isaac-hint">
        Commands are sent as UDP JSON to the host configured in Python (<code>configure_isaac_commands</code>),
        default <strong>127.0.0.1:6001</strong> (separate from pose traffic on port 6000).
        Payload shape: <code>{"type":"isaac_command","command":"sim_play",...}</code>
        &mdash; handle these in your Isaac Sim extension.
      </div>
    </div>

    <div class="card pad-10 flex">
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
        <div class="lbl" style="margin:0;">&#128221; Environment Logs</div>
        <span id="log-count" style="font-size:.66rem; color:var(--muted); font-family:monospace;">0 entries</span>
      </div>
      <div class="tabs" id="log-tabs">
        <div class="tab active" data-env="pipeline">Pipeline</div>
        <div class="tab" data-env="camera">Camera</div>
        <div class="tab" data-env="detector">Detector</div>
        <div class="tab" data-env="udp">UDP</div>
        <div class="tab" data-env="isaac">Isaac</div>
        <div class="tab" data-env="system">System</div>
      </div>
      <div id="log-view" class="scroll" style="flex:1; min-height:0; margin-top:8px;
            background:var(--bg-0); border:1px solid var(--line); border-radius:.45rem; padding:6px;">
        <div style="color:var(--dim); font-size:.74rem; padding:8px;">No log entries yet&hellip;</div>
      </div>
    </div>
  </div>

</div>

<script>
let CURRENT_LOG_ENV = "pipeline";
let LATEST_LOGS = {};

const es = new EventSource("/api/status");
es.onopen  = () => { document.getElementById("conn-dot").style.background = "#22c55e"; };
es.onerror = () => { document.getElementById("conn-dot").style.background = "#ef4444"; };

es.onmessage = (ev) => {
  const d = JSON.parse(ev.data);
  document.getElementById("fps-top").textContent  = d.fps;
  document.getElementById("frame-id").textContent = d.frame_id;
  document.getElementById("uptime").textContent   = fmtUptime(d.uptime_s);

  document.getElementById("m-fps").textContent = d.fps;
  document.getElementById("m-obj").textContent = (d.metrics && d.metrics.objects) || 0;
  document.getElementById("m-lat").textContent = (d.metrics && d.metrics.latency_ms) || 0;
  document.getElementById("m-udp").textContent = (d.metrics && d.metrics.udp_packets) || 0;

  updateStatus(d.status || {}, d.paused);
  updateDetections(d.detections || []);

  LATEST_LOGS = d.logs || {};
  renderLogs();
};

function fmtUptime(s) {
  s = Math.max(0, s|0);
  const h = String((s/3600|0)).padStart(2,"0");
  const m = String(((s%3600)/60|0)).padStart(2,"0");
  const ss = String(s%60).padStart(2,"0");
  return `${h}:${m}:${ss}`;
}

function updateStatus(st, paused) {
  ["pipeline","camera","detector","udp","isaac","system"].forEach(env => {
    const pill = document.getElementById("pill-" + env);
    if (!pill) return;
    const info = st[env] || {ok:false, detail:"unknown"};
    pill.classList.remove("ok","warn","err");
    if (env === "pipeline" && paused) {
      pill.classList.add("warn");
      pill.lastChild.textContent = "Pipeline (paused)";
    } else {
      pill.classList.add(info.ok ? "ok" : "warn");
      const label = env === "isaac" ? "Isaac Sim"
                  : env.charAt(0).toUpperCase() + env.slice(1);
      pill.lastChild.textContent = `${label}: ${info.detail || (info.ok ? "ok" : "waiting")}`;
    }
  });
}

function updateDetections(dets) {
  const cnt  = document.getElementById("det-cnt");
  const body = document.getElementById("det-body");
  if (!dets.length) {
    cnt.textContent = "\u2014";
    body.innerHTML  = '<tr><td colspan="5" style="color:var(--dim); text-align:center; padding:18px;">No detections</td></tr>';
    ["p-x","p-y","p-z"].forEach(id => document.getElementById(id).textContent = "--");
    return;
  }
  cnt.textContent = dets.length + " object(s)";
  body.innerHTML  = dets.map(r => {
    const lbl = r.label ? '<span class="tag">'+esc(r.label)+'</span> ' : "";
    return '<tr>'
      + '<td>'+lbl+'<span style="color:var(--text)">'+esc(r["class"])+'</span></td>'
      + '<td style="color:#fde68a">'+r.conf+'</td>'
      + '<td style="color:var(--accent)">'+r.cz+' m</td>'
      + '<td style="color:#cbd5e1">'+r.l_cm+'\u00d7'+r.w_cm+'\u00d7'+r.h_cm+'</td>'
      + '<td style="color:var(--muted)">'+r.roll+', '+r.pitch+', '+r.yaw+'</td>'
      + '</tr>';
  }).join("");
  const f = dets[0];
  document.getElementById("p-x").textContent = f.cx;
  document.getElementById("p-y").textContent = f.cy;
  document.getElementById("p-z").textContent = f.cz;
}

// ---- Logs ----
document.querySelectorAll("#log-tabs .tab").forEach(t => {
  t.addEventListener("click", () => {
    document.querySelectorAll("#log-tabs .tab").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    CURRENT_LOG_ENV = t.dataset.env;
    renderLogs();
  });
});

function renderLogs() {
  const env  = CURRENT_LOG_ENV;
  const list = LATEST_LOGS[env] || [];
  const view = document.getElementById("log-view");
  document.getElementById("log-count").textContent = list.length + " entries";
  if (!list.length) {
    view.innerHTML = '<div style="color:var(--dim); font-size:.74rem; padding:8px;">No log entries yet&hellip;</div>';
    return;
  }
  view.innerHTML = list.map(e => {
    const lvl = (e.level || "info").toLowerCase();
    return '<div class="log-row">'
      + '<span class="t">'+esc(e.t)+'</span>'
      + '<span class="lvl '+lvl+'">'+esc(lvl)+'</span>'
      + '<span class="msg">'+esc(e.msg)+'</span>'
      + '</div>';
  }).join("");
}

// ---- Buttons ----
document.querySelectorAll(".btn[data-action]").forEach(b => {
  b.addEventListener("click", async () => {
    const act = b.dataset.action;
    b.disabled = true;
    try {
      const r = await fetch("/api/control/" + act, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: "{}",
      });
      const j = await r.json();
      if (!j.ok) console.warn("control failed:", j);
    } catch (err) {
      console.error(err);
    } finally {
      setTimeout(() => { b.disabled = false; }, 250);
    }
  });
});

function esc(s) { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
</script>
</body>
</html>
"""
