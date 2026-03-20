"""
Edge Dashboard — Flask web server for YOLO-3D monitoring.

Exposes:
  /                       → HTML dashboard
  /video/vis              → MJPEG stream: annotated pose/detection frame
  /video/depth            → MJPEG stream: depth colormap (JET)
  /video/isaac            → MJPEG stream: Isaac Sim viewport (pushed via POST)
  /api/push_isaac_frame   → POST endpoint: Isaac Sim pushes JPEG frames here
  /api/status             → SSE stream: detections, UDP log
"""

import json
import threading
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, request

app = Flask(__name__)

# ── Shared state ──────────────────────────────────────────────────────────
_lock = threading.Lock()
_state = {
    "frames":      {"vis": None, "depth": None, "isaac": None},
    "detections":  [],
    "fps":         0.0,
    "frame_id":    0,
    "udp_log":     [],
    "isaac_ok":    False,
}

# ── Public API (called from run.py) ───────────────────────────────────────

def push_frame(key: str, frame_bgr: np.ndarray):
    """Update a live video frame (vis | depth)."""
    with _lock:
        _state["frames"][key] = frame_bgr.copy()


def push_detections(rows: list, fps: float, frame_id: int):
    """Push detection rows from run.py main loop."""
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
        _state["detections"] = slim
        _state["fps"]        = round(fps, 1)
        _state["frame_id"]   = frame_id


def push_udp_log(entry: str):
    """Append a message to the Isaac Sim UDP log."""
    with _lock:
        _state["udp_log"].insert(0, {"t": time.strftime("%H:%M:%S"), "msg": entry})
        if len(_state["udp_log"]) > 40:
            _state["udp_log"].pop()


def set_isaac_connected(ok: bool):
    with _lock:
        _state["isaac_ok"] = ok


# ── Internal helpers ──────────────────────────────────────────────────────

_NO_SIGNAL: Optional[np.ndarray] = None


def _no_signal():
    global _NO_SIGNAL
    if _NO_SIGNAL is None:
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(img, "No Signal", (68, 128), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (70, 70, 70), 2, cv2.LINE_AA)
        _NO_SIGNAL = img
    return _NO_SIGNAL


def _encode(frame: Optional[np.ndarray], quality: int = 72) -> bytes:
    if frame is None:
        frame = _no_signal()
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


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
    return Response(
        _mjpeg_gen(key),
        mimetype="multipart/x-mixed-replace; boundary=f",
    )


@app.route("/api/push_isaac_frame", methods=["POST"])
def push_isaac_frame():
    """Isaac Sim POSTs JPEG viewport frames here."""
    data = request.get_data()
    if not data:
        return "empty", 400
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return "bad image", 400
    with _lock:
        _state["frames"]["isaac"] = frame
        _state["isaac_ok"] = True
    return "ok", 200


@app.route("/api/push_log", methods=["POST"])
def push_log_endpoint():
    """Isaac Sim POSTs log messages here."""
    data = request.get_json(silent=True)
    if not data or "msg" not in data:
        return "bad request", 400
    push_udp_log(data["msg"])
    return "ok", 200


@app.route("/api/status")
def status_sse():
    def _gen():
        while True:
            with _lock:
                payload = {
                    "fps":        _state["fps"],
                    "frame_id":   _state["frame_id"],
                    "detections": _state["detections"],
                    "udp_log":    _state["udp_log"],
                    "isaac_ok":   _state["isaac_ok"],
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


# ── Start server in background thread ─────────────────────────────────────

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
    print(f"[Dashboard] Running at  http://{host}:{port}")
    return t


# ── Embedded HTML ─────────────────────────────────────────────────────────

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>YOLO-3D Edge Dashboard</title>
<style>
  * { box-sizing: border-box; margin:0; padding:0; }
  body { background:#0f172a; color:#e2e8f0; font-family:'Segoe UI',system-ui,sans-serif; overflow:hidden; }
  .card { background:#1e293b; border:1px solid #334155; border-radius:.75rem; }
  img.feed { width:100%; border-radius:.4rem; background:#0f172a; display:block; min-height:60px; }
  table { width:100%; border-collapse:collapse; font-size:.78rem; }
  th { background:#162032; color:#94a3b8; padding:5px 7px; text-align:left;
       border-bottom:1px solid #334155; white-space:nowrap; }
  td { padding:4px 7px; border-bottom:1px solid #1e293b; white-space:nowrap; }
  tr:hover td { background:#1c2e45; }
  .scroll  { overflow-y:auto; }
  ::-webkit-scrollbar { width:4px; } ::-webkit-scrollbar-thumb { background:#334155; border-radius:9999px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:9999px; font-size:.7rem; font-weight:600; }
  .ok   { background:#14532d; color:#86efac; }
  .warn { background:#78350f; color:#fde68a; }
  .tag  { display:inline-block; padding:1px 6px; border-radius:4px;
          font-size:.72rem; font-weight:700; background:#1d4ed8; color:#bfdbfe; }
  .log-row { font-size:.73rem; padding:2px 0; border-bottom:1px solid #1e293b;
             font-family:monospace; display:flex; gap:6px; }
  .log-t { color:#475569; flex-shrink:0; }
  .lbl { font-size:.68rem; font-weight:600; color:#64748b; text-transform:uppercase;
         letter-spacing:.05em; margin-bottom:6px; }
</style>
</head>
<body>

<!-- Header -->
<div style="background:#1e293b;border-bottom:1px solid #334155;
     display:flex;align-items:center;justify-content:space-between;padding:8px 20px;height:46px;">
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:1.05rem;font-weight:700;color:#22d3ee;">&#11042; YOLO-3D Edge Dashboard</span>
  </div>
  <div style="display:flex;align-items:center;gap:14px;font-size:.8rem;">
    <span id="fps-badge" class="badge ok">FPS: --</span>
    <span id="frame-badge" style="color:#64748b;font-family:monospace;font-size:.72rem;">Frame #--</span>
    <span id="isaac-badge" class="badge warn">Isaac: waiting</span>
    <span id="dot" style="width:9px;height:9px;border-radius:50%;background:#eab308;display:inline-block;"></span>
  </div>
</div>

<!-- Main 3-column grid -->
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;
     padding:10px;height:calc(100vh - 46px);overflow:hidden;">

  <!-- Col 1: Camera feeds -->
  <div style="display:flex;flex-direction:column;gap:8px;overflow-y:auto;">
    <div class="card" style="padding:8px;">
      <div class="lbl">&#128247; Pose + Size + GT Metrics</div>
      <img class="feed" src="/video/vis" alt="Vis"/>
    </div>
    <div class="card" style="padding:8px;">
      <div class="lbl">&#127752; Depth Map</div>
      <img class="feed" src="/video/depth" alt="Depth"/>
    </div>
  </div>

  <!-- Col 2: Detections + stats -->
  <div style="display:flex;flex-direction:column;gap:8px;overflow-y:auto;">
    <div class="card" style="padding:10px;flex:1;min-height:0;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <div class="lbl" style="margin:0;">&#128269; Detections</div>
        <span id="det-cnt" style="font-size:.7rem;color:#475569;">—</span>
      </div>
      <div class="scroll" style="max-height:180px;">
        <table>
          <thead><tr>
            <th>Class</th><th>Conf</th><th>Dist</th>
            <th>L×W×H cm</th><th>RPY °</th>
          </tr></thead>
          <tbody id="det-body">
            <tr><td colspan="5" style="color:#475569;text-align:center;padding:16px;">
              Waiting for detections…</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card" style="padding:10px;">
      <div class="lbl">&#127760; 3D Position (m)</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;text-align:center;">
        <div>
          <div style="font-size:.68rem;color:#475569;">X</div>
          <div id="p-x" style="font-family:monospace;font-size:1rem;color:#22d3ee;">--</div>
        </div>
        <div>
          <div style="font-size:.68rem;color:#475569;">Y</div>
          <div id="p-y" style="font-family:monospace;font-size:1rem;color:#22d3ee;">--</div>
        </div>
        <div>
          <div style="font-size:.68rem;color:#475569;">Z (depth)</div>
          <div id="p-z" style="font-family:monospace;font-size:1.05rem;font-weight:700;color:#22d3ee;">--</div>
        </div>
      </div>
    </div>

    <div class="card" style="padding:10px;">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;text-align:center;">
        <div>
          <div style="font-size:.68rem;color:#475569;margin-bottom:2px;">Pipeline FPS</div>
          <div id="st-fps" style="font-size:1.8rem;font-weight:700;font-family:monospace;color:#22d3ee;">--</div>
        </div>
        <div>
          <div style="font-size:.68rem;color:#475569;margin-bottom:2px;">Objects</div>
          <div id="st-obj" style="font-size:1.8rem;font-weight:700;font-family:monospace;color:#4ade80;">--</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Col 3: Isaac Sim viewport + UDP log -->
  <div style="display:flex;flex-direction:column;gap:8px;overflow-y:auto;">
    <div class="card" style="padding:8px;">
      <div class="lbl">&#129302; Isaac Sim Environment</div>
      <img class="feed" src="/video/isaac" alt="Isaac Sim"/>
    </div>

    <div class="card" style="padding:10px;flex:1;min-height:0;display:flex;flex-direction:column;">
      <div class="lbl">&#128228; Isaac Sim UDP Log</div>
      <div id="udp-log" class="scroll" style="flex:1;min-height:0;">
        <div style="color:#475569;font-size:.73rem;padding:8px 0;">No messages yet…</div>
      </div>
    </div>
  </div>

</div>

<script>
const es = new EventSource("/api/status");
es.onopen  = () => { document.getElementById("dot").style.background = "#22c55e"; };
es.onerror = () => { document.getElementById("dot").style.background = "#ef4444"; };

es.onmessage = (ev) => {
  const d = JSON.parse(ev.data);
  document.getElementById("fps-badge").textContent   = "FPS: " + d.fps;
  document.getElementById("frame-badge").textContent = "Frame #" + d.frame_id;
  document.getElementById("st-fps").textContent = d.fps;
  document.getElementById("st-obj").textContent = (d.detections || []).length;

  const ib = document.getElementById("isaac-badge");
  if (d.isaac_ok) { ib.textContent="Isaac: connected"; ib.className="badge ok"; }
  else            { ib.textContent="Isaac: waiting";    ib.className="badge warn"; }

  updateDetections(d.detections || []);
  updateLog(d.udp_log || []);
};

function updateDetections(dets) {
  const cnt  = document.getElementById("det-cnt");
  const body = document.getElementById("det-body");
  if (!dets.length) {
    cnt.textContent = "\u2014";
    body.innerHTML  = '<tr><td colspan="5" style="color:#475569;text-align:center;padding:14px;">No detections</td></tr>';
    ["p-x","p-y","p-z"].forEach(id => document.getElementById(id).textContent = "--");
    return;
  }
  cnt.textContent = dets.length + " object(s)";
  body.innerHTML  = dets.map(r => {
    const lbl = r.label ? '<span class="tag">'+esc(r.label)+'</span> ' : "";
    return '<tr>'
      + '<td>'+lbl+'<span style="color:#e2e8f0">'+esc(r["class"])+'</span></td>'
      + '<td style="font-family:monospace;color:#fde68a">'+r.conf+'</td>'
      + '<td style="font-family:monospace;color:#22d3ee">'+r.cz+'m</td>'
      + '<td style="font-family:monospace;color:#cbd5e1">'+r.l_cm+'\u00d7'+r.w_cm+'\u00d7'+r.h_cm+'</td>'
      + '<td style="font-family:monospace;color:#94a3b8">'+r.roll+','+r.pitch+','+r.yaw+'</td>'
      + '</tr>';
  }).join("");
  const f = dets[0];
  document.getElementById("p-x").textContent = f.cx;
  document.getElementById("p-y").textContent = f.cy;
  document.getElementById("p-z").textContent = f.cz;
}

function updateLog(logs) {
  if (!logs.length) return;
  document.getElementById("udp-log").innerHTML = logs.map(l =>
    '<div class="log-row"><span class="log-t">'+esc(l.t)+'</span>'
    + '<span style="color:#4ade80">'+esc(l.msg)+'</span></div>'
  ).join("");
}

function esc(s) { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
</script>
</body>
</html>
"""
