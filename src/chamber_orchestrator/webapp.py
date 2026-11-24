"""编曲试听界面 HTTP 服务（纯标准库）。

启动：python3 -m chamber_orchestrator.webapp --host 127.0.0.1 --port 8000
所有 /api/orchestrate 请求都实时调用约束求解器，不使用固定数据冒充结果。
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .examples import EXAMPLE_ENSEMBLE, EXAMPLES
from .instruments import CATALOG, ENSEMBLE_PRESETS
from .orchestrate import orchestrate
from .score import (ROLE_BASS, ROLE_INNER, ROLE_MELODY, Score, make_line)

_STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
_SESSION_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output",
    "sessions")


def _instrument_catalog_payload() -> list[dict]:
    return [{
        "id": i.id, "name": i.name, "family": i.family,
        "low": i.low, "high": i.high,
        "written_offset": i.written_offset,
        "polyphony": i.polyphony,
        "max_sustain_s": i.max_sustain_s,
        "phrase_budget_s": i.phrase_budget_s,
        "breath_gap_s": i.breath_gap_s,
        "min_attack_gap_s": i.min_attack_gap_s,
        "recovery_s": i.recovery_s,
        "change_time_s": i.change_time_s,
        "timbre": i.timbre,
    } for i in CATALOG.values()]


def _score_from_request(data: dict) -> Score:
    bpm = float(data.get("bpm", 96))
    layers = []
    role_map = {"melody": ROLE_MELODY, "inner": ROLE_INNER, "bass": ROLE_BASS}
    for layer in data.get("layers", []):
        entries = [(float(n["start"]), float(n["duration"]), n.get("pitch"))
                   for n in layer.get("notes", [])]
        lock = None
        if layer.get("register_lock"):
            lock = (layer["register_lock"][0], layer["register_lock"][1])
        layers.append(make_line(layer["id"], role_map[layer["role"]],
                                layer.get("name", layer["id"]), bpm,
                                entries, register_lock=lock))
    return Score(data.get("title", "界面自定义谱"), bpm, layers,
                 int(data.get("beats_per_bar", 4)))


def _ensemble_from_request(data: dict):
    ensemble = []
    for spec in data.get("ensemble", []):
        ids = [i for i in spec.get("instruments", []) if i in CATALOG]
        if ids:
            ensemble.append((spec.get("player") or f"p{len(ensemble) + 1}",
                             ids))
    return ensemble


class Handler(BaseHTTPRequestHandler):
    server_version = "ChamberOrchestrator/1.0"

    def log_message(self, fmt, *args):  # 静音默认日志，保持界面输出整洁
        sys.stderr.write("[web] " + fmt % args + "\n")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/media/"):
            return self._serve_session_media(path)
        if path.startswith("/api/"):
            return self._handle_api_get(path)
        return self._serve_static(path)

    def _handle_api_get(self, path: str):
        if path == "/api/catalog":
            return self._send_json({
                "instruments": _instrument_catalog_payload(),
                "presets": {k: [{"player": p, "instruments": ids}
                                for p, ids in v]
                            for k, v in ENSEMBLE_PRESETS.items()},
                "examples": [{"id": k, "label": label}
                             for k, (_, label) in EXAMPLES.items()],
                "example_ensemble_map": {
                    k: [{"player": p, "instruments": ids}
                        for p, ids in v] for k, v in EXAMPLE_ENSEMBLE.items()},
            })
        if path == "/api/example":
            ex_id = self.query_get("id", "long_tone")
            if ex_id not in EXAMPLES:
                return self._send_json({"error": "unknown example"}, 404)
            score = EXAMPLES[ex_id][0]()
            return self._send_json(_score_payload(score))
        return self._send_json({"error": "not found"}, 404)

    def query_get(self, key: str, default: str) -> str:
        if "?" not in self.path:
            return default
        for kv in self.path.split("?", 1)[1].split("&"):
            if kv.startswith(key + "="):
                return kv.split("=", 1)[1]
        return default

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/orchestrate":
            return self._send_json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return self._send_json({"error": "invalid JSON"}, 400)
        if data.get("example_id"):
            score = EXAMPLES[data["example_id"]][0]()
        else:
            score = _score_from_request(data)
        if data.get("preset") and not data.get("ensemble"):
            ensemble = ENSEMBLE_PRESETS[data["preset"]]
        else:
            ensemble = _ensemble_from_request(data)
        if not ensemble:
            return self._send_json(
                {"feasible": False,
                 "explanation": "未指定任何演奏者/乐器，无法配器。"}, 400)
        session = uuid.uuid4().hex[:12]
        out_dir = os.path.join(_SESSION_ROOT, session)
        payload = orchestrate(score, ensemble, out_dir)
        payload["session"] = session
        payload["media"] = _media_map(payload.get("exports", {}), session)
        return self._send_json(payload)

    def _serve_static(self, path: str):
        if path in ("/", ""):
            path = "/index.html"
        target = os.path.normpath(os.path.join(_STATIC_DIR, path.lstrip("/")))
        if not target.startswith(_STATIC_DIR) or not os.path.isfile(target):
            self.send_error(404)
            return
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if target.endswith(".musicxml"):
            ctype = "application/vnd.recordare.musicxml+xml"
        with open(target, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_session_media(self, path: str):
        rel = path[len("/media/"):]
        target = os.path.normpath(os.path.join(_SESSION_ROOT, rel))
        if not target.startswith(_SESSION_ROOT) or not os.path.isfile(target):
            self.send_error(404)
            return
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if target.endswith(".musicxml"):
            ctype = "application/vnd.recordare.musicxml+xml"
        with open(target, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _media_map(exports: dict, session: str) -> dict:
    if not exports:
        return {}
    media = {
        "wav": f"/media/{session}/ensemble.wav",
        "midi": f"/media/{session}/ensemble.mid",
        "musicxml": f"/media/{session}/parts.musicxml",
        "player_wavs": {
            pid: f"/media/{session}/player_wavs/{os.path.basename(p)}"
            for pid, p in exports.get("player_wavs", {}).items()
        },
    }
    return media


def _score_payload(score: Score) -> dict:
    return {
        "title": score.title, "bpm": score.bpm,
        "beats_per_bar": score.beats_per_bar,
        "layers": [{
            "id": l.id, "role": l.role, "name": l.name,
            "register_lock": list(l.register_lock) if l.register_lock else None,
            "notes": [{"start": n.start, "duration": n.duration,
                       "pitch": n.pitch} for n in l.notes],
        } for l in score.layers],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="室内配器编曲试听界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    os.makedirs(_SESSION_ROOT, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"室内配器编曲试听界面已启动: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
