"""可听合奏 WAV 合成（纯标准库，16-bit PCM）。

模型简化说明：这里用加法合成 + ADSR 包络近似各类音色，不是真实采样乐器；
目的是让配器结果可听、可审，音高/节奏/移调后的实际发声与求解结果一致。
"""
from __future__ import annotations

import math
import struct
import wave

from .instruments import CATALOG
from .solver import Assignment

SAMPLE_RATE = 44_100

# 各家族的谐波结构（相对振幅）与颤音参数
_TIMBRE = {
    "woodwind": {"partials": [1.0, 0.35, 0.18, 0.08],
                 "vibrato_rate": 5.5, "vibrato_depth": 0.006,
                 "attack": 0.06, "release": 0.12},
    "brass": {"partials": [1.0, 0.6, 0.4, 0.25, 0.12],
              "vibrato_rate": 5.0, "vibrato_depth": 0.004,
              "attack": 0.08, "release": 0.15},
    "string": {"partials": [1.0, 0.5, 0.28, 0.14, 0.07],
               "vibrato_rate": 6.0, "vibrato_depth": 0.008,
               "attack": 0.05, "release": 0.18},
    "keyboard": {"partials": [1.0, 0.5, 0.3, 0.15, 0.08],
                 "vibrato_rate": 0.0, "vibrato_depth": 0.0,
                 "attack": 0.008, "release": 0.35},
}
_FAMILY_GAIN = {"woodwind": 0.5, "brass": 0.55,
                "string": 0.5, "keyboard": 0.45}


def _adsr(t: float, duration: float, attack: float,
          release: float) -> float:
    if t < attack:
        return t / max(attack, 1e-6)
    if t > duration - release:
        return max(0.0, (duration - t) / max(release, 1e-6))
    return 1.0


def _tone_samples(freq: float, duration: float, family: str,
                  gain: float) -> list[float]:
    spec = _TIMBRE[family]
    n = int(duration * SAMPLE_RATE)
    out = [0.0] * n
    for i in range(n):
        t = i / SAMPLE_RATE
        vib = 1.0
        if spec["vibrato_rate"]:
            vib = 1.0 + spec["vibrato_depth"] * math.sin(
                2 * math.pi * spec["vibrato_rate"] * t)
        phase0 = 2 * math.pi * freq * t
        phase = 2 * math.pi * freq * vib * t
        sample = 0.0
        for h, amp in enumerate(spec["partials"], start=1):
            sample += amp * math.sin(phase * h)
        sample *= _adsr(t, duration, spec["attack"], spec["release"])
        out[i] = sample * gain
    return out


def render_tracks(assignments: list[Assignment]) -> dict[str, list[float]]:
    """按演奏者渲染分轨（含其全部乐器），返回 {player_id: samples}。"""
    total_s = max((a.fragment.end_s for a in assignments
                   if a.fragment.kind == "tone"), default=1.0)
    length = int((total_s + 0.5) * SAMPLE_RATE)
    tracks: dict[str, list[float]] = {}
    for a in assignments:
        f = a.fragment
        if f.kind != "tone":
            continue
        inst = CATALOG[a.instrument_id]
        freq = 440.0 * (2 ** ((f.midi - 69) / 12.0))
        voice = _tone_samples(freq, max(0.05, f.duration_s),
                              inst.family, _FAMILY_GAIN[inst.family])
        buf = tracks.setdefault(a.player_id, [0.0] * length)
        if len(buf) < length:
            buf.extend([0.0] * (length - len(buf)))
        start = int(f.start_s * SAMPLE_RATE)
        for i, v in enumerate(voice):
            if start + i < length:
                buf[start + i] += v
    return tracks


def mix(tracks: dict[str, list[float]]) -> list[float]:
    if not tracks:
        return [0.0]
    length = max(len(b) for b in tracks.values())
    mixed = [0.0] * length
    for buf in tracks.values():
        for i, v in enumerate(buf):
            mixed[i] += v
    peak = max((abs(v) for v in mixed), default=0.0)
    if peak > 0.95:
        scale = 0.95 / peak
        mixed = [v * scale for v in mixed]
    return mixed


def _write_wav(path: str, samples: list[float]) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        frames = bytearray()
        for v in samples:
            v = max(-1.0, min(1.0, v))
            frames += struct.pack("<h", int(v * 32767))
        wav.writeframes(bytes(frames))


def write_ensemble_wav(path: str,
                       assignments: list[Assignment]) -> list[float]:
    tracks = render_tracks(assignments)
    mixed = mix(tracks)
    _write_wav(path, mixed)
    return mixed


def write_player_wavs(out_dir: str,
                      assignments: list[Assignment]) -> dict[str, str]:
    import os
    tracks = render_tracks(assignments)
    paths = {}
    for pid, buf in sorted(tracks.items()):
        path = os.path.join(out_dir, f"part_{pid}.wav")
        _write_wav(path, buf)
        paths[pid] = path
    return paths
