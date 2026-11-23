"""标准 MIDI 文件导出（Type 1，纯标准库手写字节）。

MIDI 一律写实际发声音高（sounding pitch），便于 DAW/试听直接发声；
移调记谱差异在 MusicXML 分谱中体现。
"""
from __future__ import annotations

import struct

from .instruments import CATALOG
from .solver import Assignment


def _vlq(value: int) -> bytes:
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)
        value >>= 7
    out = bytearray()
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def _meta(delta: int, meta_type: int, payload: bytes) -> bytes:
    return _vlq(delta) + bytes([0xFF, meta_type]) + _vlq(len(payload)) + payload


def _track(events: list[tuple[int, bytes]]) -> bytes:
    body = b"".join(_vlq(delta) + data for delta, data in events)
    body += _vlq(0) + bytes([0xFF, 0x2F, 0x00])
    return b"MTrk" + struct.pack(">I", len(body)) + body


def build_midi(assignments: list[Assignment], bpm: float,
               ticks_per_beat: int = 480) -> bytes:
    tempo_us = int(60_000_000 / bpm)
    header = b"MThd" + struct.pack(">IHHH", 6, 1,
                                   _track_count(assignments) + 1,
                                   ticks_per_beat)
    conductor = _track([
        (0, bytes([0xFF, 0x58, 0x04, 4, 2, 24, 8])),
        (0, bytes([0xFF, 0x51, 0x03]) + tempo_us.to_bytes(3, "big")),
    ])
    tracks = [conductor]
    by_inst: dict[str, list[Assignment]] = {}
    for a in assignments:
        if a.fragment.kind != "tone":
            continue
        by_inst.setdefault(a.instrument_id, []).append(a)
    for inst_id, items in sorted(by_inst.items()):
        inst = CATALOG[inst_id]
        events: list[tuple[int, bytes]] = [
            (0, bytes([0xC0 & 0xFF, inst.midi_program & 0x7F])),
        ]
        # (tick, order, raw) 收集后按时间排序
        raw_events: list[tuple[int, int, bytes]] = []
        for a in items:
            f = a.fragment
            start_tick = int(round(f.start_s * bpm / 60.0 * ticks_per_beat))
            end_tick = int(round(f.end_s * bpm / 60.0 * ticks_per_beat))
            note = f.midi & 0x7F
            raw_events.append((start_tick, 0, bytes([0x90, note, 96])))
            raw_events.append((max(start_tick + 1, end_tick), 1,
                               bytes([0x80, note, 0])))
        raw_events.sort(key=lambda e: (e[0], e[1]))
        last = 0
        for tick, _order, data in raw_events:
            events.append((max(0, tick - last), data))
            last = tick
        tracks.append(_track(events))
    return header + b"".join(tracks)


def _track_count(assignments: list[Assignment]) -> int:
    return len({a.instrument_id for a in assignments
                if a.fragment.kind == "tone"})


def write_midi(path: str, assignments: list[Assignment], bpm: float) -> None:
    with open(path, "wb") as fh:
        fh.write(build_midi(assignments, bpm))
