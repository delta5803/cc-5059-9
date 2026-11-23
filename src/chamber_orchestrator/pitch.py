"""音高与拍点工具：MIDI 音高、音名换算、音程、拍-秒换算。"""
from __future__ import annotations

# 自然音名 -> 音级
_STEP = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
_STEP_NAME = ["C", "D", "E", "F", "G", "A", "B"]
_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_to_midi(name: str) -> int:
    """'C#4' -> 61；支持升降号与重升降（x/bb）。中央 C = C4 = MIDI 60。"""
    s = name.strip()
    step = _STEP[s[0]]
    i = 1
    alter = 0
    while i < len(s) and s[i] in "#b":
        alter += 1 if s[i] == "#" else -1
        i += 1
    if i < len(s) and s[i] == "x":
        alter += 2
        i += 1
    octave = int(s[i:])
    return (octave + 1) * 12 + step + alter


def midi_to_note(midi: int) -> str:
    octave = midi // 12 - 1
    return f"{_SHARP_NAMES[midi % 12]}{octave}"


def interval_semitones(low: int, high: int) -> int:
    return high - low


def beats_to_seconds(beats: float, bpm: float) -> float:
    return beats * 60.0 / bpm


def seconds_to_beats(seconds: float, bpm: float) -> float:
    return seconds * bpm / 60.0


def split_in_chromatic_steps(low: str, high: str):
    """返回闭区间内每个半音的 MIDI 号。"""
    a, b = note_to_midi(low), note_to_midi(high)
    return range(a, b + 1)
