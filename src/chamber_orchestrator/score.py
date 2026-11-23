"""音乐材料模型：主旋律、内声部、低音三层材料。"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .pitch import note_to_midi

ROLE_MELODY = "melody"
ROLE_INNER = "inner"
ROLE_BASS = "bass"
ROLES = (ROLE_MELODY, ROLE_INNER, ROLE_BASS)
ROLE_LABEL = {ROLE_MELODY: "主旋律", ROLE_INNER: "内声部", ROLE_BASS: "低音"}


@dataclass(frozen=True)
class Note:
    """单音。pitch 为音名（None 表示休止，用于显式换气占位）。时间单位为拍。"""
    start: float
    duration: float
    pitch: str | None = None
    layer_id: str = ""
    id: str = ""

    @property
    def midi(self) -> int | None:
        return note_to_midi(self.pitch) if self.pitch else None

    @property
    def end(self) -> float:
        return self.start + self.duration

    def is_rest(self) -> bool:
        return self.pitch is None


@dataclass
class MaterialLayer:
    """一条材料线（单声部旋律线；和弦由不同层或复音能力处理）。"""
    id: str
    role: str
    name: str
    notes: list[Note] = field(default_factory=list)
    # 用户锁定的音区（MIDI 闭区间）；求解器不得用整体八度移位逃出
    register_lock: tuple[int, int] | None = None

    def sounding_notes(self) -> list[Note]:
        return [n for n in self.notes if not n.is_rest()]


@dataclass
class Score:
    title: str
    bpm: float
    layers: list[MaterialLayer]
    beats_per_bar: int = 4

    @property
    def duration_beats(self) -> float:
        return max((n.end for layer in self.layers for n in layer.notes), default=0.0)

    @property
    def duration_s(self) -> float:
        return self.duration_beats * 60.0 / self.bpm

    def all_notes(self) -> list[Note]:
        out = []
        for layer in self.layers:
            for n in layer.notes:
                out.append(replace(n, layer_id=layer.id))
        return sorted(out, key=lambda n: (n.start, n.end, n.layer_id))


def make_note(start: float, duration: float, pitch: str | None,
              layer_id: str = "", idx: int = 0) -> Note:
    return Note(start, duration, pitch, layer_id, f"{layer_id}.n{idx}")


def make_line(layer_id: str, role: str, name: str, bpm: float,
              entries: list[tuple[float, float, str | None]],
              register_lock: tuple[str, str] | None = None) -> MaterialLayer:
    """entries: (start_beat, duration_beats, pitch or None) 列表。"""
    notes = [
        Note(s, d, p, layer_id, f"{layer_id}.n{i}")
        for i, (s, d, p) in enumerate(entries)
    ]
    lock = (note_to_midi(register_lock[0]), note_to_midi(register_lock[1])) \
        if register_lock else None
    return MaterialLayer(layer_id, role, name, notes, lock)
