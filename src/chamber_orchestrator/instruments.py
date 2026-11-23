"""乐器模型目录。

硬性演奏参数（按真实声学近似建模，数值为可执行的工程近似值）：
- low/high            实际发声音域（MIDI 号）
- written_offset      记谱移调：written_midi = sounding_midi + written_offset
                      （降 B 管 +2、F 圆号 +7、短笛 -12、倍大提琴 +12、C 调乐器 0）
- polyphony           最大同时发声音数（管乐 1，弦乐双音 2，钢琴 6）
- max_sustain_s       一口气单音最大持续（仅管乐）
- phrase_budget_s     一口气累计发声预算（仅管乐）
- breath_gap_s        换气所需休止时长
- min_attack_gap_s    两次吐音/起音的最小起奏间隔
- recovery_s          用尽气息或长音后的恢复休止
- max_double_stop     双音最大音程（半音，弦乐）
- shift_leap/shift_gap_s 换把大跳阈值与额外起奏间隔（弦乐）
- change_time_s       换奏该乐器所需时间（兼任模型）
- timbre              各编配角色的音色偏好分（0~1，软偏好）
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .pitch import note_to_midi


@dataclass(frozen=True)
class Instrument:
    id: str
    name: str
    family: str  # woodwind / brass / string / keyboard
    low: str
    high: str
    written_offset: int = 0
    polyphony: int = 1
    max_sustain_s: float | None = None
    phrase_budget_s: float | None = None
    breath_gap_s: float = 0.0
    min_attack_gap_s: float = 0.0
    recovery_s: float = 0.0
    max_double_stop: int | None = None
    shift_leap_semitones: int | None = None
    shift_gap_s: float = 0.0
    change_time_s: float = 3.0
    clef: str = "G"
    midi_program: int = 0
    timbre: dict[str, float] = field(default_factory=dict)

    @property
    def low_midi(self) -> int:
        return note_to_midi(self.low)

    @property
    def high_midi(self) -> int:
        return note_to_midi(self.high)

    @property
    def is_wind(self) -> bool:
        return self.family in ("woodwind", "brass")

    def written_midi(self, sounding_midi: int) -> int:
        return sounding_midi + self.written_offset

    def timbre_for(self, role: str) -> float:
        return self.timbre.get(role, 0.5)


_INSTRUMENTS: list[Instrument] = [
    Instrument("flute", "长笛", "woodwind", "C4", "D7",
               max_sustain_s=10.0, phrase_budget_s=12.0, breath_gap_s=0.6,
               min_attack_gap_s=0.12, recovery_s=1.2, change_time_s=2.5,
               midi_program=73,
               timbre={"melody": 0.90, "inner": 0.55, "bass": 0.20}),
    Instrument("piccolo", "短笛", "woodwind", "D5", "C8", written_offset=-12,
               max_sustain_s=8.0, phrase_budget_s=10.0, breath_gap_s=0.5,
               min_attack_gap_s=0.12, recovery_s=1.0, change_time_s=2.5,
               midi_program=72,
               timbre={"melody": 0.75, "inner": 0.35, "bass": 0.05}),
    Instrument("oboe", "双簧管", "woodwind", "Bb3", "G6",
               max_sustain_s=9.0, phrase_budget_s=10.0, breath_gap_s=0.8,
               min_attack_gap_s=0.14, recovery_s=1.4, change_time_s=3.0,
               midi_program=68,
               timbre={"melody": 0.85, "inner": 0.60, "bass": 0.25}),
    Instrument("clarinet_bb", "降B单簧管", "woodwind", "D3", "Bb6",
               written_offset=2,
               max_sustain_s=12.0, phrase_budget_s=14.0, breath_gap_s=0.7,
               min_attack_gap_s=0.12, recovery_s=1.2, change_time_s=3.0,
               midi_program=71,
               timbre={"melody": 0.80, "inner": 0.75, "bass": 0.40}),
    Instrument("bassoon", "大管", "woodwind", "Bb1", "Eb5",
               max_sustain_s=9.0, phrase_budget_s=11.0, breath_gap_s=0.9,
               min_attack_gap_s=0.16, recovery_s=1.5, change_time_s=3.0, clef="F",
               midi_program=70,
               timbre={"melody": 0.50, "inner": 0.75, "bass": 0.85}),
    Instrument("horn_f", "F圆号", "brass", "B1", "F5", written_offset=7,
               max_sustain_s=10.0, phrase_budget_s=12.0, breath_gap_s=1.0,
               min_attack_gap_s=0.18, recovery_s=1.8, change_time_s=3.5,
               midi_program=60,
               timbre={"melody": 0.60, "inner": 0.80, "bass": 0.70}),
    Instrument("trumpet_bb", "降B小号", "brass", "E3", "D6", written_offset=2,
               max_sustain_s=7.0, phrase_budget_s=8.0, breath_gap_s=0.7,
               min_attack_gap_s=0.15, recovery_s=1.4, change_time_s=3.0,
               midi_program=56,
               timbre={"melody": 0.80, "inner": 0.55, "bass": 0.35}),
    Instrument("trombone", "长号", "brass", "E2", "F5",
               max_sustain_s=9.0, phrase_budget_s=10.0, breath_gap_s=0.9,
               min_attack_gap_s=0.18, recovery_s=1.6, change_time_s=3.5, clef="F",
               midi_program=57,
               timbre={"melody": 0.45, "inner": 0.70, "bass": 0.85}),
    Instrument("violin", "小提琴", "string", "G3", "A7",
               polyphony=2, max_double_stop=12,
               shift_leap_semitones=14, shift_gap_s=0.18, change_time_s=2.0,
               midi_program=40,
               timbre={"melody": 0.90, "inner": 0.70, "bass": 0.25}),
    Instrument("viola", "中提琴", "string", "C3", "E6",
               polyphony=2, max_double_stop=12,
               shift_leap_semitones=14, shift_gap_s=0.20, change_time_s=2.0, clef="C",
               midi_program=41,
               timbre={"melody": 0.55, "inner": 0.85, "bass": 0.45}),
    Instrument("cello", "大提琴", "string", "C2", "C6",
               polyphony=2, max_double_stop=14,
               shift_leap_semitones=12, shift_gap_s=0.22, change_time_s=2.0, clef="F",
               midi_program=42,
               timbre={"melody": 0.65, "inner": 0.75, "bass": 0.90}),
    Instrument("double_bass", "倍大提琴", "string", "E1", "G4",
               written_offset=12, polyphony=2, max_double_stop=12,
               shift_leap_semitones=10, shift_gap_s=0.25, change_time_s=2.5,
               clef="F", midi_program=43,
               timbre={"melody": 0.30, "inner": 0.50, "bass": 0.95}),
    Instrument("piano", "钢琴", "keyboard", "A0", "C8",
               polyphony=6, change_time_s=4.0, clef="G",
               midi_program=0,
               timbre={"melody": 0.55, "inner": 0.80, "bass": 0.70}),
]

CATALOG: dict[str, Instrument] = {i.id: i for i in _INSTRUMENTS}

# 编制预设：每个元素是 (演奏者id, 可兼任乐器id 列表)
ENSEMBLE_PRESETS: dict[str, list[tuple[str, list[str]]]] = {
    # 五重管乐 + 弦乐四重奏 + 钢琴 的室内乐团
    "full": [
        ("p1", ["flute", "piccolo"]),
        ("p2", ["oboe"]),
        ("p3", ["clarinet_bb"]),
        ("p4", ["bassoon"]),
        ("p5", ["horn_f"]),
        ("p6", ["trumpet_bb"]),
        ("p7", ["trombone"]),
        ("p8", ["violin"]),
        ("p9", ["viola"]),
        ("p10", ["cello"]),
        ("p11", ["double_bass"]),
        ("p12", ["piano"]),
    ],
    # 小编制：无低音扩展乐器，用于制造音域冲突无解场景
    "small": [
        ("p1", ["flute"]),
        ("p2", ["clarinet_bb"]),
        ("p3", ["violin"]),
        ("p4", ["viola"]),
        ("p5", ["horn_f"]),
    ],
    # 弦乐四重奏
    "strings": [
        ("p1", ["violin"]),
        ("p2", ["violin"]),
        ("p3", ["viola"]),
        ("p4", ["cello"]),
    ],
    # 管乐五重奏（含长笛/短笛兼任）：强制在管乐呼吸模型内解决
    "winds": [
        ("p1", ["flute", "piccolo"]),
        ("p2", ["oboe"]),
        ("p3", ["clarinet_bb"]),
        ("p4", ["bassoon"]),
        ("p5", ["horn_f"]),
    ],
}
