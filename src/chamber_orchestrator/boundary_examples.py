"""演奏能力边界谱例：可知可行 / 可知不可行，附独立推导的期望值。

这些谱例的期望**不是**从求解器输出抄来的，而是根据 `instruments.py` 的文档化
硬约束手工推导，再写进常量（`*_EXPECTED`）。测试用它们验证：
- 求解器在边界上的判定正确（可行/不可行）；
- 独立验收器给出的能力违例与声部覆盖与这些手推期望一致；
- “单人可演奏、合奏无解”的反例确实存在，且能被回放到可行重配。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .score import (ROLE_BASS, ROLE_INNER, ROLE_MELODY, Score, make_line)

# 长笛（flute）的文档化硬约束（instruments.py）：
#   max_sustain_s=10.0, phrase_budget_s=12.0, breath_gap_s=0.6,
#   recovery_s=1.2, min_attack_gap_s=0.12, polyphony=1
FLUTE_MAX_SUSTAIN = 10.0
FLUTE_PHRASE_BUDGET = 12.0
FLUTE_BREATH_GAP = 0.6
FLUTE_RECOVERY = 1.2


@dataclass
class BoundaryCase:
    key: str
    title: str
    score_factory: callable
    ensemble: list[tuple[str, list[str]]]
    expect_feasible: bool
    # 手推的单人可演奏性：每个声部单独交给（足够的）编制时是否可演奏
    per_voice_playable: dict[str, bool] = field(default_factory=dict)
    # 手推的能力违例代码（当无解且被强制覆盖时验收器应报出的代码）
    expected_violation_codes: list[str] = field(default_factory=list)
    note: str = ""


def _two_voice_long():
    """两条 16s 同时持续音（@60bpm）。"""
    bpm = 60
    return Score("边界：两条同时长音", bpm, [
        make_line("mel", ROLE_MELODY, "长音旋律", bpm, [(0, 16, "G4")]),
        make_line("inner", ROLE_INNER, "长音内声", bpm, [(0, 16, "A4")]),
    ])


def _single_voice_long(pitch: str, layer_id: str, role: str, dur=16):
    def factory():
        bpm = 60
        return Score(f"边界：单声部长音 {layer_id}", bpm,
                     [make_line(layer_id, role, layer_id, bpm,
                                [(0, dur, pitch)])])
    return factory


def _low_note_score():
    """一条 E1 极低音（仅倍大提琴可覆盖）。"""
    bpm = 90
    return Score("边界：低音覆盖", bpm, [
        make_line("bass", ROLE_BASS, "极低音", bpm,
                  [(0, 4, "E1"), (4, 4, "G1")]),
    ])


# --------------------------------------------------------------------------- #
# 可知不可行：两条同时 16s 长音 + 两把长笛
# --------------------------------------------------------------------------- #
# 推导：两条声部在 [0,16] 全程同时发声；唯一两件乐器都是单声部长笛，故两把长笛
# 在 [0,16] 全程都被占用、无法换气。单音 16s > max_sustain 10s 必然被拒。
# ⇒ 合奏无解。但把任一声部单独交给两把长笛（10s + 6s 移交）或一把钢琴都可演奏，
# 故“单人（单声部）可演奏、合奏无解”。
TWO_FLUTES_SIMULTANEOUS_LONG = BoundaryCase(
    key="capability_two_flutes_long",
    title="两条同时长音 vs 两把长笛（合奏无解、单声部可演奏）",
    score_factory=_two_voice_long,
    ensemble=[("p1", ["flute"]), ("p2", ["flute"])],
    expect_feasible=False,
    per_voice_playable={"mel": True, "inner": True},
    expected_violation_codes=["sustain"],
    note="两条全程同响 + 单声部乐器 ⇒ 无换气缝隙，16s 单音超最大持续 10s。",
)


def _four_flute_ensemble():
    return [("p1", ["flute"]), ("p2", ["flute"]),
            ("p3", ["flute"]), ("p4", ["flute"])]


# 回放可行版本：四把长笛即可轮转（每条长音 10s + 6s 由两人交接），
# 用人工排定的可行分配验证验收器通过（证明“无解”是编制不足，而非记号不可演奏）。
FOUR_FLUTES_SIMULTANEOUS_LONG = BoundaryCase(
    key="capability_four_flutes_long",
    title="两条同时长音 vs 四把长笛（轮转可解，回放）",
    score_factory=_two_voice_long,
    ensemble=_four_flute_ensemble(),
    expect_feasible=True,
    per_voice_playable={"mel": True, "inner": True},
    note="每条长音由两把长笛在 t=10s 无缝交接；四把长笛互不争用。",
)


# --------------------------------------------------------------------------- #
# 可知可行 / 不可行：低音覆盖（缺少低音乐器）
# --------------------------------------------------------------------------- #
LOW_NOTE_WITH_BASS = BoundaryCase(
    key="coverage_low_with_bass",
    title="E1 低音 vs 倍大提琴（覆盖通过）",
    score_factory=_low_note_score,
    ensemble=[("p1", ["double_bass"])],
    expect_feasible=True,
    note="倍大提琴实际音域 E1..G4 覆盖 E1/G1，单人可演奏。",
)

LOW_NOTE_WITHOUT_BASS = BoundaryCase(
    key="coverage_low_without_bass",
    title="E1 低音 vs 无低音乐器（能力违例）",
    score_factory=_low_note_score,
    ensemble=[("p1", ["flute"]), ("p2", ["violin"])],
    expect_feasible=False,
    expected_violation_codes=["range"],
    note="长笛最低 C4、小提琴最低 G3，均无法覆盖 E1；求解器应无解。",
)


# 人为把低音强行分给长笛（忽略求解器），验收器必须报 range 违例 + 声部覆盖失败。
def forced_low_note_on_flute():
    from .solver import Assignment, Fragment
    score = _low_note_score()
    assignments = [
        Assignment("p1", "flute",
                   Fragment("bass.n0", "bass", "bass", 0.0,
                            beats(4, score.bpm), 26)),
        Assignment("p1", "flute",
                   Fragment("bass.n1", "bass", "bass", beats(4, score.bpm),
                            beats(4, score.bpm), 31)),
    ]
    return score, assignments, [("p1", ["flute"])]


def beats(beats_value: float, bpm: float) -> float:
    return beats_value * 60.0 / bpm


BOUNDARY_CASES = {
    c.key: c for c in (
        TWO_FLUTES_SIMULTANEOUS_LONG,
        FOUR_FLUTES_SIMULTANEOUS_LONG,
        LOW_NOTE_WITH_BASS,
        LOW_NOTE_WITHOUT_BASS,
    )
}


def build_handoff_assignments():
    """人工排定“两条长音、四把长笛、t=10s 交接”的可行回放。"""
    from .solver import Assignment, Fragment
    score = _two_voice_long()
    a = [
        Assignment("p1", "flute",
                   Fragment("mel.n0", "mel", "melody", 0.0, 10.0, 64, 0, 2)),
        Assignment("p2", "flute",
                   Fragment("inner.n0", "inner", "inner", 0.0, 10.0, 65,
                            0, 2)),
        Assignment("p3", "flute",
                   Fragment("mel.n0", "mel", "melody", 10.0, 6.0, 64, 1, 2,
                            handoff_from="p1")),
        Assignment("p4", "flute",
                   Fragment("inner.n0", "inner", "inner", 10.0, 6.0, 65, 1, 2,
                            handoff_from="p2")),
    ]
    return score, a, _four_flute_ensemble()
