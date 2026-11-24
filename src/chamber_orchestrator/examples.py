"""测试谱例：长音移交、快速交替、音域冲突、锁定音区与兼任换奏。"""
from __future__ import annotations

from .score import (ROLE_BASS, ROLE_INNER, ROLE_MELODY, Score, make_line)


def example_long_tone() -> Score:
    """长音：16 拍持续音（@60bpm = 16s），超过任何管乐单口气，必须移交或拆分。"""
    melody = make_line("mel", ROLE_MELODY, "长音旋律", 60, [
        (0.0, 16.0, "G4"),
        (16.0, 4.0, "A4"),
    ])
    inner = make_line("inner", ROLE_INNER, "长音内声", 60, [
        (0.0, 8.0, "E4"), (8.0, 8.0, "F4"), (16.0, 4.0, "D4"),
    ])
    bass = make_line("bass", ROLE_BASS, "长音低音", 60, [
        (0.0, 10.0, "C3"), (10.0, 10.0, "G2"),
    ])
    return Score("谱例一：长音与呼吸移交", 60, [melody, inner, bass])


def example_fast_alternation() -> Score:
    """快速交替：144bpm 十六分音符吐音，起奏间隔 0.104s，超过管乐吐音下限。"""
    bpm = 144
    seq = ["C5", "D5", "E5", "D5", "F5", "E5", "D5", "C5",
           "E5", "F5", "G5", "F5", "E5", "D5", "C5", "D5"]
    entries = [(i * 0.25, 0.25, p) for i, p in enumerate(seq)]
    melody = make_line("mel", ROLE_MELODY, "快速交替旋律", bpm, entries)
    inner_entries = [(i * 0.5, 0.5, p) for i, p in enumerate(
        ["G4", "A4", "G4", "F4", "E4", "F4", "G4", "A4"])]
    inner = make_line("inner", ROLE_INNER, "内声脉动", bpm, inner_entries)
    bass_entries = [(0, 2.0, "C3"), (2, 2.0, "G2"), (4, 2.0, "A2"),
                    (6, 2.0, "F2")]
    bass = make_line("bass", ROLE_BASS, "低音支点", bpm, bass_entries)
    return Score("谱例二：快速交替与吐音极限", bpm, [melody, inner, bass])


def example_range_conflict() -> Score:
    """音域冲突：低音线到达 E1，小编制（最低 F3 附近）无乐器可覆盖 -> 无解。"""
    bpm = 90
    melody = make_line("mel", ROLE_MELODY, "旋律", bpm, [
        (0, 2, "E5"), (2, 2, "G5"), (4, 2, "A5"), (6, 2, "F5"),
    ])
    inner = make_line("inner", ROLE_INNER, "内声", bpm, [
        (0, 2, "C4"), (2, 2, "D4"), (4, 2, "E4"), (6, 2, "C4"),
    ])
    bass = make_line("bass", ROLE_BASS, "极低音线", bpm, [
        (0, 2, "E1"), (2, 2, "G1"), (4, 2, "A1"), (6, 2, "C2"),
    ])
    return Score("谱例三：音域冲突", bpm, [melody, inner, bass])


def example_locked_register() -> Score:
    """锁定音区：旋律锁定在 E5~G5 之间，禁止整体八度移位消解冲突。"""
    bpm = 96
    entries = [(0, 1.5, "G5"), (1.5, 0.5, "F5"), (2, 1.0, "E5"),
               (3, 1.0, "G5"), (4, 2.0, "A5"), (6, 2.0, "E5")]
    melody = make_line("mel", ROLE_MELODY, "锁定音区旋律", bpm, entries,
                       register_lock=("E5", "A5"))
    inner = make_line("inner", ROLE_INNER, "内声", bpm, [
        (0, 2, "C5"), (2, 2, "D5"), (4, 2, "B4"), (6, 2, "C5"),
    ])
    bass = make_line("bass", ROLE_BASS, "低音", bpm, [
        (0, 4, "C3"), (4, 4, "G2"),
    ])
    return Score("谱例四：用户锁定音区", bpm, [melody, inner, bass])


def example_doubling_conflict() -> Score:
    """兼任换奏：同一演奏者在长笛/短笛间切换，间隔不足换奏时间。

    前段 G4 短笛吹不了（只能长笛），紧接 C8 长笛吹不了（只能短笛），
    中间 0.25 拍（0.125s）空隙远小于换奏时间 2.5s。
    """
    bpm = 120
    melody = make_line("mel", ROLE_MELODY, "兼任冲突旋律", bpm, [
        (0, 2, "G4"), (2, 2, "A4"), (4.25, 1.5, "C8"), (6, 2, "B7"),
    ])
    inner = make_line("inner", ROLE_INNER, "内声", bpm, [
        (0, 2, "E4"), (2, 2, "F4"), (4.25, 2, "G4"), (6.5, 1.5, "A4"),
    ])
    bass = make_line("bass", ROLE_BASS, "低音", bpm, [
        (0, 4, "C3"), (4, 4, "D3"),
    ])
    return Score("谱例五：兼任换奏争抢", bpm, [melody, inner, bass])


def _capability_two_flutes_long() -> Score:
    """两条 16s 同时持续音（@60bpm）；仅两把长笛时合奏无解。"""
    bpm = 60
    return Score("边界：两条同时长音（两把长笛）", bpm, [
        make_line("mel", ROLE_MELODY, "长音旋律", bpm, [(0, 16, "G4")]),
        make_line("inner", ROLE_INNER, "长音内声", bpm, [(0, 16, "A4")]),
    ])


def _capability_four_flutes_long() -> Score:
    """同上材料；四把长笛可在 t=10s 无缝交接，构成可行重配回放。"""
    return _capability_two_flutes_long()


EXAMPLES = {
    "long_tone": (example_long_tone, "长音与呼吸移交"),
    "fast_alternation": (example_fast_alternation, "快速交替与吐音极限"),
    "range_conflict": (example_range_conflict, "音域冲突（无解演示）"),
    "locked_register": (example_locked_register, "用户锁定音区"),
    "doubling_conflict": (example_doubling_conflict, "兼任换奏争抢"),
    # 演奏能力边界谱例：单人可演奏而合奏无解，及其可行重配回放
    "capability_two_flutes_long": (
        _capability_two_flutes_long,
        "边界：两条同时长音 vs 两把长笛（合奏无解、单声部可演奏）"),
    "capability_four_flutes_long": (
        _capability_four_flutes_long,
        "边界：两条同时长音 vs 四把长笛（t=10s 交接可解）"),
}


# 边界谱例默认使用的编制（供界面“载入谱例”时自动选中）
EXAMPLE_ENSEMBLE = {
    "capability_two_flutes_long": [
        ("p1", ["flute"]), ("p2", ["flute"])],
    "capability_four_flutes_long": [
        ("p1", ["flute"]), ("p2", ["flute"]),
        ("p3", ["flute"]), ("p4", ["flute"])],
}
