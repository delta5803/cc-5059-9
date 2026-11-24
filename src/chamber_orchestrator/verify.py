"""独立演奏能力验收（independent performance-capability verification）。

本模块**不复用求解器内部判定**：它只依赖乐器的文档化硬约束（`instruments.py`）
与材料谱面（`score.py`），从零重新推导每一项演奏条件，用于对求解结果做交叉复核。

验收内容：
1. 实际发声音域（sounding MIDI 是否落在 instrument.low..high）
2. 移调记谱：记谱音高 = 实际音高 + written_offset（并核验导出的 MusicXML 分谱）
3. 呼吸限制：单音最大持续、乐句累计预算、换气休止、恢复时长
4. 换奏时间、最小起奏间隔、换把大跳、复音/双音上限
5. 休息区间：换气休止是否足够、是否与发声重叠、拆分前后段是否可衔接
6. 声部覆盖：每个材料的每个音必须被片段完整、无重叠、同音高地覆盖
7. 混音试听：导出 WAV 是否非静音、时长与峰值合理、分轨齐全

设计原则：
- **不伪报**：求解器的可行解必须能原样通过全部检查（单元测试强制）。
- **证据化**：每条违例都带 player / instrument / note / layer / 时刻，
  足以定位冲突声部与交接时刻。
- **枚举为真值**：`enumerate_feasible` 对小谱例穷举乐器组合得到权威可行域；
  `derive_rest_segments` 则从“指定休息”反推可行交接，两者在边界谱例上必须一致。
"""
from __future__ import annotations

import math
import os
import wave
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .instruments import CATALOG, Instrument
from .pitch import beats_to_seconds, note_to_midi

EPS = 1e-6

# 违例代码 -> 中文说明（CLI / 界面共用）
CODE_LABEL = {
    "range": "超出实际发声音域",
    "transposition": "移调记谱不一致",
    "sustain": "单音超过最大持续",
    "phrase_budget": "乐句累计气息超预算",
    "recovery": "恢复时长不足",
    "breath_rest": "换气休止不足或缺失",
    "change_time": "换奏时间不足",
    "attack_gap": "起奏间隔不足",
    "shift": "弦乐换把大跳间隔不足",
    "polyphony": "超过复音上限",
    "double_stop": "双音音程超上限",
    "not_configured": "使用了未配置的乐器",
    "rest_overlap": "换气休止与发声重叠",
    "coverage_gap": "声部未被完整覆盖",
    "coverage_overlap": "声部被重复覆盖",
    "coverage_pitch": "覆盖音高与材料不一致",
    "coverage_unknown": "出现材料中不存在的音符",
    "media_missing": "试听产物缺失",
    "media_silent": "混音静音",
    "media_duration": "混音时长异常",
    "media_clipping": "混音削波",
    "xml_pitch": "分谱记谱音高与移调不符",
    "xml_transpose": "分谱缺少移调声明",
}


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Violation:
    code: str
    kind: str  # capability / coverage / media
    detail: str
    player: str | None = None
    instrument: str | None = None
    note_id: str | None = None
    layer: str | None = None
    role: str | None = None
    time_s: float | None = None
    conflict_notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return CODE_LABEL.get(self.code, self.code)

    def to_dict(self) -> dict:
        return {
            "code": self.code, "kind": self.kind, "label": self.label,
            "detail": self.detail, "player": self.player,
            "instrument": self.instrument, "note_id": self.note_id,
            "layer": self.layer, "role": self.role,
            "time_s": None if self.time_s is None else round(self.time_s, 3),
            "conflict_notes": list(self.conflict_notes),
        }


@dataclass
class CoverageEntry:
    note_id: str
    layer: str
    role: str
    start_s: float
    end_s: float
    midi: int
    covered: bool
    gaps: list[tuple[float, float]] = field(default_factory=list)
    overlaps: list[tuple[float, float]] = field(default_factory=list)
    fragments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "note_id": self.note_id, "layer": self.layer, "role": self.role,
            "start_s": round(self.start_s, 3), "end_s": round(self.end_s, 3),
            "midi": self.midi, "covered": self.covered,
            "gaps": [[round(a, 3), round(b, 3)] for a, b in self.gaps],
            "overlaps": [[round(a, 3), round(b, 3)] for a, b in self.overlaps],
            "fragments": self.fragments,
        }


@dataclass
class VerifyReport:
    has_assignments: bool
    capability_violations: list[Violation] = field(default_factory=list)
    coverage_violations: list[Violation] = field(default_factory=list)
    media_violations: list[Violation] = field(default_factory=list)
    coverage: list[CoverageEntry] = field(default_factory=list)
    voices_expected: int = 0
    voices_covered: int = 0
    conflict_voices: list[dict] = field(default_factory=list)
    handoff_moments: list[dict] = field(default_factory=list)

    @property
    def all_violations(self) -> list[Violation]:
        return (self.capability_violations + self.coverage_violations
                + self.media_violations)

    @property
    def coverage_complete(self) -> bool:
        return self.voices_expected > 0 and \
            self.voices_covered == self.voices_expected

    @property
    def passes(self) -> bool:
        return (self.has_assignments and self.coverage_complete
                and not self.all_violations)

    def summary(self) -> str:
        if not self.has_assignments:
            return "无分配结果，无法验收。"
        state = "通过" if self.passes else "未通过"
        return (f"演奏能力验收{state}：能力违例 "
                f"{len(self.capability_violations)}，声部覆盖违例 "
                f"{len(self.coverage_violations)}，试听异常 "
                f"{len(self.media_violations)}；声部覆盖 "
                f"{self.voices_covered}/{self.voices_expected}。")

    def to_dict(self) -> dict:
        return {
            "has_assignments": self.has_assignments,
            "passes": self.passes,
            "status": "pass" if self.passes else (
                "no-assignments" if not self.has_assignments else "fail"),
            "summary": self.summary(),
            "coverage_complete": self.coverage_complete,
            "voices_expected": self.voices_expected,
            "voices_covered": self.voices_covered,
            "conflict_voices": list(self.conflict_voices),
            "handoff_moments": list(self.handoff_moments),
            "capability_violations": [v.to_dict()
                                      for v in self.capability_violations],
            "coverage_violations": [v.to_dict()
                                    for v in self.coverage_violations],
            "media_violations": [v.to_dict() for v in self.media_violations],
            "coverage": [c.to_dict() for c in self.coverage],
        }


# --------------------------------------------------------------------------- #
# 参考数据：直接由谱面与乐器表推导（不经过求解器）
# --------------------------------------------------------------------------- #
def reference_notes(score) -> list[dict]:
    """从材料谱面重新推导每个发声音符的时间区间与音高（秒 / MIDI）。"""
    layer_role = {l.id: l.role for l in score.layers}
    out = []
    for layer in score.layers:
        for n in layer.notes:
            if n.is_rest():
                continue
            out.append({
                "note_id": n.id,
                "layer": layer.id,
                "role": layer_role.get(layer.id, ""),
                "start_s": beats_to_seconds(n.start, score.bpm),
                "end_s": beats_to_seconds(n.end, score.bpm),
                "midi": note_to_midi(n.pitch),
            })
    return out


def _inst(inst_id: str) -> Instrument | None:
    return CATALOG.get(inst_id)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def verify_assignments(score, assignments, ensemble,
                       export_dir: str | None = None) -> VerifyReport:
    """对一组具体分配做独立演奏能力验收。

    ensemble: [(player_id, [instrument_id, ...]), ...]
    export_dir: 若给出，额外核验该目录下的混音/分谱产物。
    """
    inst_map = dict(CATALOG)
    configured: dict[str, set[str]] = {
        pid: set(ids) for pid, ids in ensemble}
    report = VerifyReport(has_assignments=bool(assignments))

    if not assignments:
        report.coverage = _build_coverage(reference_notes(score), [], report)
        return report

    tone_frags = [a for a in assignments
                  if getattr(a.fragment, "kind", "tone") == "tone"]
    by_player: dict[str, list] = {}
    for a in assignments:
        by_player.setdefault(a.player_id, []).append(a)

    for pid, frags in sorted(by_player.items()):
        _verify_player(pid, frags, configured.get(pid, set()),
                       inst_map, report)

    report.coverage = _build_coverage(reference_notes(score),
                                      assignments, report)
    _aggregate_conflicts(report, assignments)

    if export_dir:
        report.media_violations.extend(
            verify_export(export_dir, score, assignments, ensemble))
    return report


def _aggregate_conflicts(report: VerifyReport, assignments) -> None:
    """把能力违例聚合成“冲突声部”与“交接时刻”两份索引，便于定位。"""
    voices: dict[str, dict] = {}
    for v in report.capability_violations:
        if not v.note_id:
            continue
        rec = voices.setdefault(v.note_id, {
            "note_id": v.note_id, "layer": v.layer, "role": v.role,
            "player": v.player, "instrument": v.instrument,
            "codes": [], "times": [],
        })
        if v.code not in rec["codes"]:
            rec["codes"].append(v.code)
        if v.time_s is not None:
            rec["times"].append(round(v.time_s, 3))
        if v.conflict_notes:
            rec.setdefault("conflicts_with", [])
            for n in v.conflict_notes:
                if n and n not in rec["conflicts_with"]:
                    rec["conflicts_with"].append(n)
    for rec in voices.values():
        rec["times"] = sorted(set(rec["times"]))
    report.conflict_voices = sorted(voices.values(),
                                    key=lambda r: r["note_id"])

    # 交接/拆分时刻：同一 note_id 出现多个发声片段时的片段边界
    counts: dict[str, int] = {}
    last_end: dict[str, float] = {}
    for a in sorted(assignments, key=lambda x: x.fragment.start_s):
        f = a.fragment
        if getattr(f, "kind", "tone") != "tone":
            continue
        counts[f.note_id] = counts.get(f.note_id, 0) + 1
        if f.note_id in last_end:
            report.handoff_moments.append({
                "note_id": f.note_id, "layer": f.layer_id, "role": f.role,
                "time_s": round(f.start_s, 3),
                "from_player": last_end[f.note_id],
                "to_player": a.player_id,
            })
        last_end[f.note_id] = a.player_id


def _verify_player(player_id: str, frags: list, allowed: set[str],
                   inst_map: dict, report: VerifyReport) -> None:
    ordered = sorted(frags, key=lambda a: (
        a.fragment.start_s,
        0 if getattr(a.fragment, "kind", "tone") == "tone" else 1,
        a.fragment.end_s))
    state = _PlayerState()
    active: list = []
    prev_tone = None
    for a in ordered:
        f = a.fragment
        inst = inst_map.get(a.instrument_id)
        if inst is None:
            report.capability_violations.append(Violation(
                "not_configured", "capability",
                f"未知乐器 {a.instrument_id}", player=player_id,
                instrument=a.instrument_id, note_id=f.note_id,
                layer=f.layer_id, time_s=f.start_s))
            continue
        if a.instrument_id not in allowed:
            report.capability_violations.append(Violation(
                "not_configured", "capability",
                f"{player_id} 未配置乐器 {inst.name}", player=player_id,
                instrument=inst.name, note_id=f.note_id,
                layer=f.layer_id, time_s=f.start_s))
            continue
        if getattr(f, "kind", "tone") != "tone":
            _check_rest(player_id, inst, f, allowed, ordered, report)
            state.on_rest(f)
            continue

        # 1) 实际发声音域 + 移调一致性
        if f.midi is None:
            report.capability_violations.append(Violation(
                "range", "capability", f"{f.note_id} 缺少实际音高",
                player=player_id, instrument=inst.name, note_id=f.note_id,
                layer=f.layer_id, role=f.role, time_s=f.start_s))
        else:
            if not (inst.low_midi <= f.midi <= inst.high_midi):
                report.capability_violations.append(Violation(
                    "range", "capability",
                    f"{inst.name} 实际发声音域 [{inst.low},{inst.high}] "
                    f"(MIDI {inst.low_midi}-{inst.high_midi})，"
                    f"该音 MIDI {f.midi} 超出",
                    player=player_id, instrument=inst.name,
                    note_id=f.note_id, layer=f.layer_id, role=f.role,
                    time_s=f.start_s))
            written = inst.written_midi(f.midi)
            if written - inst.written_offset != f.midi:
                report.capability_violations.append(Violation(
                    "transposition", "capability",
                    f"{inst.name} 移调偏移 {inst.written_offset} 反推不一致",
                    player=player_id, instrument=inst.name,
                    note_id=f.note_id, layer=f.layer_id, time_s=f.start_s))

        # 2) 换奏时间（与上一段发声/休止比较）
        if state.current_inst is not None and \
                state.current_inst != a.instrument_id:
            gap = f.start_s - state.last_end
            if gap < inst.change_time_s - EPS:
                report.capability_violations.append(Violation(
                    "change_time", "capability",
                    f"换奏 {inst.name} 需 {inst.change_time_s}s，"
                    f"实际间隔 {gap:.3f}s（{state.current_inst}→"
                    f"{a.instrument_id}）",
                    player=player_id, instrument=inst.name,
                    note_id=f.note_id, layer=f.layer_id, role=f.role,
                    time_s=f.start_s,
                    conflict_notes=[state.prev_note_id]
                    if state.prev_note_id else []))

        overlapping = [x for x in active if x.end_s > f.start_s + EPS]

        # 3) 起奏间隔 / 换把（仅对非同时发声的连续音；双音为一次起音）
        if prev_tone is not None and prev_tone[2].end_s <= f.start_s + EPS:
            gap = f.start_s - prev_tone[2].start_s
            if gap < inst.min_attack_gap_s - EPS:
                report.capability_violations.append(Violation(
                    "attack_gap", "capability",
                    f"{inst.name} 起奏间隔 {gap:.3f}s < 最小 "
                    f"{inst.min_attack_gap_s}s",
                    player=player_id, instrument=inst.name,
                    note_id=f.note_id, layer=f.layer_id, role=f.role,
                    time_s=f.start_s,
                    conflict_notes=[prev_tone[2].note_id]))
            if (inst.shift_leap_semitones is not None
                    and prev_tone[2].midi is not None
                    and f.midi is not None
                    and abs(f.midi - prev_tone[2].midi)
                    >= inst.shift_leap_semitones
                    and gap < inst.shift_gap_s - EPS):
                report.capability_violations.append(Violation(
                    "shift", "capability",
                    f"{inst.name} 换把大跳 "
                    f"{abs(f.midi - prev_tone[2].midi)} 半音，"
                    f"需 {inst.shift_gap_s}s 间隔，实际 {gap:.3f}s",
                    player=player_id, instrument=inst.name,
                    note_id=f.note_id, layer=f.layer_id, role=f.role,
                    time_s=f.start_s,
                    conflict_notes=[prev_tone[2].note_id]))

        # 4) 复音 / 双音
        if overlapping:
            other_insts = {x.inst_id for x in overlapping}
            if other_insts - {a.instrument_id}:
                report.capability_violations.append(Violation(
                    "polyphony", "capability",
                    f"{player_id} 同时持 {inst.name} 与其它乐器发声，"
                    f"单人不具备该复音能力",
                    player=player_id, instrument=inst.name,
                    note_id=f.note_id, layer=f.layer_id, role=f.role,
                    time_s=f.start_s,
                    conflict_notes=[x.note_id for x in overlapping]))
            elif len(overlapping) + 1 > inst.polyphony:
                report.capability_violations.append(Violation(
                    "polyphony", "capability",
                    f"{inst.name} 复音上限 {inst.polyphony}，"
                    f"同时发声 {len(overlapping) + 1} 音",
                    player=player_id, instrument=inst.name,
                    note_id=f.note_id, layer=f.layer_id, role=f.role,
                    time_s=f.start_s,
                    conflict_notes=[x.note_id for x in overlapping]))
            else:
                for x in overlapping:
                    if (inst.max_double_stop is not None
                            and f.midi is not None and x.midi is not None
                            and abs(x.midi - f.midi) > inst.max_double_stop):
                        report.capability_violations.append(Violation(
                            "double_stop", "capability",
                            f"{inst.name} 双音音程 "
                            f"{abs(x.midi - f.midi)} 半音 > "
                            f"{inst.max_double_stop}",
                            player=player_id, instrument=inst.name,
                            note_id=f.note_id, layer=f.layer_id,
                            role=f.role, time_s=f.start_s,
                            conflict_notes=[x.note_id]))

        # 5) 呼吸（仅管乐）
        if inst.is_wind and f.midi is not None:
            _check_wind(player_id, inst, f, state, report, overlapping)

        state.on_tone(a, inst)
        active = [x for x in active if x.end_s > f.start_s + EPS]
        active.append(_ActiveFrag(f.note_id, a.instrument_id, f.start_s,
                                  f.end_s, f.midi))
        prev_tone = (f.note_id, a.instrument_id, f)


@dataclass
class _ActiveFrag:
    note_id: str
    inst_id: str
    start_s: float
    end_s: float
    midi: int | None


class _PlayerState:
    """独立的演奏者时间线状态（从文档化模型重新实现，不引用求解器）。"""

    def __init__(self) -> None:
        self.current_inst: str | None = None
        self.last_end: float = -1e9
        self.last_start: float = -1e9
        self.last_midi: int | None = None
        self.phrase_used: float = 0.0
        self.exhausted_until: float = -1e9
        self.prev_note_id: str | None = None

    def on_tone(self, a, inst: Instrument) -> None:
        f = a.fragment
        if inst.is_wind:
            reset = (self.last_end < 0
                     or f.start_s - self.last_end >= inst.breath_gap_s - EPS)
            if reset:
                self.phrase_used = 0.0
            self.phrase_used += f.duration_s
            if (self.phrase_used >= inst.phrase_budget_s - EPS
                    or f.duration_s >= inst.max_sustain_s - EPS):
                self.exhausted_until = f.end_s + inst.recovery_s
        self.current_inst = a.instrument_id
        self.last_start = f.start_s
        self.last_end = f.end_s
        self.last_midi = f.midi
        self.prev_note_id = f.note_id

    def on_rest(self, f) -> None:
        self.phrase_used = 0.0
        self.last_end = max(self.last_end, f.end_s)


def _check_wind(player_id: str, inst: Instrument, f, state: _PlayerState,
                report: VerifyReport, overlapping: list) -> None:
    if inst.max_sustain_s is None:
        return
    if f.start_s < state.exhausted_until - EPS:
        report.capability_violations.append(Violation(
            "recovery", "capability",
            f"{inst.name} 仍在恢复中（至 {state.exhausted_until:.2f}s），"
            f"该音起点 {f.start_s:.2f}s",
            player=player_id, instrument=inst.name, note_id=f.note_id,
            layer=f.layer_id, role=f.role, time_s=f.start_s))
        return
    reset = (state.last_end < 0
             or f.start_s - state.last_end >= inst.breath_gap_s - EPS)
    used = 0.0 if reset else state.phrase_used
    remaining = inst.phrase_budget_s - used
    if f.duration_s > inst.max_sustain_s + EPS:
        report.capability_violations.append(Violation(
            "sustain", "capability",
            f"{inst.name} 单音 {f.duration_s:.2f}s > 最大持续 "
            f"{inst.max_sustain_s}s",
            player=player_id, instrument=inst.name, note_id=f.note_id,
            layer=f.layer_id, role=f.role, time_s=f.start_s))
    elif f.duration_s > remaining + EPS:
        report.capability_violations.append(Violation(
            "phrase_budget", "capability",
            f"{inst.name} 乐句剩余气息 {remaining:.2f}s 不足 "
            f"{f.duration_s:.2f}s（已用 {used:.2f}/"
            f"{inst.phrase_budget_s}s）",
            player=player_id, instrument=inst.name, note_id=f.note_id,
            layer=f.layer_id, role=f.role, time_s=f.start_s))


def _check_rest(player_id: str, inst: Instrument, f, allowed: set[str],
                ordered: list, report: VerifyReport) -> None:
    if f.duration_s < -EPS:
        report.capability_violations.append(Violation(
            "breath_rest", "capability", "换气休止时长为负",
            player=player_id, instrument=inst.name, note_id=f.note_id,
            layer=f.layer_id, time_s=f.start_s))
        return
    if inst.is_wind and f.duration_s + EPS < inst.breath_gap_s:
        report.capability_violations.append(Violation(
            "breath_rest", "capability",
            f"{inst.name} 换气休止 {f.duration_s:.2f}s < "
            f"所需 {inst.breath_gap_s}s",
            player=player_id, instrument=inst.name, note_id=f.note_id,
            layer=f.layer_id, time_s=f.start_s))
    for a in ordered:
        g = a.fragment
        if getattr(g, "kind", "tone") != "tone":
            continue
        if f.start_s < g.end_s - EPS and g.start_s < f.end_s - EPS:
            report.capability_violations.append(Violation(
                "rest_overlap", "capability",
                f"换气休止 [{f.start_s:.2f},{f.end_s:.2f}] 与发声 "
                f"{g.note_id} [{g.start_s:.2f},{g.end_s:.2f}] 重叠",
                player=player_id, instrument=inst.name, note_id=f.note_id,
                layer=f.layer_id, time_s=f.start_s,
                conflict_notes=[g.note_id]))


# --------------------------------------------------------------------------- #
# 声部覆盖（逐音、逐时间区间）
# --------------------------------------------------------------------------- #
def _build_coverage(ref_notes: list[dict], assignments,
                    report: VerifyReport) -> list[CoverageEntry]:
    by_note: dict[str, list] = {}
    ref_ids = {r["note_id"] for r in ref_notes}
    for a in assignments:
        f = a.fragment
        if getattr(f, "kind", "tone") != "tone":
            continue
        by_note.setdefault(f.note_id, []).append(a)
        if f.note_id not in ref_ids:
            report.coverage_violations.append(Violation(
                "coverage_unknown", "coverage",
                f"分配中出现材料谱面不存在的音符 {f.note_id}",
                player=a.player_id, instrument=a.instrument_id,
                note_id=f.note_id, layer=f.layer_id, time_s=f.start_s))

    entries: list[CoverageEntry] = []
    covered_voices = 0
    for r in ref_notes:
        frags = sorted(by_note.get(r["note_id"], []),
                       key=lambda a: a.fragment.start_s)
        entry = CoverageEntry(
            r["note_id"], r["layer"], r["role"],
            r["start_s"], r["end_s"], r["midi"], covered=False)
        # 只把“乐器音域真的能发出该音、且音高一致”的片段算作有效覆盖；
        # 超出音域的片段另有 range 能力违例，不能用来冒充声部覆盖。
        effective = []
        for a in frags:
            f = a.fragment
            inst = CATALOG.get(a.instrument_id)
            capable = (inst is not None and f.midi is not None
                       and inst.low_midi <= f.midi <= inst.high_midi)
            entry.fragments.append({
                "player": a.player_id, "instrument": a.instrument_id,
                "start_s": round(f.start_s, 3), "end_s": round(f.end_s, 3),
                "midi": f.midi, "range_capable": capable,
            })
            if f.midi != r["midi"]:
                report.coverage_violations.append(Violation(
                    "coverage_pitch", "coverage",
                    f"{r['note_id']} 覆盖音高 MIDI {f.midi} ≠ 材料 "
                    f"{r['midi']}（禁止整体八度移位或以错音充数）",
                    player=a.player_id, instrument=a.instrument_id,
                    note_id=r["note_id"], layer=r["layer"], role=r["role"],
                    time_s=f.start_s))
            if capable and f.midi == r["midi"]:
                effective.append((f.start_s, f.end_s, a))
        gaps, overlaps = _interval_cover(
            r["start_s"], r["end_s"], [(s, e) for s, e, _ in effective])
        entry.gaps, entry.overlaps = gaps, overlaps
        entry.covered = (not gaps and not overlaps and bool(effective))
        for a, b in gaps:
            report.coverage_violations.append(Violation(
                "coverage_gap", "coverage",
                f"{r['note_id']}（{r['role']}）在 "
                f"[{a:.3f},{b:.3f}]s 未被任何可发该音的乐器覆盖",
                note_id=r["note_id"], layer=r["layer"], role=r["role"],
                time_s=a))
        for a, b in overlaps:
            note_ids = [x.fragment.note_id for s, e, x in effective
                        if s < b - EPS and e > a + EPS]
            report.coverage_violations.append(Violation(
                "coverage_overlap", "coverage",
                f"{r['note_id']} 在 [{a:.3f},{b:.3f}]s 被重复覆盖",
                note_id=r["note_id"], layer=r["layer"], role=r["role"],
                time_s=a, conflict_notes=note_ids))
        if entry.covered:
            covered_voices += 1
        entries.append(entry)

    report.voices_expected = len(ref_notes)
    report.voices_covered = covered_voices
    return entries


def _interval_cover(start: float, end: float,
                    intervals: list[tuple[float, float]]):
    """返回 [start,end] 内未被覆盖的空隙与被重复覆盖的区间。"""
    if not intervals:
        return [(start, end)], []
    clipped = sorted((max(start, s), min(end, e)) for s, e in intervals
                     if e > start + EPS and s < end - EPS)
    gaps: list[tuple[float, float]] = []
    overlaps: list[tuple[float, float]] = []
    cursor = start
    for s, e in clipped:
        if s > cursor + EPS:
            gaps.append((cursor, s))
        elif s < cursor - EPS:
            overlaps.append((s, min(cursor, e)))
        cursor = max(cursor, e)
    if cursor < end - EPS:
        gaps.append((cursor, end))
    return gaps, overlaps


# --------------------------------------------------------------------------- #
# 混音 / 分谱产物核验（执行证据）
# --------------------------------------------------------------------------- #
def verify_export(export_dir: str, score, assignments,
                  ensemble) -> list[Violation]:
    out: list[Violation] = []
    wav_path = os.path.join(export_dir, "ensemble.wav")
    mid_path = os.path.join(export_dir, "ensemble.mid")
    xml_path = os.path.join(export_dir, "parts.musicxml")
    for path, name in ((wav_path, "ensemble.wav"), (mid_path,
                                                     "ensemble.mid"),
                       (xml_path, "parts.musicxml")):
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            out.append(Violation("media_missing", "media",
                                 f"缺少或为空：{name}"))
    if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
        out.extend(_verify_wav(wav_path, score))
    if os.path.isfile(xml_path) and os.path.getsize(xml_path) > 0:
        out.extend(_verify_musicxml(xml_path, assignments))
    # 各演奏者分轨
    pdir = os.path.join(export_dir, "player_wavs")
    players_with_tones = sorted({a.player_id for a in assignments
                                 if getattr(a.fragment, "kind", "tone")
                                 == "tone"})
    for pid in players_with_tones:
        path = os.path.join(pdir, f"part_{pid}.wav")
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            out.append(Violation("media_missing", "media",
                                 f"缺少演奏者分轨 {pid}.wav", player=pid))
            continue
        out.extend(_verify_wav(path, score, player=pid))
    return out


def _verify_wav(path: str, score, player: str | None = None) -> list[Violation]:
    out: list[Violation] = []
    who = f"（{player}）" if player else ""
    try:
        with wave.open(path) as w:
            frames = w.getnframes()
            rate = w.getframerate()
            raw = w.readframes(frames)
    except (wave.Error, EOFError) as exc:
        return [Violation("media_duration", "media",
                          f"{os.path.basename(path)}{who} 无法读取：{exc}",
                          player=player)]
    if frames <= 0:
        return [Violation("media_silent", "media",
                          f"{os.path.basename(path)}{who} 无音频帧",
                          player=player)]
    peak = 0
    nz = 0
    for i in range(0, len(raw) - 1, 2):
        v = int.from_bytes(raw[i:i + 2], "little", signed=True)
        a = abs(v)
        if a > peak:
            peak = a
        if a > 8:
            nz += 1
    if peak < 128 or nz == 0:
        out.append(Violation("media_silent", "media",
                             f"{os.path.basename(path)}{who} 几乎静音"
                             f"（峰值 {peak}）", player=player))
    if peak >= 32767:
        out.append(Violation("media_clipping", "media",
                             f"{os.path.basename(path)}{who} 削波",
                             player=player))
    duration = frames / rate if rate else 0.0
    expected = score.duration_s
    if player is None and abs(duration - expected) > max(1.0,
                                                          expected * 0.25):
        out.append(Violation("media_duration", "media",
                             f"混音时长 {duration:.2f}s 与谱面 "
                             f"{expected:.2f}s 偏差过大"))
    return out


def _verify_musicxml(path: str, assignments) -> list[Violation]:
    """核验分谱的移调正确性。

    职责边界：MusicXML 分谱按每件乐器每条声部写成单声部谱（多演奏者共用同一
    乐器时合并为一件），不保证“实际发声的每个音都逐字出现”——那是声部覆盖
    验收（从分配与音频直接核验）的职责。这里只检查“写出来的音高是否按移调
    偏移正确记谱”，以及移调乐器是否声明了 <transpose>，避免记谱错音。
    """
    out: list[Violation] = []
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        return [Violation("media_missing", "media",
                          f"分谱 MusicXML 解析失败：{exc}")]
    root = tree.getroot()
    name_to_inst = {i.name: i for i in CATALOG.values()}
    expected: dict[str, set[int]] = {}
    for a in assignments:
        f = a.fragment
        if getattr(f, "kind", "tone") != "tone" or f.midi is None:
            continue
        inst = CATALOG.get(a.instrument_id)
        if inst is None:
            continue
        expected.setdefault(inst.name, set()).add(
            inst.written_midi(f.midi))
    for part in root.findall("part"):
        pid = part.get("id")
        name_el = root.find(f".//score-part[@id='{pid}']/part-name")
        if name_el is None or name_el.text not in name_to_inst:
            continue
        inst = name_to_inst[name_el.text]
        if inst.written_offset != 0 and part.find(".//transpose") is None:
            out.append(Violation(
                "xml_transpose", "media",
                f"{inst.name} 分谱缺少 <transpose>（offset "
                f"{inst.written_offset}）", instrument=inst.name))
        written_found = set()
        for note in part.iter("note"):
            pitch = note.find("pitch")
            if pitch is None:
                continue
            step = pitch.findtext("step")
            alter = int(pitch.findtext("alter") or 0)
            octv = int(pitch.findtext("octave"))
            midi = _step_to_midi(step, alter, octv)
            if midi is not None:
                written_found.add(midi)
        want = expected.get(inst.name, set())
        for midi in sorted(written_found - want):
            out.append(Violation(
                "xml_pitch", "media",
                f"{inst.name} 分谱出现未经正确移调的记谱音高 {midi}"
                f"（应为实际音高 + 偏移 {inst.written_offset}）",
                instrument=inst.name))
    return out


def _step_to_midi(step: str, alter: int, octave: int) -> int | None:
    steps = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    if step not in steps:
        return None
    return (octave + 1) * 12 + steps[step] + alter


# --------------------------------------------------------------------------- #
# 枚举真值 + 指定休息反推交接
# --------------------------------------------------------------------------- #
@dataclass
class EnumerationResult:
    feasible: bool
    solutions: int
    explored: int
    note: str = ""

    def to_dict(self) -> dict:
        return {"feasible": self.feasible, "solutions": self.solutions,
                "explored": self.explored, "note": self.note}


def enumerate_feasible(score, ensemble, max_explore: int = 200_000,
                       solution_cap: int = 500) -> EnumerationResult:
    """穷举“音符→(演奏者,乐器)”组合的可行域，作为可演奏性的真值参照。

    该参照只考虑原样放置（不做跨段拆分/移交），因此适合用于判定
    “不给任何休息时，单靠乐器组合能否覆盖全部材料”这一边界问题。
    """
    from .solver import Assignment, Fragment

    notes = reference_notes(score)
    options: list[list[tuple[str, str]]] = []
    for r in notes:
        opts = [(pid, iid) for pid, ids in ensemble for iid in ids
                if (i := CATALOG.get(iid)) is not None
                and i.low_midi <= r["midi"] <= i.high_midi]
        if not opts:
            return EnumerationResult(False, 0, 0,
                                     f"{r['note_id']} 无乐器可覆盖")
        options.append(opts)

    explored = 0
    solutions = 0

    def dfs(idx: int, picked: list) -> None:
        nonlocal explored, solutions
        if solutions >= solution_cap:
            return
        if idx == len(notes):
            explored += 1
            if explored > max_explore:
                return
            assignments = []
            for r, (pid, iid) in zip(notes, picked):
                frag = Fragment(r["note_id"], r["layer"], r["role"],
                                r["start_s"], r["end_s"] - r["start_s"],
                                r["midi"])
                assignments.append(Assignment(pid, iid, frag))
            rep = verify_assignments(score, assignments, ensemble)
            if rep.passes:
                solutions += 1
            return
        for opt in options[idx]:
            dfs(idx + 1, picked + [opt])

    dfs(0, [])
    feasible = solutions > 0
    note = (f"枚举 {explored} 组组合，可行解 {solutions}"
            if feasible else f"枚举 {explored} 组组合，无可行解")
    return EnumerationResult(feasible, solutions, explored, note)


def derive_rest_segments(duration_s: float, max_sustain_s: float,
                         breath_gap_s: float, rest_s: float,
                         phrase_budget_s: float | None = None,
                         recovery_s: float = 0.0,
                         same_player: bool = False,
                         ) -> list[tuple[float, float]] | None:
    """从“指定休息”反推可行交接：把一段长音切成若干可演奏片段。

    返回 [(相对起点, 时长), ...]；不可行返回 None。休息**占用音乐时间**：
    片段时长之和 + 段间休息必须铺满 duration_s。

    两种情形：
    - 默认（交接给**另一位**演奏者）：段间可不留休息（seamless handoff，
      rest_s=0 合法），每段 ≤ min(单音最大持续, 乐句预算)；接棒者是新人，
      无需为前一段的耗气恢复。
    - same_player=True（同一人换气再奏）：段间休息 rest_s 必须 ≥ 该段结束后的
      所需恢复——达到最大持续的段需 recovery_s，否则需 breath_gap_s。
    """
    if duration_s <= 0:
        return []
    budget = phrase_budget_s if phrase_budget_s is not None else max_sustain_s
    cap = min(max_sustain_s, budget)
    if cap <= EPS:
        return None
    segments: list[tuple[float, float]] = []
    cursor = 0.0
    filled = 0.0
    while filled < duration_s - EPS:
        take = min(cap, duration_s - filled)
        segments.append((cursor, take))
        cursor += take
        filled += take
        if filled >= duration_s - EPS:
            break
        if same_player:
            need = recovery_s if take >= max_sustain_s - EPS else breath_gap_s
            if rest_s + EPS < need:
                return None
        if filled + rest_s > duration_s + EPS:
            return None
        cursor += rest_s
        filled += rest_s
    return segments


def handoff_moment(duration_s: float, max_sustain_s: float,
                   breath_gap_s: float, rest_s: float,
                   phrase_budget_s: float | None = None,
                   recovery_s: float = 0.0,
                   same_player: bool = False) -> float | None:
    """返回第一段结束（即移交/换气）的时刻；不可行返回 None。"""
    segs = derive_rest_segments(duration_s, max_sustain_s, breath_gap_s,
                                rest_s, phrase_budget_s, recovery_s,
                                same_player)
    if not segs:
        return None
    return round(segs[0][0] + segs[0][1], 6)
