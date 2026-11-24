"""配器约束求解器。

搜索目标：把每条材料线的每个音分配给具体演奏者/乐器，先满足全部硬性演奏条件，
再在可行解里优化音色偏好分（软目标）。不做整体八度移位，不做随机换音色。

硬性约束：
1. 实际发声音域（instrument.low/high，MIDI 实际音高）
2. 用户锁定音区（register_lock，按实际音高校验）
3. 复音上限（管乐=1 禁止重叠；弦乐双音限音程；钢琴和弦上限）
4. 最小起奏间隔（吐音/换把大跳）
5. 管乐呼吸模型：单音最大持续、乐句累计预算、换气休止、恢复时长
6. 换奏时间（兼任乐器切换需要音乐时间）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .instruments import CATALOG, Instrument
from .pitch import beats_to_seconds
from .score import Note, ROLES, Score


@dataclass
class Fragment:
    """一次实际演奏片段（可能由原音符拆分而来，用于移交/换气）。"""
    note_id: str
    layer_id: str
    role: str
    start_s: float
    duration_s: float
    midi: int
    split_index: int = 0
    split_total: int = 1
    kind: str = "tone"          # tone / breath_rest
    handoff_from: str | None = None  # 上一片段的演奏者 id（移交）
    reason: str = ""

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass
class Assignment:
    player_id: str
    instrument_id: str
    fragment: Fragment


@dataclass
class FailureRecord:
    note_id: str
    layer_id: str
    role: str
    midi: int
    start_s: float
    duration_s: float
    reasons: list[str]


@dataclass
class PlayerRuntime:
    player_id: str
    instruments: list[str]
    current_inst: str | None = None
    # 当前乐器上时间最晚的发声片段
    last_tone_end: float = -1e9
    last_tone_start: float = -1e9
    last_tone_midi: int | None = None
    # 活跃发声（按 end 排序维护）
    active: list[Fragment] = field(default_factory=list)
    # 管乐呼吸模型
    phrase_start: float = 0.0
    phrase_used: float = 0.0
    last_breath_end: float = -1e9
    exhausted_until: float = -1e9
    assignments: list[Assignment] = field(default_factory=list)
    # 事件时间线（含换气/换奏占位），用于恢复时长检验
    events: list[dict] = field(default_factory=list)

    def clone(self) -> "PlayerRuntime":
        return PlayerRuntime(
            self.player_id, list(self.instruments), self.current_inst,
            self.last_tone_end, self.last_tone_start, self.last_tone_midi,
            list(self.active), self.phrase_start, self.phrase_used,
            self.last_breath_end, self.exhausted_until,
            list(self.assignments), list(self.events),
        )


@dataclass
class SolveResult:
    feasible: bool
    assignments: list[Assignment]
    score: float
    nodes: int
    failures: list[FailureRecord]
    explanation: str
    split_notes: dict[str, int]          # note_id -> 拆分片段数
    role_fallbacks: list[dict]           # 改配内声等角色让步记录
    trace: list[dict]                    # 可追踪分配过程
    player_summary: list[dict]           # 每位演奏者恢复时长检验


class Solver:
    def __init__(self, score: Score, ensemble: list[tuple[str, list[str]]],
                 node_budget: int = 12_000, handoff_penalty: float = 0.12,
                 role_fallback_penalty: float = 0.25,
                 candidate_width: int = 6):
        self.score = score
        self.bpm = score.bpm
        self.players = [
            PlayerRuntime(pid, [iid for iid in ids if iid in CATALOG])
            for pid, ids in ensemble
        ]
        self.players = [p for p in self.players if p.instruments]
        self.node_budget = node_budget
        self.candidate_width = candidate_width
        self.handoff_penalty = handoff_penalty
        self.role_fallback_penalty = role_fallback_penalty
        self.nodes = 0
        self.best: list[Assignment] | None = None
        self.best_score = -math.inf
        self.failures: list[FailureRecord] = []
        self.trace: list[dict] = []
        self.role_fallbacks: list[dict] = []
        self.split_notes: dict[str, int] = {}
        self._plan: list[Fragment] = []
        self._note_max_gain: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # 入口
    # ------------------------------------------------------------------ #
    def solve(self) -> SolveResult:
        notes = [n for n in self.score.all_notes() if not n.is_rest()]
        layer_role = {l.id: l.role for l in self.score.layers}
        self._plan = []
        for n in notes:
            start_s = beats_to_seconds(n.start, self.bpm)
            dur_s = beats_to_seconds(n.duration, self.bpm)
            self._plan.append(Fragment(n.id, n.layer_id, layer_role[n.layer_id],
                                       start_s, dur_s, n.midi, reason="原始音符"))
        self._note_max_gain = self._estimate_max_gains()
        # 阶段 1：贪心快速得到可行解，保证可交付、不打满预算
        greedy = self._greedy(self._plan, [p.clone() for p in self.players])
        if greedy is not None:
            self.best = greedy
            self.best_score = self._score_of(greedy)
        # 阶段 2：有限回溯，在硬性可行空间内优化音色/连续性/负载
        self._search(self._plan, 0, [p.clone() for p in self.players], 0.0, [])
        if self.best is not None:
            for a in self.best:
                f = a.fragment
                if f.split_total > 1:
                    self.split_notes[f.note_id] = max(
                        self.split_notes.get(f.note_id, 1), f.split_total)
            trace = self._build_trace(self.best)
            summary = self._player_summary(self.best)
            self._record_role_fallbacks(self.best)
            return SolveResult(
                True, self.best, self.best_score, self.nodes, [],
                "找到满足全部硬性约束的可执行配器方案。",
                dict(self.split_notes), list(self.role_fallbacks),
                trace, summary,
            )
        explanation = self._build_explanation()
        return SolveResult(
            False, [], 0.0, self.nodes, self.failures, explanation,
            dict(self.split_notes), list(self.role_fallbacks),
            list(self.trace), [],
        )

    def _build_trace(self, assignments: list[Assignment]) -> list[dict]:
        rows = []
        for a in sorted(assignments, key=lambda x: (x.fragment.start_s,
                                                    x.player_id)):
            f = a.fragment
            inst = CATALOG[a.instrument_id]
            if f.kind == "breath_rest":
                action = "breath-rest"
            elif f.handoff_from:
                action = "handoff-in"
            elif f.split_total > 1:
                action = "handoff-out" if f.split_index == 0 else "split"
            else:
                action = "direct"
            rows.append({
                "note_id": f.note_id,
                "layer": f.layer_id,
                "role": f.role,
                "midi": f.midi,
                "start_s": round(f.start_s, 3),
                "duration_s": round(f.duration_s, 3),
                "player": a.player_id,
                "instrument": inst.id,
                "instrument_name": inst.name,
                "action": action,
                "split": f"{f.split_index + 1}/{f.split_total}"
                if f.split_total > 1 else "",
                "reason": f.reason,
            })
        return rows

    def _record_role_fallbacks(self, assignments: list[Assignment]) -> None:
        """主旋律被非旋律强项乐器承担（管乐争抢后的改配）记录为角色让步。"""
        seen = set()
        for a in assignments:
            f = a.fragment
            if f.kind != "tone" or f.role != "melody":
                continue
            inst = CATALOG[a.instrument_id]
            if inst.timbre_for("melody") < 0.6 and f.note_id not in seen:
                seen.add(f.note_id)
                self.role_fallbacks.append({
                    "note_id": f.note_id,
                    "player": a.player_id,
                    "instrument": inst.name,
                    "original_role": "melody",
                    "reason": "旋律线被管乐呼吸/换把约束占用，改由内声型乐器承担",
                })

    # ------------------------------------------------------------------ #
    # 回溯搜索（按音符全局起点顺序；拆分在放置时动态生成）
    # ------------------------------------------------------------------ #
    def _estimate_max_gains(self) -> dict[str, float]:
        """每个音符在全体乐器中的最高音色分（分支限界上界）。"""
        out: dict[str, float] = {}
        for frag in self._plan:
            best = 0.0
            for inst in CATALOG.values():
                if inst.low_midi <= frag.midi <= inst.high_midi:
                    best = max(best, inst.timbre_for(frag.role))
            out[frag.note_id] = best + 0.05
        return out

    def _greedy(self, plan: list[Fragment],
                players: list[PlayerRuntime]) -> list[Assignment] | None:
        """逐音符选当前最优可行候选；任何音符无候选即判定无解。"""
        assigned: list[Assignment] = []
        for frag in plan:
            layer = next(l for l in self.score.layers if l.id == frag.layer_id)
            candidates = self._candidates(frag, layer, players)
            if not candidates:
                self._record_failure(frag, layer, players)
                return None
            plan_steps, _gain = candidates[0]
            for pid, inst_id, f in plan_steps:
                target = next(p for p in players if p.player_id == pid)
                if not self._place(target, inst_id, f):
                    self._record_failure(frag, layer, players)
                    return None
                assigned.append(Assignment(pid, inst_id, f))
        return assigned

    def _score_of(self, assignments: list[Assignment]) -> float:
        """方案级评分：音色偏好 + 同线连续性 - 移交/换气拆分 - 负载失衡。"""
        total = 0.0
        last_by_layer: dict[str, tuple[str, str]] = {}
        load: dict[str, float] = {}
        for a in assignments:
            f = a.fragment
            if f.kind == "breath_rest":
                continue
            inst = CATALOG[a.instrument_id]
            total += inst.timbre_for(f.role)
            load[a.player_id] = load.get(a.player_id, 0.0) + f.duration_s
            key = f.layer_id
            if key in last_by_layer and last_by_layer[key] == (
                    a.player_id, a.instrument_id):
                total += 0.05
            if f.split_total > 1:
                total -= self.handoff_penalty * 0.5
            if f.handoff_from:
                total -= self.handoff_penalty
            last_by_layer[key] = (a.player_id, a.instrument_id)
        if load:
            peak = max(load.values())
            total -= 0.02 * peak * len(load)
        return total

    def _search(self, plan: list[Fragment], idx: int,
                players: list[PlayerRuntime], cur_score: float,
                assigned: list[Assignment]) -> bool:
        if self.nodes >= self.node_budget:
            return self.best is not None
        if idx >= len(plan):
            if cur_score > self.best_score:
                self.best_score = cur_score
                self.best = list(assigned)
            return True
        # 分支限界：即使后续全取上界也超不过当前最优，则剪枝
        upper = cur_score + sum(
            self._note_max_gain.get(plan[j].note_id, 1.0)
            for j in range(idx, len(plan)))
        if upper <= self.best_score:
            return self.best is not None
        self.nodes += 1
        frag = plan[idx]
        layer = next(l for l in self.score.layers if l.id == frag.layer_id)
        candidates = self._candidates(frag, layer, players)
        found_any = False
        for plan_steps, gain in candidates:
            new_players = [p.clone() for p in players]
            new_assigned = list(assigned)
            ok = True
            for pid, inst_id, f in plan_steps:
                target = next(p for p in new_players
                              if p.player_id == pid)
                if not self._place(target, inst_id, f):
                    ok = False
                    break
                new_assigned.append(Assignment(pid, inst_id, f))
            if not ok:
                continue
            found = self._search(plan, idx + 1, new_players,
                                 cur_score + gain, new_assigned)
            found_any = found_any or found
        if not found_any:
            self._record_failure(frag, layer, players)
        return found_any

    # ------------------------------------------------------------------ #
    # 候选生成：对每个演奏者/乐器生成 (直接放置 / 移交拆分 / 同管换气拆分)
    # ------------------------------------------------------------------ #
    def _candidates(self, frag: Fragment, layer,
                    players: list[PlayerRuntime]):
        out = []
        role = frag.role
        for p in players:
            for inst_id in p.instruments:
                inst = CATALOG[inst_id]
                # 静态硬条件：音域、锁定音区
                if self._static_check(inst, frag, layer.register_lock):
                    continue
                for plan_steps, gain in self._plans_for(p, inst, frag, role,
                                                        players):
                    # 负载均衡：已承担发声越多，增量分越低
                    gain -= 0.015 * len(p.assignments)
                    out.append((plan_steps, gain))
        # 音色偏好 + 连续性打分排序（软目标，硬条件已先过滤）
        out.sort(key=lambda c: -c[1])
        # 束搜索式截断：只保留得分最高的若干候选，控制组合爆炸
        return out[:self.candidate_width]

    def _static_check(self, inst: Instrument, frag: Fragment,
                      register_lock) -> str | None:
        if not (inst.low_midi <= frag.midi <= inst.high_midi):
            return (f"音 {frag.midi} 超出 {inst.name} 实际音域 "
                    f"[{inst.low_midi},{inst.high_midi}]")
        if register_lock and not (register_lock[0] <= frag.midi <= register_lock[1]):
            return f"音 {frag.midi} 逃离用户锁定音区 {register_lock}"
        return None

    def _plans_for(self, p: PlayerRuntime, inst: Instrument,
                   frag: Fragment, role: str,
                   players: list[PlayerRuntime]):
        """返回 [(plan_steps, 得分增量), ...]；不可行返回 []。

        plan_steps 为 [(player_id, instrument_id, fragment), ...]。
        长音/超预算音可能给出多套移交方案（不同接棒者），供回溯择优。"""
        # 1) 原样放置
        if self._can_place(p, inst, frag):
            gain = self._tone_score(p, inst, frag, role, split=False)
            return [([(p.player_id, inst.id, frag)], gain)]

        if not inst.is_wind:
            return []

        # 2) 管乐：超口气/超长音 -> 移交拆分优先，其次同管换气拆分
        return self._wind_split_plans(p, inst, frag, role, players)

    def _wind_split_plans(self, p: PlayerRuntime, inst: Instrument,
                          frag: Fragment, role: str,
                          players: list[PlayerRuntime]):
        """长音/超预算音的两种拆分：移交（优先）或同一人换气再奏。"""
        remaining_budget = self._phrase_remaining(p, inst, frag.start_s)
        if remaining_budget is None:
            return []
        # 前段不得超过单音最大持续，也不超过当前口气剩余
        first_dur = min(remaining_budget, inst.max_sustain_s)
        plans = []
        # 情形 A：无缝移交给另一位管乐演奏者（同一旋律线不中断）
        plans.extend(self._handoff_plans(p, inst, frag, role, first_dur,
                                         players))

        # 情形 B：同一人换气后再奏后段（音乐时间内插入真实换气休止）
        need = frag.duration_s
        if first_dur > 0 and need > first_dur + inst.breath_gap_s:
            f1 = replace(frag, duration_s=first_dur, split_index=0,
                         split_total=2,
                         reason=f"口气预算内前段({first_dur:.2f}s)")
            rest = Fragment(frag.note_id, frag.layer_id, frag.role,
                            f1.end_s, inst.breath_gap_s, frag.midi,
                            split_index=0, split_total=2, kind="breath_rest",
                            reason=f"{inst.name}换气休止 {inst.breath_gap_s}s")
            f2_start = rest.end_s
            f2 = replace(frag, start_s=f2_start,
                         duration_s=frag.end_s - f2_start,
                         split_index=1, split_total=2,
                         reason="换气后再奏后段")
            probe = p.clone()
            if self._can_place(probe, inst, f1):
                self._place(probe, inst.id, f1)
                self._place_rest(probe, inst, rest)
                if self._can_place(probe, inst, f2):
                    gain = (self._tone_score(p, inst, f1, role, split=True)
                            + self._tone_score(p, inst, f2, role, split=True)
                            - self.handoff_penalty)
                    plans.append(([(p.player_id, inst.id, f1),
                                   (p.player_id, inst.id, rest),
                                   (p.player_id, inst.id, f2)], gain))
        return plans

    def _handoff_plans(self, p: PlayerRuntime, inst: Instrument,
                       frag: Fragment, role: str, first_dur: float,
                       players: list[PlayerRuntime]):
        """为长音前段生成若干无缝移交方案（接棒者按音色分从优到劣）。

        关键：接棒者从**当前搜索状态** `players` 挑选并现场校验，
        避免选中已被占用或仍在恢复的演奏者；返回多套方案使回溯能绕过
        刚奏满最大持续的演奏者。"""
        if first_dur <= 0 or frag.duration_s <= first_dur + 1e-9:
            return []
        f1 = replace(frag, duration_s=first_dur, split_index=0, split_total=2,
                     reason=f"移交前段 {first_dur:.2f}s")
        f2 = replace(frag, start_s=frag.start_s + first_dur,
                     duration_s=frag.duration_s - first_dur,
                     split_index=1, split_total=2,
                     handoff_from=p.player_id, reason="无缝移交后段")
        targets = []
        for other in players:
            if other.player_id == p.player_id:
                continue
            for oid in other.instruments:
                oinst = CATALOG[oid]
                if self._static_check(oinst, f2, self._lock_of(frag.layer_id)):
                    continue
                if not oinst.is_wind:
                    continue
                if not self._can_place(other, oinst, f2):
                    continue
                sc = self._tone_score(other, oinst, f2, role, split=True)
                targets.append((sc, other.player_id, oid, other))
        targets.sort(key=lambda t: -t[0])
        plans = []
        for sc, other_pid, oid, other in targets:
            gain = (self._tone_score(p, inst, f1, role, split=True)
                    + sc - self.handoff_penalty)
            plans.append(([(p.player_id, inst.id, f1),
                           (other_pid, oid, f2)], gain))
        return plans

    # ------------------------------------------------------------------ #
    # 可行性检查与状态放置
    # ------------------------------------------------------------------ #
    def _lock_of(self, layer_id: str):
        return next(l.register_lock for l in self.score.layers
                    if l.id == layer_id)

    def _phrase_remaining(self, p: PlayerRuntime, inst: Instrument,
                          start_s: float) -> float | None:
        """返回该演奏者在 start_s 时刻当前口气还能发多少秒；None 表示还在恢复/换气。"""
        if start_s < p.exhausted_until:
            return None
        # 新乐句：距上一发声有 >= breath_gap 的休止
        if (p.last_tone_end < 0 or start_s - p.last_tone_end
                >= inst.breath_gap_s):
            return inst.phrase_budget_s
        return max(0.0, inst.phrase_budget_s - p.phrase_used)

    def _can_place(self, p: PlayerRuntime, inst: Instrument,
                   frag: Fragment) -> bool:
        return self._violations(p, inst, frag) == []

    def _violations(self, p: PlayerRuntime, inst: Instrument,
                    frag: Fragment) -> list[str]:
        reasons: list[str] = []
        # 换奏时间
        if p.current_inst is not None and p.current_inst != inst.id:
            gap = frag.start_s - p.last_tone_end
            if gap < inst.change_time_s:
                reasons.append(
                    f"换奏 {inst.name} 需要 {inst.change_time_s}s，仅间隔 {gap:.2f}s")
        # 最小起奏间隔 / 弦乐换把（与上一次起音比较，独立于重叠判断）
        if p.last_tone_start >= 0:
            gap = frag.start_s - p.last_tone_start
            if gap < inst.min_attack_gap_s - 1e-9:
                reasons.append(
                    f"起奏间隔 {gap:.3f}s < 最小 {inst.min_attack_gap_s}s")
            if (inst.shift_leap_semitones is not None
                    and p.last_tone_midi is not None
                    and abs(frag.midi - p.last_tone_midi)
                    >= inst.shift_leap_semitones
                    and gap < inst.shift_gap_s):
                reasons.append(
                    f"{inst.name}换把大跳 {abs(frag.midi - p.last_tone_midi)} 半音，"
                    f"需 {inst.shift_gap_s}s 间隔")
        # 复音 / 重叠
        overlapping = [a for a in p.active if a.end_s > frag.start_s + 1e-9]
        if overlapping:
            if inst.polyphony <= 1:
                reasons.append(f"{inst.name}复音上限为1，存在重叠音")
            elif len(overlapping) + 1 > inst.polyphony:
                reasons.append(f"{inst.name}复音上限 {inst.polyphony}，"
                               f"重叠音数 {len(overlapping) + 1}")
            else:
                for a in overlapping:
                    if (inst.max_double_stop is not None
                            and abs(a.midi - frag.midi) > inst.max_double_stop):
                        reasons.append(
                            f"双音音程 {abs(a.midi - frag.midi)} 半音 > "
                            f"{inst.max_double_stop}")
        # 管乐呼吸
        if inst.is_wind:
            if frag.start_s < p.exhausted_until - 1e-9:
                reasons.append(
                    f"仍在恢复中（至 {p.exhausted_until:.2f}s）")
            else:
                remaining = self._phrase_remaining(p, inst, frag.start_s)
                if remaining is None:
                    reasons.append("换气/恢复未完成")
                elif frag.duration_s > inst.max_sustain_s + 1e-9:
                    reasons.append(
                        f"单音 {frag.duration_s:.2f}s > 最大持续 "
                        f"{inst.max_sustain_s}s，需移交或拆分")
                elif frag.duration_s > remaining + 1e-9:
                    reasons.append(
                        f"乐句剩余气息 {remaining:.2f}s 不足 {frag.duration_s:.2f}s")
        return reasons

    def _place(self, p: PlayerRuntime, inst_id: str, frag: Fragment) -> bool:
        inst = CATALOG[inst_id]
        if frag.kind == "breath_rest":
            self._place_rest(p, inst, frag)
            return True
        if not self._can_place(p, inst, frag):
            return False
        if inst.is_wind:
            if (p.last_tone_end < 0 or frag.start_s - p.last_tone_end
                    >= inst.breath_gap_s):
                p.phrase_start = frag.start_s
                p.phrase_used = 0.0
            p.phrase_used += frag.duration_s
            # 用尽口气 -> 进入恢复
            if (p.phrase_used >= inst.phrase_budget_s - 1e-9
                    or frag.duration_s >= inst.max_sustain_s - 1e-9):
                p.exhausted_until = frag.end_s + inst.recovery_s
                p.last_breath_end = frag.end_s + inst.breath_gap_s
                p.events.append({"type": "exhaust", "time": frag.end_s,
                                 "until": p.exhausted_until,
                                 "instrument": inst.name})
            else:
                p.last_breath_end = frag.end_s + inst.breath_gap_s
        p.current_inst = inst_id
        p.last_tone_start = frag.start_s
        p.last_tone_end = frag.end_s
        p.last_tone_midi = frag.midi
        p.active = [a for a in p.active if a.end_s > frag.start_s + 1e-9]
        p.active.append(frag)
        p.active.sort(key=lambda a: a.end_s)
        p.assignments.append(Assignment(p.player_id, inst_id, frag))
        p.events.append({"type": "tone", "time": frag.start_s,
                         "end": frag.end_s, "midi": frag.midi,
                         "instrument": inst.name, "note_id": frag.note_id})
        return True

    def _place_rest(self, p: PlayerRuntime, inst: Instrument,
                    frag: Fragment) -> None:
        p.events.append({"type": "breath", "time": frag.start_s,
                         "end": frag.end_s, "instrument": inst.name,
                         "note_id": frag.note_id})
        # 换气重置乐句预算
        p.phrase_used = 0.0
        p.phrase_start = frag.end_s
        p.last_tone_end = max(p.last_tone_end, frag.end_s)

    # ------------------------------------------------------------------ #
    # 评分（软偏好）
    # ------------------------------------------------------------------ #
    def _tone_score(self, p: PlayerRuntime, inst: Instrument,
                    frag: Fragment, role: str, split: bool) -> float:
        base = inst.timbre_for(role)
        # 连续性奖励：与该演奏者上一音符同乐器同线，减少移交
        if p.assignments and p.assignments[-1].instrument_id == inst.id:
            base += 0.05
        if split:
            base -= 0.05
        # 角色让步（旋律被内声型乐器承担）施加软代价，但绝不越过硬性约束
        if role == "melody" and inst.timbre_for("melody") < 0.6:
            base -= self.role_fallback_penalty
        return base

    # ------------------------------------------------------------------ #
    # 无解记录与解释
    # ------------------------------------------------------------------ #
    def _record_failure(self, frag: Fragment, layer,
                        players: list[PlayerRuntime]) -> None:
        reasons: list[str] = []
        for p in players:
            for inst_id in p.instruments:
                inst = CATALOG[inst_id]
                sr = self._static_check(inst, frag, layer.register_lock)
                if sr:
                    reasons.append(f"{p.player_id}/{inst.name}: {sr}")
                    continue
                probe = p.clone()
                vr = self._violations(probe, inst, frag)
                if vr:
                    reasons.append(f"{p.player_id}/{inst.name}: "
                                   + "；".join(vr))
        key = (frag.note_id, tuple(sorted(reasons)))
        if not any(f.note_id == frag.note_id for f in self.failures):
            self.failures.append(FailureRecord(
                frag.note_id, frag.layer_id, frag.role, frag.midi,
                frag.start_s, frag.duration_s, reasons))

    def _build_explanation(self) -> str:
        if not self.failures:
            return ("搜索节点预算内未找到完整方案，但未记录具体冲突；"
                    "可扩大编制或放宽材料。")
        lines = ["无可行配器方案。按音符列出全部被硬性约束拒绝的原因："]
        for f in self.failures:
            lines.append(
                f"- {f.note_id}（{f.role}，MIDI {f.midi}，"
                f"起点 {f.start_s:.2f}s，时长 {f.duration_s:.2f}s）:")
            for r in f.reasons:
                lines.append(f"    * {r}")
        lines.append("建议：增加可覆盖该音区的乐器、拆分材料线、"
                     "增加换气休止或允许声部移交。")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 恢复时长检验（方案级复核）
    # ------------------------------------------------------------------ #
    def _player_summary(self, assignments: list[Assignment]) -> list[dict]:
        by_player: dict[str, list[Assignment]] = {}
        for a in assignments:
            by_player.setdefault(a.player_id, []).append(a)
        summary = []
        for p in self.players:
            items = sorted(by_player.get(p.player_id, []),
                           key=lambda a: a.fragment.start_s)
            tones = [a for a in items if a.fragment.kind == "tone"]
            rests = [a for a in items if a.fragment.kind == "breath_rest"]
            inst_changes = 0
            cur = None
            recovery_checks = []
            phrase_used = 0.0
            phrase_start = None
            last_end = None
            for a in items:
                inst = CATALOG[a.instrument_id]
                f = a.fragment
                if cur is not None and cur != a.instrument_id:
                    inst_changes += 1
                if f.kind == "breath_rest":
                    phrase_used = 0.0
                    phrase_start = f.end_s
                    cur = a.instrument_id
                    continue
                if inst.is_wind and (last_end is None
                                     or f.start_s - last_end >= inst.breath_gap_s):
                    phrase_used = 0.0
                    phrase_start = f.start_s
                phrase_used += f.duration_s
                ok = True
                detail = ""
                if inst.is_wind:
                    ok = (f.duration_s <= inst.max_sustain_s + 1e-9
                          and phrase_used <= inst.phrase_budget_s + 1e-9)
                    detail = (f"乐句累计 {phrase_used:.2f}/"
                              f"{inst.phrase_budget_s}s，"
                              f"单音 {f.duration_s:.2f}/{inst.max_sustain_s}s")
                recovery_checks.append({
                    "note_id": f.note_id, "instrument": inst.name,
                    "start_s": round(f.start_s, 3),
                    "duration_s": round(f.duration_s, 3),
                    "ok": ok, "detail": detail,
                })
                last_end = f.end_s
                cur = a.instrument_id
            total_play = sum(a.fragment.duration_s for a in tones)
            summary.append({
                "player": p.player_id,
                "instruments": p.instruments,
                "tones": len(tones),
                "breath_rests": len(rests),
                "instrument_changes": inst_changes,
                "total_playing_s": round(total_play, 3),
                "all_recoveries_ok": all(c["ok"] for c in recovery_checks),
                "checks": recovery_checks,
            })
        return summary
