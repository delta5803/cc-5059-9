"""演奏能力验收测试：独立参照、能力违例、声部覆盖、边界真值、受控错误。

运行：python3 -m unittest discover -s tests -v

设计要点（防止配器约束测试伪报）：
- 期望值从**原始谱面 + instruments.py 文档化常量**独立推导，不抄求解器输出。
- 每个“坏结果”用例都是对真实可行解做**受控错误注入**，断言验收器确实报错。
- 反例“单人可演奏、合奏无解”用枚举真值 + 具体冲突声部/交接时刻双重佐证。
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chamber_orchestrator import CATALOG, ENSEMBLE_PRESETS, Solver
from chamber_orchestrator.examples import EXAMPLES
from chamber_orchestrator.orchestrate import orchestrate
from chamber_orchestrator.pitch import beats_to_seconds
from chamber_orchestrator.score import (ROLE_INNER, ROLE_MELODY,
                                         Score, make_line)
from chamber_orchestrator.solver import Assignment, Fragment
from chamber_orchestrator import boundary_examples as bx
from chamber_orchestrator.verify import (derive_rest_segments,
                                         enumerate_feasible, handoff_moment,
                                         verify_assignments)


# --------------------------------------------------------------------------- #
# 独立参照工具（与求解器实现无关，仅用谱面 + 常量重算）
# --------------------------------------------------------------------------- #
def ref_notes(score):
    """从谱面独立重算 (note_id, layer, role, start_s, end_s, midi)。"""
    role = {l.id: l.role for l in score.layers}
    out = []
    for layer in score.layers:
        for n in layer.notes:
            if n.is_rest():
                continue
            out.append({
                "note_id": n.id, "layer": layer.id,
                "role": role[layer.id],
                "start_s": beats_to_seconds(n.start, score.bpm),
                "end_s": beats_to_seconds(n.end, score.bpm),
                "midi": (n.midi if n.midi is not None else None),
            })
    return out


def expected_wind_violations(score, assignments):
    """独立推导每条管乐发声应触发的能力违例代码集合。

    只对“原样放置片段”成立：逐条片段按起止时间推演单音持续与乐句预算。
    这里刻意用与验收器不同的写法（先分组再排时间线），用于交叉对照。
    """
    out = {}
    by_player = {}
    for a in assignments:
        if a.fragment.kind != "tone":
            continue
        by_player.setdefault(a.player_id, []).append(a)
    for pid, items in by_player.items():
        items.sort(key=lambda a: a.fragment.start_s)
        used = 0.0
        prev_end = None
        for a in items:
            f = a.fragment
            inst = CATALOG[a.instrument_id]
            codes = set()
            if inst.is_wind:
                reset = prev_end is None or \
                    f.start_s - prev_end >= inst.breath_gap_s - 1e-9
                if reset:
                    used = 0.0
                if f.duration_s > inst.max_sustain_s + 1e-9:
                    codes.add("sustain")
                elif f.duration_s > inst.phrase_budget_s - used + 1e-9:
                    codes.add("phrase_budget")
                used += f.duration_s
            out.setdefault(f.note_id, set()).update(codes)
            prev_end = f.end_s
    return out


def code_set(report):
    return {v.code for v in report.capability_violations}


# --------------------------------------------------------------------------- #
# 1. 独立参照：真实可行解必须原样通过验收（不伪报）
# --------------------------------------------------------------------------- #
class ReferenceAgreementTests(unittest.TestCase):
    CASES = [("long_tone", "winds"), ("fast_alternation", "winds"),
             ("locked_register", "full"), ("doubling_conflict", "winds"),
             ("range_conflict", "full")]

    def test_all_feasible_solutions_pass_independent_verifier(self):
        for key, preset in self.CASES:
            with self.subTest(case=key):
                score = EXAMPLES[key][0]()
                ens = ENSEMBLE_PRESETS[preset]
                res = Solver(score, ens).solve()
                self.assertTrue(res.feasible)
                rep = verify_assignments(score, res.assignments, ens)
                self.assertTrue(rep.passes, rep.summary())
                self.assertEqual(rep.capability_violations, [])
                self.assertEqual(rep.coverage_violations, [])
                self.assertTrue(rep.coverage_complete)

    def test_voice_coverage_matches_reference_notes(self):
        score = EXAMPLES["locked_register"][0]()
        res = Solver(score, ENSEMBLE_PRESETS["full"]).solve()
        rep = verify_assignments(score, res.assignments,
                                 ENSEMBLE_PRESETS["full"])
        refs = ref_notes(score)
        self.assertEqual(rep.voices_expected, len(refs))
        self.assertEqual(rep.voices_covered, len(refs))
        by_id = {c.note_id: c for c in rep.coverage}
        for r in refs:
            entry = by_id[r["note_id"]]
            self.assertTrue(entry.covered)
            self.assertEqual(entry.midi, r["midi"])
            self.assertEqual(entry.gaps, [])

    def test_independent_cross_check_of_wind_violations(self):
        """交叉对照：可行解零违例；对坏解，独立重算与验收器给出相同管乐代码。"""
        score = EXAMPLES["long_tone"][0]()
        ens = ENSEMBLE_PRESETS["winds"]
        res = Solver(score, ens).solve()
        expected = expected_wind_violations(score, res.assignments)
        self.assertEqual({nid: c for nid, c in expected.items() if c}, {})

        # 坏解：把 p4 低音片段拆成 4 段各 3.0s 背靠背（单段<最大持续 9s，
        # 但累计 12s > 大管乐句预算 11s，且间隔 0 < 换气下限）
        bad = []
        for a in res.assignments:
            f = a.fragment
            if a.player_id == "p4" and f.note_id == "bass.n0":
                for k in range(4):
                    bad.append(Assignment("p4", "bassoon",
                                          Fragment("bass.n0", "bass", "bass",
                                                   k * 3.0, 3.0, 48)))
            else:
                bad.append(a)
        independent = expected_wind_violations(score, bad)
        rep = verify_assignments(score, bad, ens)
        verifier_codes = code_set(rep)
        # 独立重算标出的管乐违例代码，必须被验收器同集合报出
        self.assertTrue(any(c for c in independent.values()))
        self.assertTrue(set().union(*independent.values())
                        <= verifier_codes, (independent, verifier_codes))


# --------------------------------------------------------------------------- #
# 2. 能力违例：受控错误注入必须被验收器揭露
# --------------------------------------------------------------------------- #
class CapabilityViolationTests(unittest.TestCase):
    def setUp(self):
        self.score = EXAMPLES["long_tone"][0]()
        self.ens = ENSEMBLE_PRESETS["winds"]
        self.res = Solver(self.score, self.ens).solve()
        self.assertTrue(self.res.feasible)

    def _clone(self, **override):
        out = []
        for a in self.res.assignments:
            f = a.fragment
            new = dict(note_id=f.note_id, layer_id=f.layer_id, role=f.role,
                       start_s=f.start_s, duration_s=f.duration_s,
                       midi=f.midi, split_index=f.split_index,
                       split_total=f.split_total, kind=f.kind,
                       handoff_from=f.handoff_from, reason=f.reason)
            new.update(override.get(f.note_id, {}))
            out.append(Assignment(a.player_id, a.instrument_id,
                                  Fragment(**new)))
        return out

    def test_stretch_note_beyond_sustain_reports_sustain(self):
        # 把长笛前段强行拉长到 11s（> max_sustain 10s），必须报 sustain
        bad = self._clone(**{"mel.n0": {"duration_s": 11.0}})
        rep = verify_assignments(self.score, bad, self.ens)
        self.assertFalse(rep.passes)
        self.assertIn("sustain", code_set(rep))
        conflicts = {c["note_id"] for c in rep.conflict_voices}
        self.assertIn("mel.n0", conflicts)

    def test_remove_breath_rest_reports_phrase_budget(self):
        # 受控错误：同一人背靠背奏两个 8s 音（均 < 单音最大持续 10s，但累计
        # 16s > 乐句预算 12s，且间隔 0 < 换气下限），必须报 phrase_budget。
        score = Score("背靠背长音", 60, [
            make_line("mel", ROLE_MELODY, "m", 60,
                      [(0, 8, "G4"), (8, 8, "G4")])])
        ens = [("p1", ["flute"])]
        forced = [
            Assignment("p1", "flute",
                       Fragment("mel.n0", "mel", "melody", 0.0, 8.0, 67)),
            Assignment("p1", "flute",
                       Fragment("mel.n1", "mel", "melody", 8.0, 8.0, 67)),
        ]
        rep = verify_assignments(score, forced, ens)
        self.assertFalse(rep.passes)
        self.assertIn("phrase_budget", code_set(rep))

    def test_wrong_pitch_reports_coverage_pitch(self):
        bad = self._clone(**{"mel.n0": {"midi": 67}})  # 应为 64
        rep = verify_assignments(self.score, bad, self.ens)
        self.assertIn("coverage_pitch", {v.code
                                         for v in rep.coverage_violations})

    def test_gap_in_coverage_reported(self):
        bad = self._clone(**{"mel.n0": {"duration_s": 3.0}})
        rep = verify_assignments(self.score, bad, self.ens)
        self.assertIn("coverage_gap",
                      {v.code for v in rep.coverage_violations})
        self.assertFalse(rep.coverage_complete)

    def test_invented_note_reported(self):
        extra = Assignment("p1", "flute",
                           Fragment("ghost.n0", "mel", "melody", 0.0, 1.0,
                                    64))
        rep = verify_assignments(self.score,
                                 self.res.assignments + [extra], self.ens)
        self.assertIn("coverage_unknown",
                      {v.code for v in rep.coverage_violations})

    def test_unconfigured_instrument_reported(self):
        # 把一位演奏者标成未配置的乐器 id
        bad = [Assignment(a.player_id, "tuba", a.fragment)
               for a in self.res.assignments]
        rep = verify_assignments(self.score, bad, self.ens)
        self.assertIn("not_configured", code_set(rep))

    def test_wrong_player_assignment_reported(self):
        # 把 p1 的音硬塞给配置了不同乐器的 p2 -> 未配置
        bad = []
        for a in self.res.assignments:
            pid = "p1" if a.player_id == "p3" else a.player_id
            bad.append(Assignment(pid, a.instrument_id, a.fragment))
        rep = verify_assignments(self.score, bad, self.ens)
        self.assertIn("not_configured", code_set(rep))


# --------------------------------------------------------------------------- #
# 3. 声部覆盖：每声部必须被完整、无重叠、同音高覆盖
# --------------------------------------------------------------------------- #
class VoiceCoverageTests(unittest.TestCase):
    def test_swapping_instrument_config_must_fail(self):
        """故意交换乐器配置：求解器会据此重解；验收时用错误配置必须失败。"""
        score = EXAMPLES["range_conflict"][0]()
        real = ENSEMBLE_PRESETS["full"]
        res = Solver(score, real).solve()
        self.assertTrue(res.feasible)
        # 错误配置：把倍大提琴换成短笛（无法覆盖 E1）
        swapped = [(pid, ["piccolo"] if ids == ["double_bass"] else ids)
                   for pid, ids in real]
        rep = verify_assignments(score, res.assignments, swapped)
        self.assertFalse(rep.passes)
        self.assertIn("not_configured", code_set(rep))

    def test_per_voice_instrument_presence_is_not_playability(self):
        """每个声部都分到乐器 ≠ 配器可演奏：用“每人一件但音高错误”构造。"""
        score = EXAMPLES["locked_register"][0]()
        ens = ENSEMBLE_PRESETS["full"]
        res = Solver(score, ens).solve()
        self.assertTrue(res.feasible)
        # 每个演奏者都有乐器，但把所有旋律音整体下移八度（模拟错误移调）
        bad = []
        for a in res.assignments:
            f = a.fragment
            midi = f.midi - 12 if f.layer_id == "mel" else f.midi
            bad.append(Assignment(a.player_id, a.instrument_id,
                                  Fragment(f.note_id, f.layer_id, f.role,
                                           f.start_s, f.duration_s, midi,
                                           f.split_index, f.split_total,
                                           f.kind, f.handoff_from)))
        rep = verify_assignments(score, bad, ens)
        self.assertFalse(rep.passes)
        self.assertIn("coverage_pitch",
                      {v.code for v in rep.coverage_violations})


# --------------------------------------------------------------------------- #
# 4. 边界真值：枚举可行域 vs 指定休息反推交接必须一致
# --------------------------------------------------------------------------- #
class BoundaryTruthTests(unittest.TestCase):
    def test_two_flutes_simultaneous_long_is_provably_infeasible(self):
        case = bx.TWO_FLUTES_SIMULTANEOUS_LONG
        score = case.score_factory()
        res = Solver(score, case.ensemble).solve()
        self.assertFalse(res.feasible)
        # 枚举真值（只做原样放置）确认非原样放置也无解
        enum = enumerate_feasible(score, case.ensemble)
        self.assertFalse(enum.feasible)

    def test_each_voice_alone_is_playable(self):
        """单人（单声部）可演奏：每条长音单独交给两把长笛都能成。"""
        case = bx.TWO_FLUTES_SIMULTANEOUS_LONG
        for vid, pitch in (("mel", "G4"), ("inner", "A4")):
            with self.subTest(voice=vid):
                score = Score(f"单声部{vid}", 60, [
                    make_line(vid, ROLE_MELODY if vid == "mel" else ROLE_INNER,
                              vid, 60, [(0, 16, pitch)])])
                r = Solver(score, case.ensemble).solve()
                self.assertTrue(r.feasible)

    def test_enumeration_and_rest_derivation_agree_on_handoff(self):
        """两条长音 16s：借“指定休息反推”确定交接时刻，并与枚举真值一致。"""
        # 单把长笛：同一人换气再奏时，10s+休止+6s 超出 16s 窗口，且达到最大
        # 持续后需 recovery 1.2s > breath_gap 0.6s，故同人拆分不可行。
        self.assertIsNone(derive_rest_segments(
            16.0, bx.FLUTE_MAX_SUSTAIN, bx.FLUTE_BREATH_GAP,
            rest_s=bx.FLUTE_BREATH_GAP,
            phrase_budget_s=bx.FLUTE_PHRASE_BUDGET,
            recovery_s=bx.FLUTE_RECOVERY, same_player=True))
        # 无缝交接给另一位演奏者：10s + 6s，交接时刻 10s。
        moment = handoff_moment(16.0, bx.FLUTE_MAX_SUSTAIN,
                                bx.FLUTE_BREATH_GAP, rest_s=0.0,
                                phrase_budget_s=bx.FLUTE_PHRASE_BUDGET)
        self.assertEqual(moment, 10.0)
        # 枚举真值（纯原样放置）在 2 把长笛下无解，与“需要交接”的推断一致。
        score = bx.TWO_FLUTES_SIMULTANEOUS_LONG.score_factory()
        self.assertFalse(enumerate_feasible(
            score, bx.TWO_FLUTES_SIMULTANEOUS_LONG.ensemble).feasible)

    def test_handoff_assignments_validate_independently(self):
        """可行回放：四把长笛在 t=10s 交接的双声部安排必须通过验收。"""
        score, assignments, ens = bx.build_handoff_assignments()
        rep = verify_assignments(score, assignments, ens)
        self.assertTrue(rep.passes, rep.summary())
        moments = {m["note_id"]: m["time_s"] for m in rep.handoff_moments}
        self.assertEqual(moments["mel.n0"], 10.0)
        self.assertEqual(moments["inner.n0"], 10.0)

    def test_counterexample_reports_conflict_voices_and_moments(self):
        """反例的冲突声部与交接时刻可定位：强制排定必须报出冲突声部。"""
        score, assignments, ens = bx.build_handoff_assignments()
        # 故意压回两把长笛：p3/p4 未配置 -> 冲突声部 mel/inner
        bad = [a for a in assignments if a.player_id in ("p1", "p2")]
        rep = verify_assignments(score, bad, [("p1", ["flute"]),
                                              ("p2", ["flute"])])
        self.assertFalse(rep.passes)
        # 覆盖缺口定位到两条声部
        gaps = {v.note_id for v in rep.coverage_violations
                if v.code == "coverage_gap"}
        self.assertEqual(gaps, {"mel.n0", "inner.n0"})


# --------------------------------------------------------------------------- #
# 5. 低音乐器缺失 / 长音：正反例与能力违例 + 覆盖
# --------------------------------------------------------------------------- #
class LowInstrumentAndLongToneTests(unittest.TestCase):
    def test_low_note_with_bass_has_coverage(self):
        case = bx.LOW_NOTE_WITH_BASS
        score = case.score_factory()
        res = Solver(score, case.ensemble).solve()
        self.assertTrue(res.feasible)
        rep = verify_assignments(score, res.assignments, case.ensemble)
        self.assertTrue(rep.passes)
        self.assertEqual(rep.voices_covered, rep.voices_expected)

    def test_low_note_without_bass_reports_range(self):
        case = bx.LOW_NOTE_WITHOUT_BASS
        score = case.score_factory()
        res = Solver(score, case.ensemble).solve()
        self.assertFalse(res.feasible)
        # 强制把低音塞给长笛，验收器必须报 range 且覆盖失败
        score, assignments, ens = bx.forced_low_note_on_flute()
        rep = verify_assignments(score, assignments, ens)
        self.assertFalse(rep.passes)
        self.assertIn("range", code_set(rep))
        self.assertFalse(rep.coverage_complete)

    def test_long_tone_positive_and_negative(self):
        # 正例：长音内声在管乐编制里被拆分/移交后通过验收
        score = EXAMPLES["long_tone"][0]()
        ens = ENSEMBLE_PRESETS["winds"]
        res = Solver(score, ens).solve()
        self.assertTrue(res.feasible)
        self.assertTrue(verify_assignments(score, res.assignments,
                                           ens).passes)
        # 反例：单把长笛吹 16s，验收器必须报 sustain
        s1 = Score("单笛长音", 60, [make_line("mel", ROLE_MELODY, "m", 60,
                                              [(0, 16, "G4")])])
        forced = [Assignment("p1", "flute",
                             Fragment("mel.n0", "mel", "melody", 0.0, 16.0,
                                      67))]
        rep = verify_assignments(s1, forced, [("p1", ["flute"])])
        self.assertIn("sustain", code_set(rep))


# --------------------------------------------------------------------------- #
# 6. 受控错误注入到“室内配器样例”：能力违例 / 覆盖 / 混音必须暴露
# --------------------------------------------------------------------------- #
class ControlledErrorTests(unittest.TestCase):
    def test_inject_capability_error_and_expect_fail(self):
        score = EXAMPLES["fast_alternation"][0]()
        ens = ENSEMBLE_PRESETS["winds"]
        res = Solver(score, ens).solve()
        self.assertTrue(res.feasible)
        # 受控错误：把所有音符起奏间隔压到 0（重叠同人）
        by_player = {}
        for a in res.assignments:
            by_player.setdefault(a.player_id, []).append(a)
        bad = []
        for pid, items in by_player.items():
            t = 0.0
            for a in sorted(items, key=lambda x: x.fragment.start_s):
                f = a.fragment
                if f.kind != "tone":
                    continue
                bad.append(Assignment(pid, a.instrument_id,
                                      Fragment(f.note_id, f.layer_id,
                                               f.role, t, f.duration_s,
                                               f.midi)))
                t += 0.01
        rep = verify_assignments(score, bad, ens)
        self.assertFalse(rep.passes)
        self.assertTrue({"attack_gap", "polyphony", "coverage_gap"}
                        & code_set(rep) or rep.coverage_violations)

    def test_inject_media_error_silent_wav_detected(self):
        """受控错误：导出静音 WAV 必须被混音试听核验发现。"""
        score = EXAMPLES["long_tone"][0]()
        ens = ENSEMBLE_PRESETS["winds"]
        res = Solver(score, ens).solve()
        out = tempfile.mkdtemp()
        payload = orchestrate(score, ens, out)
        self.assertTrue(payload["verification"]["passes"])
        # 覆盖 WAV 为等长静音
        import wave
        wav_path = payload["exports"]["wav"]
        with wave.open(payload["exports"]["wav"], "rb") as w:
            params = (w.getnchannels(), w.getsampwidth(), w.getframerate(),
                      w.getnframes())
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(params[0])
            w.setsampwidth(params[1])
            w.setframerate(params[2])
            w.writeframes(b"\x00\x00" * params[3])
        rep = verify_assignments(score, res.assignments, ens,
                                 export_dir=out)
        self.assertIn("media_silent", {v.code for v in rep.media_violations})
        self.assertFalse(rep.passes)

    def test_inject_bad_musicxml_detected(self):
        """受控错误：分谱写错移调音高必须被 XML 核验发现。"""
        score = EXAMPLES["range_conflict"][0]()
        ens = ENSEMBLE_PRESETS["full"]
        res = Solver(score, ens).solve()
        out = tempfile.mkdtemp()
        payload = orchestrate(score, ens, out)
        xml_path = payload["exports"]["musicxml"]
        with open(xml_path, encoding="utf-8") as fh:
            data = fh.read()
        # 删掉所有 <transpose>：移调乐器分谱将记录实际音高而非记谱音高
        import re
        data = re.sub(r"<transpose>.*?</transpose>", "", data, flags=re.S)
        with open(xml_path, "w", encoding="utf-8") as fh:
            fh.write(data)
        rep = verify_assignments(score, res.assignments, ens,
                                 export_dir=out)
        codes = {v.code for v in rep.media_violations}
        self.assertTrue({"xml_transpose", "xml_pitch"} & codes, codes)


# --------------------------------------------------------------------------- #
# 7. 求解器回归：单演奏者长音不得崩溃（历史缺陷）
# --------------------------------------------------------------------------- #
class SolverRegressionTests(unittest.TestCase):
    def test_single_player_long_tone_does_not_crash(self):
        score = Score("单笛长音", 60, [make_line("mel", ROLE_MELODY, "m",
                                                 60, [(0, 16, "G4")])])
        # 两把长笛时可通过移交解决；不得抛异常
        res = Solver(score, [("p1", ["flute"]), ("p2", ["flute"])]).solve()
        self.assertTrue(res.feasible)
        # 单把长笛应为无解（不崩溃）
        res2 = Solver(score, [("p1", ["flute"])]).solve()
        self.assertFalse(res2.feasible)


if __name__ == "__main__":
    unittest.main(verbosity=2)
