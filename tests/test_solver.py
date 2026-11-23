"""硬性约束与求解行为的单元测试。运行：python3 -m unittest discover -s tests -v"""
from __future__ import annotations

import os
import sys
import unittest
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chamber_orchestrator import CATALOG, ENSEMBLE_PRESETS, Solver
from chamber_orchestrator.examples import EXAMPLES
from chamber_orchestrator.orchestrate import orchestrate


def solve(example_key: str, preset: str):
    score = EXAMPLES[example_key][0]()
    return Solver(score, ENSEMBLE_PRESETS[preset]).solve()


class RangeTests(unittest.TestCase):
    def test_sounding_range_enforced(self):
        res = solve("range_conflict", "small")
        self.assertFalse(res.feasible)
        self.assertTrue(res.failures)
        self.assertIn("超出", res.failures[0].reasons[0])

    def test_range_conflict_solvable_with_bass_instruments(self):
        res = solve("range_conflict", "full")
        self.assertTrue(res.feasible)
        for a in res.assignments:
            if a.fragment.kind != "tone":
                continue
            inst = CATALOG[a.instrument_id]
            self.assertTrue(inst.low_midi <= a.fragment.midi
                            <= inst.high_midi)

    def test_locked_register_blocks_octave_escape(self):
        score = EXAMPLES["locked_register"][0]()
        lock = next(l for l in score.layers if l.id == "mel").register_lock
        res = Solver(score, ENSEMBLE_PRESETS["full"]).solve()
        self.assertTrue(res.feasible)
        for a in res.assignments:
            if a.fragment.layer_id != "mel":
                continue
            self.assertTrue(lock[0] <= a.fragment.midi <= lock[1])


class BreathAndHandoffTests(unittest.TestCase):
    def test_long_tone_is_split_or_handoffed(self):
        res = solve("long_tone", "winds")
        self.assertTrue(res.feasible)
        self.assertIn("mel.n0", res.split_notes)
        actions = [t["action"] for t in res.trace if t["note_id"] == "mel.n0"]
        self.assertIn("handoff-in", actions)

    def test_no_wind_exceeds_sustain_or_phrase_budget(self):
        res = solve("long_tone", "winds")
        self.assertTrue(all(s["all_recoveries_ok"]
                            for s in res.player_summary))
        for a in res.assignments:
            inst = CATALOG[a.instrument_id]
            if inst.is_wind and a.fragment.kind == "tone":
                self.assertLessEqual(a.fragment.duration_s,
                                     inst.max_sustain_s + 1e-9)

    def test_fast_alternation_respects_attack_gap(self):
        res = solve("fast_alternation", "winds")
        self.assertTrue(res.feasible)
        # 同一演奏者相邻起音间隔不得低于其乐器下限
        last_start: dict[str, float] = {}
        for a in sorted(res.assignments,
                        key=lambda x: x.fragment.start_s):
            f = a.fragment
            if f.kind != "tone":
                continue
            inst = CATALOG[a.instrument_id]
            prev = last_start.get(a.player_id)
            if prev is not None:
                self.assertGreaterEqual(
                    f.start_s - prev, inst.min_attack_gap_s - 1e-9)
            last_start[a.player_id] = f.start_s


class PolyphonyTests(unittest.TestCase):
    def test_wind_never_overlaps(self):
        for key in ("long_tone", "fast_alternation", "locked_register"):
            res = solve(key, "winds" if key != "locked_register" else "full")
            by_player = {}
            for a in res.assignments:
                if a.fragment.kind != "tone":
                    continue
                inst = CATALOG[a.instrument_id]
                if not inst.is_wind:
                    continue
                by_player.setdefault(a.player_id, []).append(a.fragment)
            for frags in by_player.values():
                frags.sort(key=lambda f: f.start_s)
                for x, y in zip(frags, frags[1:]):
                    self.assertLessEqual(x.end_s, y.start_s + 1e-9)

    def _two_voice_score(self, inner_pitch: str):
        from chamber_orchestrator.score import (ROLE_INNER, ROLE_MELODY,
                                                 Score, make_line)
        bpm = 60
        mel = make_line("mel", ROLE_MELODY, "m", bpm, [(0, 2, "G5")])
        inner = make_line("inner", ROLE_INNER, "i", bpm,
                          [(0, 2, inner_pitch)])
        return Score("双音", bpm, [mel, inner])

    def test_string_double_stop_within_interval_allowed(self):
        # G5-G4 = 12 半音，恰在双音上限内
        res = Solver(self._two_voice_score("G4"),
                     [("p1", ["violin"])]).solve()
        self.assertTrue(res.feasible)

    def test_string_double_stop_beyond_interval_rejected(self):
        # G5-E4 = 15 半音，超过双音上限 12，且两音都在小提琴音域内
        res = Solver(self._two_voice_score("E4"),
                     [("p1", ["violin"])]).solve()
        self.assertFalse(res.feasible)
        self.assertTrue(any("双音音程" in r
                            for f in res.failures for r in f.reasons))

    def test_three_voices_exceed_string_polyphony(self):
        from chamber_orchestrator.score import (ROLE_BASS, ROLE_INNER,
                                                 ROLE_MELODY, Score, make_line)
        bpm = 60
        layers = [
            make_line("mel", ROLE_MELODY, "m", bpm, [(0, 2, "G5")]),
            make_line("i1", ROLE_INNER, "i1", bpm, [(0, 2, "E5")]),
            make_line("i2", ROLE_BASS, "i2", bpm, [(0, 2, "C5")]),
        ]
        res = Solver(Score("三音", bpm, layers),
                     [("p1", ["violin"])]).solve()
        self.assertFalse(res.feasible)
        self.assertTrue(any("复音上限" in r
                            for f in res.failures for r in f.reasons))


class TranspositionTests(unittest.TestCase):
    def test_written_pitch_offsets(self):
        trumpet = CATALOG["trumpet_bb"]
        horn = CATALOG["horn_f"]
        piccolo = CATALOG["piccolo"]
        self.assertEqual(trumpet.written_midi(60), 62)
        self.assertEqual(horn.written_midi(60), 67)
        self.assertEqual(piccolo.written_midi(84), 72)


class ChangeTimeTests(unittest.TestCase):
    def _solver(self):
        score = EXAMPLES["doubling_conflict"][0]()
        return Solver(score, ENSEMBLE_PRESETS["winds"])

    def test_fast_instrument_switch_is_rejected(self):
        from chamber_orchestrator.solver import Fragment, PlayerRuntime
        solver = self._solver()
        p = PlayerRuntime("p1", ["flute", "piccolo"])
        # 先在长笛上放一个 G4
        self.assertTrue(solver._place(
            p, "flute", Fragment("a", "mel", "melody", 0.0, 1.0, 67)))
        # 0.5s 后要用短笛吹 C8，间隔 < 换奏时间 2.5s
        frag = Fragment("b", "mel", "melody", 1.5, 1.0, 108)
        reasons = solver._violations(p, CATALOG["piccolo"], frag)
        self.assertTrue(any("换奏" in r for r in reasons))

    def test_switch_after_enough_gap_is_allowed(self):
        from chamber_orchestrator.solver import Fragment, PlayerRuntime
        solver = self._solver()
        p = PlayerRuntime("p1", ["flute", "piccolo"])
        self.assertTrue(solver._place(
            p, "flute", Fragment("a", "mel", "melody", 0.0, 1.0, 67)))
        # 3s 后切换，超过 2.5s 换奏时间
        frag = Fragment("b", "mel", "melody", 4.0, 1.0, 108)
        self.assertEqual(
            solver._violations(p, CATALOG["piccolo"], frag), [])


class ExportTests(unittest.TestCase):
    def test_orchestrate_produces_all_exports(self):
        score = EXAMPLES["long_tone"][0]()
        out = os.path.join(os.path.dirname(__file__), "tmp_out")
        payload = orchestrate(score, ENSEMBLE_PRESETS["winds"], out)
        self.assertTrue(payload["feasible"])
        for key in ("midi", "wav", "musicxml"):
            self.assertTrue(os.path.isfile(payload["exports"][key]))
        with wave.open(payload["exports"]["wav"]) as w:
            self.assertEqual(w.getframerate(), 44100)
            self.assertGreater(w.getnframes(), 0)
        # MIDI 头校验
        with open(payload["exports"]["midi"], "rb") as fh:
            self.assertEqual(fh.read(4), b"MThd")


class DeterminismTests(unittest.TestCase):
    def test_solver_is_deterministic_not_random(self):
        score = EXAMPLES["locked_register"][0]()
        r1 = Solver(score, ENSEMBLE_PRESETS["full"]).solve()
        r2 = Solver(score, ENSEMBLE_PRESETS["full"]).solve()
        key = lambda r: [(a.player_id, a.instrument_id, a.fragment.note_id,
                          round(a.fragment.start_s, 3)) for a in r.assignments]
        self.assertEqual(key(r1), key(r2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
