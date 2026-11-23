"""配器门面：求解 -> 可追踪结果 -> 分谱/合奏导出，含模型简化范围声明。"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from .audio import write_ensemble_wav, write_player_wavs
from .instruments import CATALOG, ENSEMBLE_PRESETS
from .midi_export import write_midi
from .musicxml_export import write_musicxml
from .score import ROLE_LABEL
from .solver import Solver

SIMPLIFICATION_SCOPE = [
    "音域为静态发声音域，未建模动态音准、极端力度下的边缘音可靠性。",
    "呼吸模型用单音最大持续、乐句累计预算、固定换气/恢复时长近似，"
    "未区分乐句结构与个人肺活量差异。",
    "弦乐只建模换把大跳间隔与双音音程上限，未建模把位、空弦、揉弦与弓法细节。",
    "复音仅约束同时发声数与双音音程，未做和弦排列/省略音的和声学校验。",
    "换奏时间按固定秒数建模，未考虑舞台空间与乐器摆放调度。",
    "音频为加法合成近似音色，非真实采样；力度、 articulation 标记与微分音未建模。",
    "移调乐器在 MusicXML 写记谱音高与 <transpose>，MIDI/WAV 一律为实际发声音高。",
    "不做整体八度移位：所有分配保持材料实际音高，锁定音区为硬约束。",
]


def orchestrate(score, ensemble: list[tuple[str, list[str]]],
                out_dir: str | None = None,
                node_budget: int = 12_000) -> dict:
    solver = Solver(score, ensemble, node_budget=node_budget)
    result = solver.solve()
    payload = {
        "title": score.title,
        "bpm": score.bpm,
        "duration_s": round(score.duration_s, 3),
        "feasible": result.feasible,
        "score": round(result.score, 4),
        "nodes": result.nodes,
        "explanation": result.explanation,
        "split_notes": result.split_notes,
        "role_fallbacks": result.role_fallbacks,
        "trace": result.trace,
        "player_summary": result.player_summary,
        "failures": [asdict(f) for f in result.failures],
        "simplification_scope": SIMPLIFICATION_SCOPE,
        "ensemble": [
            {"player": pid,
             "instruments": [{"id": iid, "name": CATALOG[iid].name,
                              "family": CATALOG[iid].family,
                              "range": [CATALOG[iid].low, CATALOG[iid].high],
                              "written_offset": CATALOG[iid].written_offset}
                             for iid in ids]}
            for pid, ids in ensemble
        ],
        "exports": {},
    }
    if result.feasible and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        midi_path = os.path.join(out_dir, "ensemble.mid")
        wav_path = os.path.join(out_dir, "ensemble.wav")
        xml_path = os.path.join(out_dir, "parts.musicxml")
        write_midi(midi_path, result.assignments, score.bpm)
        write_ensemble_wav(wav_path, result.assignments)
        parts_dir = os.path.join(out_dir, "player_wavs")
        os.makedirs(parts_dir, exist_ok=True)
        player_wavs = write_player_wavs(parts_dir, result.assignments)
        write_musicxml(xml_path, result.assignments, score, score.title)
        payload["exports"] = {
            "midi": midi_path, "wav": wav_path,
            "musicxml": xml_path, "player_wavs": player_wavs,
        }
        payload["per_instrument"] = _per_instrument(result)
    return payload


def _per_instrument(result) -> list[dict]:
    grouped: dict[str, dict] = {}
    for a in result.assignments:
        f = a.fragment
        inst = CATALOG[a.instrument_id]
        rec = grouped.setdefault(a.instrument_id, {
            "instrument": inst.id, "name": inst.name,
            "family": inst.family, "player": a.player_id,
            "clef": inst.clef,
            "written_offset": inst.written_offset,
            "events": [],
        })
        rec["events"].append({
            "note_id": f.note_id, "role": f.role,
            "role_label": ROLE_LABEL.get(f.role, f.role),
            "sounding_midi": f.midi if f.kind == "tone" else None,
            "written_midi": (inst.written_midi(f.midi)
                             if f.kind == "tone" else None),
            "start_s": round(f.start_s, 3),
            "duration_s": round(f.duration_s, 3),
            "kind": f.kind, "action": _action_of(f),
            "reason": f.reason,
        })
    for rec in grouped.values():
        rec["events"].sort(key=lambda e: (e["start_s"], e["kind"]))
    return sorted(grouped.values(), key=lambda r: r["name"])


def _action_of(f) -> str:
    if f.kind == "breath_rest":
        return "breath-rest"
    if f.handoff_from:
        return "handoff-in"
    if f.split_total > 1:
        return "split"
    return "direct"


def write_report_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
