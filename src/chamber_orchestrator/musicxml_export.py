"""MusicXML 分谱导出（partwise，纯标准库 xml.etree）。

- 每件使用到的乐器导出为独立 <part>（即各乐器分谱）
- 移调乐器按 written_offset 写记谱音高，并写 <transpose>
- 换气拆分产生的真实休止写为 rest，拆分/移交在乐谱文字中注明
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .instruments import CATALOG
from .pitch import midi_to_note
from .solver import Assignment
from .score import ROLE_LABEL

_STEP_TO_DEGREE = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
_DEGREE_TO_STEP = ["C", "D", "E", "F", "G", "A", "B"]
# 移调 -> MusicXML transpose (chromatic, diatonic, octave-change)
TRANSPOSE_SPEC = {
    "clarinet_bb": (2, 1, 0),
    "trumpet_bb": (2, 1, 0),
    "horn_f": (7, 4, 0),
    "piccolo": (-12, 0, 1),       # 记谱比实际高八度
    "double_bass": (12, 0, -1),   # 记谱比实际低八度
}


def _midi_to_step_octave(midi: int) -> tuple[str, int, int]:
    name = midi_to_note(midi)
    step = name[0]
    alter = 0
    i = 1
    while i < len(name) and name[i] in "#b":
        alter += 1 if name[i] == "#" else -1
        i += 1
    octave = int(name[i:])
    return step, alter, octave


def _note_xml(parent: ET.Element, written_midi: int | None,
              duration_beats: float, divisions: int,
              is_rest: bool, text: str | None) -> None:
    note_el = ET.SubElement(parent, "note")
    if is_rest:
        ET.SubElement(note_el, "rest")
        if text:
            notation = ET.SubElement(note_el, "notations")
            technical = ET.SubElement(notation, "technical")
            ET.SubElement(technical, "breath-mark")
    else:
        pitch = ET.SubElement(note_el, "pitch")
        step, alter, octave = _midi_to_step_octave(written_midi)
        ET.SubElement(pitch, "step").text = step
        if alter:
            ET.SubElement(pitch, "alter").text = str(alter)
        ET.SubElement(pitch, "octave").text = str(octave)
    ET.SubElement(note_el, "duration").text = str(
        int(round(duration_beats * divisions)))
    ET.SubElement(note_el, "type").text = _duration_type(duration_beats)
    if text:
        lyric = ET.SubElement(note_el, "lyric")
        ET.SubElement(lyric, "syllabic").text = "single"
        ET.SubElement(lyric, "text").text = text


def _duration_type(beats: float) -> str:
    table = {4.0: "whole", 2.0: "half", 1.0: "quarter", 0.5: "eighth",
             0.25: "16th", 0.125: "32nd"}
    if beats in table:
        return table[beats]
    if beats >= 4:
        return "whole"
    if beats >= 2:
        return "half"
    if beats >= 1:
        return "quarter"
    if beats >= 0.5:
        return "eighth"
    return "16th"


def _divisions(bpm_quantize: int = 16) -> int:
    return bpm_quantize


def build_musicxml(assignments: list[Assignment], score,
                   title: str) -> bytes:
    divisions = 16  # 每拍 16 division，支持 16 分音符
    root = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = title
    ident = ET.SubElement(root, "identification")
    encoding = ET.SubElement(ident, "encoding")
    ET.SubElement(encoding, "software").text = "chamber-orchestrator"
    # 按乐器分组
    by_inst: dict[str, list[Assignment]] = {}
    for a in assignments:
        by_inst.setdefault(a.instrument_id, []).append(a)
    part_list = ET.SubElement(root, "part-list")
    part_ids = sorted(by_inst)
    for idx, inst_id in enumerate(part_ids):
        inst = CATALOG[inst_id]
        score_part = ET.SubElement(part_list, "score-part",
                                   id=f"P{idx + 1}")
        ET.SubElement(score_part, "part-name").text = inst.name
        ET.SubElement(score_part, "part-abbreviation").text = inst.name
        for clef_oct in (("G", 2), ("F", 4), ("C", 3)):
            if clef_oct[0] == inst.clef:
                clef = ET.SubElement(score_part, "clef")
                ET.SubElement(clef, "sign").text = inst.clef
                ET.SubElement(clef, "line").text = str(clef_oct[1])
    measures_total = max(
        1, int(-(-score.duration_beats // score.beats_per_bar)))
    for idx, inst_id in enumerate(part_ids):
        inst = CATALOG[inst_id]
        part_el = ET.SubElement(root, "part", id=f"P{idx + 1}")
        items = sorted(by_inst[inst_id],
                       key=lambda a: (a.fragment.start_s,
                                      0 if a.fragment.kind == "tone" else 1))
        for bar in range(measures_total):
            measure = ET.SubElement(part_el, "measure", number=str(bar + 1))
            if bar == 0:
                attrs = ET.SubElement(measure, "attributes")
                ET.SubElement(attrs, "divisions").text = str(divisions)
                key = ET.SubElement(attrs, "key")
                ET.SubElement(key, "fifths").text = "0"
                time = ET.SubElement(attrs, "time")
                ET.SubElement(time, "beats").text = str(score.beats_per_bar)
                ET.SubElement(time, "beat-type").text = "4"
                clef = ET.SubElement(attrs, "clef")
                ET.SubElement(clef, "sign").text = inst.clef
                line = {"G": 2, "F": 4, "C": 3}[inst.clef]
                ET.SubElement(clef, "line").text = str(line)
                if inst_id in TRANSPOSE_SPEC:
                    chrom, dia, octv = TRANSPOSE_SPEC[inst_id]
                    tr = ET.SubElement(attrs, "transpose")
                    ET.SubElement(tr, "diatonic").text = str(dia)
                    ET.SubElement(tr, "chromatic").text = str(chrom)
                    if octv:
                        ET.SubElement(tr, "octave-change").text = str(octv)
                sound = ET.SubElement(measure, "direction")
                ped = ET.SubElement(sound, "sound")
                ped.set("tempo", str(score.bpm))
            _fill_measure(measure, items, inst, bar, score, divisions)
    ET.indent(root, space="  ")
    return b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        root, encoding="utf-8")


def _fill_measure(measure: ET.Element, items: list[Assignment],
                  inst, bar: int, score, divisions: int) -> None:
    bar_start = bar * score.beats_per_bar
    bar_end = bar_start + score.beats_per_bar
    cursor = bar_start
    for a in items:
        f = a.fragment
        start_beat = f.start_s * score.bpm / 60.0
        end_beat = f.end_s * score.bpm / 60.0
        if end_beat <= bar_start or start_beat >= bar_end:
            continue
        # 与光标之间的空隙写休止
        if start_beat > cursor + 1e-9:
            gap = min(start_beat, bar_end) - cursor
            _note_xml(measure, None, gap, divisions, True, None)
            cursor += gap
        visible_start = max(start_beat, cursor)
        visible_end = min(end_beat, bar_end)
        dur = visible_end - visible_start
        if dur <= 0:
            continue
        if f.kind == "breath_rest":
            _note_xml(measure, None, dur, divisions, True, "换气")
        else:
            written = inst.written_midi(f.midi)
            text = None
            if f.split_total > 1 or f.handoff_from:
                text = (f"{'移交' if f.handoff_from else '拆分'} "
                        f"{ROLE_LABEL.get(f.role, f.role)}")
            _note_xml(measure, written, dur, divisions, False, text)
        cursor = visible_end
    # 小节剩余休止
    if cursor < bar_end - 1e-9:
        _note_xml(measure, None, bar_end - cursor, divisions, True, None)


def write_musicxml(path: str, assignments: list[Assignment],
                   score, title: str) -> None:
    with open(path, "wb") as fh:
        fh.write(build_musicxml(assignments, score, title))
