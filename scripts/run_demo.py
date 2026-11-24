#!/usr/bin/env python3
"""命令行批量演示：对全部谱例跑配器并导出分谱/合奏到 output/。

用法：python3 scripts/run_demo.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chamber_orchestrator import ENSEMBLE_PRESETS
from chamber_orchestrator.examples import EXAMPLES
from chamber_orchestrator.orchestrate import write_report_json, orchestrate

# 每个谱例使用的编制（range_conflict 先用小编制演示无解，再用完整编制求解）
PLAN = [
    ("long_tone", "winds"),
    ("fast_alternation", "winds"),
    ("range_conflict", "small"),       # 预期无解
    ("range_conflict", "full"),        # 对照：加入低音乐器后可解
    ("locked_register", "full"),
    ("doubling_conflict", "winds"),
]
# 边界谱例使用自带编制（两把长笛 -> 合奏无解；四把长笛 -> t=10s 交接可解）
BOUNDARY_PLAN = [
    ("capability_two_flutes_long", [("p1", ["flute"]), ("p2", ["flute"])]),
    ("capability_four_flutes_long", [("p1", ["flute"]), ("p2", ["flute"]),
                                     ("p3", ["flute"]), ("p4", ["flute"])]),
]


def main() -> int:
    root = os.path.join(os.path.dirname(__file__), "..", "output")
    rows = [(key, ENSEMBLE_PRESETS[preset], preset) for key, preset in PLAN]
    rows += [(key, ens, "boundary") for key, ens in BOUNDARY_PLAN]
    for key, ensemble, tag_preset in rows:
        score = EXAMPLES[key][0]()
        tag = f"{key}__{tag_preset}"
        out_dir = os.path.join(root, tag)
        os.makedirs(out_dir, exist_ok=True)
        payload = orchestrate(score, ensemble, out_dir)
        write_report_json(os.path.join(out_dir, "report.json"), payload)
        status = "可行" if payload["feasible"] else "无解"
        verify = payload.get("verification")
        vtag = ""
        if verify:
            vtag = f" 验收={'通过' if verify['passes'] else '未通过'}" \
                   f" 覆盖={verify['voices_covered']}/{verify['voices_expected']}"
        print(f"[{status}] {tag:34s} 节点={payload['nodes']:6d} "
              f"评分={payload['score']:7.3f}{vtag} 导出={out_dir}")
        if not payload["feasible"]:
            for line in payload["explanation"].splitlines()[:4]:
                print("    " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
