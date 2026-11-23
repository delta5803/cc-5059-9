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


def main() -> int:
    root = os.path.join(os.path.dirname(__file__), "..", "output")
    for key, preset in PLAN:
        score = EXAMPLES[key][0]()
        tag = f"{key}__{preset}"
        out_dir = os.path.join(root, tag)
        os.makedirs(out_dir, exist_ok=True)
        payload = orchestrate(score, ENSEMBLE_PRESETS[preset], out_dir)
        write_report_json(os.path.join(out_dir, "report.json"), payload)
        status = "可行" if payload["feasible"] else "无解"
        print(f"[{status}] {tag:28s} 节点={payload['nodes']:6d} "
              f"评分={payload['score']:7.3f} 导出={out_dir}")
        if not payload["feasible"]:
            for line in payload["explanation"].splitlines()[:4]:
                print("    " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
