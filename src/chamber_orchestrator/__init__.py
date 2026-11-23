"""室内配器辅助器：约束求解 + 分谱/合奏导出 + 试听界面。"""
from .solver import Assignment, Fragment, SolveResult, Solver
from .instruments import CATALOG, ENSEMBLE_PRESETS, Instrument
from .score import Score, MaterialLayer, Note, ROLES, ROLE_LABEL
from .examples import EXAMPLES

__all__ = [
    "Assignment", "Fragment", "SolveResult", "Solver",
    "CATALOG", "ENSEMBLE_PRESETS", "Instrument",
    "Score", "MaterialLayer", "Note", "ROLES", "ROLE_LABEL", "EXAMPLES",
]
