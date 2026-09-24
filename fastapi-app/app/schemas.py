from enum import Enum
from typing import Optional
from pydantic import BaseModel


class DefectType(str, Enum):
    """PCB 缺陷类型枚举 - 与 YOLOv8 训练标签严格对齐"""
    MISSING_HOLE = "missing_hole"
    MOUSE_BITE = "mouse_bite"
    OPEN_CIRCUIT = "open_circuit"
    SHORT = "short"
    SPURIOUS_COPPER = "spurious_copper"
    SPUR = "spur"


