# app/schemas.py
"""
PCB 质检系统的 Pydantic 数据模型定义
====================================

【职责】
    为对外 API（FastAPI 响应、请求体）和内部跨模块传递提供强类型契约。
    与 YOLOv8 训练时的 6 类缺陷标签严格对齐，避免字符串硬编码导致的
    大小写/拼写漂移（如 "Missing_Hole" vs "missing_hole"）。

【设计原则】
    1. 用 Enum 而非裸字符串 —— 编译器/IDE 能提示拼写错误，且可迭代。
    2. Enum 值必须与 YOLOv8 训练时的 data.yaml `names` 字段完全一致。
    3. 与 data/ipc_standards.yaml 的 key 一一对应。
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class DefectType(str, Enum):
    """
    PCB 缺陷类型枚举 —— 与 YOLOv8 训练标签严格对齐。

    Why str + Enum？
        继承 str 让 Enum 成员可直接当字符串用（JSON 序列化、dict key），
        同时保留 Enum 的类型安全特性。

    6 类缺陷来源: PKU-Market-PCB 数据集 (Peking University)
        - 每类对应 data/ipc_standards.yaml 中的一条 IPC-A-610G 标准
        - 每类对应 YOLOv8 模型输出的一个 class_id (0~5)

    ⚠️ 修改本枚举时必须同步修改：
        1. data/ipc_standards.yaml  —— 知识库 key
        2. data/pcb_dataset.yaml    —— YOLO 训练配置的 names
        3. YOLOv8 训练标签的顺序    —— 否则模型输出的 class_id 会错位
    """
    MISSING_HOLE = "missing_hole"          # 漏孔：设计有孔未钻
    MOUSE_BITE = "mouse_bite"              # 鼠咬：板边半圆缺口
    OPEN_CIRCUIT = "open_circuit"          # 开路：导线断裂
    SHORT = "short"                        # 短路：相邻导体桥接
    SPURIOUS_COPPER = "spurious_copper"    # 残铜：不应有铜的地方有余铜
    SPUR = "spur"                          # 毛刺：导线边缘的尖刺