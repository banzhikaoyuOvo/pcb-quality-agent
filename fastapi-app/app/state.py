# app/state.py
"""
PCB质检多Agent系统 - 全局共享状态定义
✅ R39-A：企业级 State 重构 —— Reducer 防覆盖 + 向量字段独立 + Python 3.13 原生类型
✅ R40-fix：新增 visual_done 标志，修复 decision_agent 跳过 visual 节点的路由缺陷
"""
from typing import TypedDict, Annotated, Optional
import operator


def _vector_reducer(existing: list[list[float]], new: list[list[float]]) -> list[list[float]]:
    """
    自定义向量 Reducer：
    - 如果 new 非空，用 new 替换 existing（每轮检测产生新一批向量）
    - 如果 new 为空，保留 existing（防止中间节点误清空）
    
    为什么不用 operator.add？
    因为 operator.add 会把多轮循环的向量全部拼接，导致列表无限膨胀。
    我们需要的是"最新一轮的完整向量集"，而非历史累积。
    """
    return new if new else existing


class PCBQualityState(TypedDict):
    """
    LangGraph 全局状态：所有 Agent 节点读写同一个对象
    
    ✅ R39-A 变更说明：
    1. defects 增加 operator.add Reducer → 多轮循环追加而非覆盖
    2. feature_vectors 独立为顶级字段 → 与业务数据解耦，Milvus 直取
    3. 全部使用 Python 3.13 原生 list 替代 typing.List
    4. 每个字段标注 [写入者] → [读取者] 数据流契约
    
    ✅ R40-fix 变更说明：
    5. 新增 visual_done 标志 → 解决初始 state 无 defects 时 decision_agent 跳过 visual 的问题
    """
    
    # === 输入层 ===
    image_path: str                          # [API] → [Visual Agent] 待检图片路径
    
    # === 检测层（带 Reducer 防覆盖）===
    defects: Annotated[list[dict], operator.add]  
    # [Visual Agent] → [Standard/Decision/Report]
    # 格式: [{"type": str, "confidence": float, "bbox": list[int]}]
    # ⚠️ vector 不再存于此，已独立为 feature_vectors
    
    feature_vectors: Annotated[list[list[float]], _vector_reducer]
    # [Visual Agent] → [DB Agent/Milvus]
    # 格式: [[0.12, -0.34, ...], ...] 每个元素为 256 维 L2 归一化向量
    # 与 defects 按索引一一对应
    
    visual_done: bool                        # [Visual Agent] → [Decision Agent] 标记视觉检测是否已执行
    # ✅ R40-fix：替代原先用 "defects not in state" 判断的逻辑
    # 初始 state 无此字段 → state.get("visual_done") 返回 None → falsy → 路由到 visual
    # visual_agent 完成后设为 True → 后续决策不再重复进入 visual
    
    # === 分析层 ===
    matched_standards: list[str]             # [Standard Agent] → [Report/Decision]
    root_cause: str                          # [RootCause Agent] → [Report]
    
    # === 决策层 ===
    next_step: str                           # [Decision Agent] → [Router]
    
    # === 输出层 ===
    final_report: str                        # [Report Agent] → [API Response]
    db_write_count: int                      # [DB Agent] → [Report Agent] Milvus写入成功条数
    
    # === 异常层 ===
    error: Optional[str]                     # [任意节点] → [API Response] 错误信息