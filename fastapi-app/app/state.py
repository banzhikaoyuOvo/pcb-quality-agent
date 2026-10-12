# app/state.py
"""
PCB 质检多 Agent 系统 —— 全局共享状态定义（Single Source of Truth）
================================================================

【设计目标】
    定义 LangGraph 工作流中所有 Agent 节点共享的唯一状态结构（TypedDict）。
    所有 Agent 读写的都是同一个 State 对象，LangGraph 负责在节点间合并
    各节点返回的"增量字典"（dict）到主 State。

【为什么用 TypedDict 而非 Pydantic BaseModel？】
    - LangGraph StateGraph 原生支持 TypedDict，字段即键，合并逻辑最简单。
    - Pydantic 会做强制类型校验，会拖慢流式 Token 场景下的状态更新。
    - 校验职责交给节点内部的业务代码（如 float() 显式转换）更灵活。

【Reducer 机制 —— 本文件最核心的设计】
    LangGraph 中，若字段类型用 Annotated[T, reducer_fn] 声明，
    则多个节点返回同一字段时，会用 reducer_fn(existing, new) 合并，
    而非简单覆盖。这是防止"多轮回环时字段被覆盖/无限膨胀"的关键。

【历史修复记录】
    - R39-A: defects 增加 operator.add reducer（防多轮覆盖）
              feature_vectors 独立为顶级字段（与业务数据解耦）
    - R40-fix: 新增 visual_done 标志（修复 decision_agent 跳过 visual 的 bug）
"""
from typing import TypedDict, Annotated, Optional
import operator


# =============================================================================
# 自定义 Reducer
# =============================================================================

def _vector_reducer(
    existing: list[list[float]],
    new: list[list[float]],
) -> list[list[float]]:
    """
    特征向量列表的合并策略：**"新覆盖旧"而非"追加"**。

    Args:
        existing: State 中已有的向量列表（上一轮 Visual Agent 产出的）
        new:      当前节点返回的新向量列表

    Returns:
        最新一轮的完整向量集

    Why not operator.add？
        operator.add 会把多轮回环产生的向量全部拼接：
            [vec_a1, vec_a2] + [vec_b1, vec_b2] + ... → 无限膨胀
        但业务语义上，每轮检测是"重新检测"，只需要最新一批向量。
        所以用"new 非空则替换，new 为空则保留 existing"的策略，
        既防止无限膨胀，又防止中间节点（如 db_store）误清空。

    Note:
        空 list 的判空 `if new` 是刻意的 —— 空向量列表语义上代表
        "本轮无有效数据"，此时应保留上一轮结果而非覆盖为空。
    """
    return new if new else existing


# =============================================================================
# 全局状态定义
# =============================================================================

class PCBQualityState(TypedDict):
    """
    LangGraph 全局状态：所有 Agent 节点读写同一个对象。

    数据流契约（每个字段标注 [写入者] → [读取者]）：

    ┌────────────────── 输入层 ──────────────────┐
    │ image_path:      [API] → [Visual]          │
    └────────────────────────────────────────────┘

    ┌────────────────── 检测层（带 Reducer）────────┐
    │ defects:         [Visual] → [DB/Std/Rpt/Dec]  │
    │ feature_vectors: [Visual] → [DB/Milvus]       │
    │ visual_done:     [Visual] → [Decision]        │
    └────────────────────────────────────────────┘

    ┌────────────────── 分析层 ──────────────────┐
    │ matched_standards: [Standard] → [Rpt/Dec]  │
    │ root_cause:        [RootCause] → [Rpt]     │
    └────────────────────────────────────────────┘

    ┌────────────────── 决策层 ──────────────────┐
    │ next_step:       [Decision] → [Router]     │
    └────────────────────────────────────────────┘

    ┌────────────────── 输出层 ──────────────────┐
    │ final_report:    [Report] → [API Response] │
    │ db_write_count:  [DB] → [Report]           │
    └────────────────────────────────────────────┘

    ┌────────────────── 异常层 ──────────────────┐
    │ error:           [任意节点] → [API Response]│
    └────────────────────────────────────────────┘
    """

    # === 输入层 ===
    image_path: str
    # [API] → [Visual Agent] 待检图片的绝对路径。
    # 上传文件由 main.py 保存后写入，Visual Agent 读取并送入 YOLO。

    # === 检测层（带 Reducer 防覆盖）===
    defects: Annotated[list[dict], operator.add]
    # [Visual Agent] → [Standard/Decision/Report]
    # 格式: [{"defect_id": str, "type": str, "confidence": float, "bbox": list[int]}]
    #
    # Why operator.add？
    #   若用默认覆盖语义，多轮回环（Decision 回环到 Standard 等）时，
    #   后续节点的空 dict 返回会把 defects 覆盖为空。
    #   用 operator.add 保证 defects 只会"追加"不会"消失"。
    #
    # ⚠️ 注意: vector 不存于此！已独立为 feature_vectors 字段（R39-A 解耦）。

    feature_vectors: Annotated[list[list[float]], _vector_reducer]
    # [Visual Agent] → [DB Agent / Milvus]
    # 格式: [[0.12, -0.34, ...], ...]  每个元素为 256 维 L2 归一化向量。
    # 与 defects 按**索引一一对应**：feature_vectors[i] 对应 defects[i]。
    # 使用自定义 _vector_reducer（新覆盖旧）而非 operator.add（防膨胀）。

    visual_done: bool
    # [Visual Agent] → [Decision Agent] 标记"视觉检测是否已执行"。
    #
    # ✅ R40-fix 关键修复：
    #   原逻辑用 "defects not in state" 判断是否已执行检测，但存在致命 bug：
    #   初始 state 无 defects 字段 → 判断为"未执行"→ 进入 visual →
    #   visual 返回空 defects: [] → 下一轮 decision 又判断 "defects not in state"
    #   → 再次进入 visual → **无限死循环 (GraphRecursionError)**。
    #   加入 visual_done 标志位后：
    #     初始 state 无此字段 → get() 返回 None → falsy → 路由到 visual
    #     visual 完成后设为 True → 后续决策不再重复进入 visual
    #   ⚠️ 这是本字段存在的唯一理由，切勿删除。

    # === 分析层 ===
    matched_standards: list[str]
    # [Standard Agent] → [Report/Decision]
    # 格式: [{"defect_type": str, "confidence": float, "bbox": list, "standard": dict}]
    # 每项对应一个缺陷的 IPC 标准匹配结果。

    root_cause: str
    # [RootCause Agent] → [Report]
    # DeepSeek LLM 生成的 200-400 字中文根因分析文本。

    # === 决策层 ===
    next_step: str
    # [Decision Agent] → [Router]
    # 取值: "visual" | "db_store" | "standard" | "rootcause" | "report" | "END"
    # Router (route_decision) 读取此字段决定下一个节点。

    # === 输出层 ===
    final_report: str
    # [Report Agent] → [API Response]
    # LLM 生成的 Markdown 格式结构化质检报告。

    db_write_count: int
    # [DB Agent] → [Report Agent]
    # Milvus 写入成功条数，也是 Decision 路由的哨兵值：
    #   > 0: 写入成功，可继续后续流程
    #   = 0: 尚未写入（Decision 会路由到 db_store）
    #   = -1: 写入失败（Decision 会跳过 db_store 防死循环，R43-fix）

    # === 异常层 ===
    error: Optional[str]
    # [任意节点] → [API Response]
    # 任一节点捕获到不可恢复异常时写入，Decision Agent 检测到后短路到 END。