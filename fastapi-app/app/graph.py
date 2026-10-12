# app/graph.py
"""
PCB 质检多 Agent 系统 —— LangGraph 工作流定义
==============================================

【架构模式】
    Orchestrator-Worker:
        - Decision Agent 作为 Supervisor，统管路由
        - 4 个业务 Worker: Visual / DB Store / Standard / RootCause / Report
        - 所有 Worker 完成后通过回边 (add_edge) 强制回 Decision 汇报

【节点与状态流转】
    START → Decision ─┬→ Visual ────┐
                      ├→ DB Store ──┤
                      ├→ Standard ──┤→ Decision (回环)
                      ├→ RootCause ─┤
                      └→ Report ────┴→ END

【历史修复记录】
    R39-A: 企业级重构，适配新 State，修复 Reducer 翻倍 bug
    R40-fix: 修复 decision_agent 跳过 visual 节点的路由缺陷
    R41-stream: report_agent 改为 LLM 流式生成
    R42-async: 全链路异步化改造，防止阻塞 SSE 事件循环
    R43-fix: 修复 DB 写入失败导致的 GraphRecursionError 无限死循环
"""
import os
import hashlib
import logging
import asyncio
from datetime import datetime
from typing import Any

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.state import PCBQualityState
from app.feature_extractor import FeatureExtractor
from app.knowledge_base import kb
from app.root_cause_agent import analyze_root_cause
from app.db import upsert_defect_features

logger = logging.getLogger(__name__)


# =============================================================================
# 模块级单例
# =============================================================================
# ✅ 统一从环境变量读取 LLM 配置，移除硬编码假 Key（安全 + 可配置）
_llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    temperature=0.1,
    streaming=True,  # ✅ 必须开启，否则 report_agent 无法流式输出
    api_key=os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    base_url=os.getenv("LLM_BASE_URL", os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")),
)

# 特征提取器单例 —— 模型加载开销大，全进程共享一个实例
_extractor = FeatureExtractor()

# 展示层置信度阈值。
# 原值 0.30 导致 0.273 的有效缺陷被误过滤，调到 0.20 后召回率合理。
DISPLAY_CONF_THRESHOLD = 0.20


# =============================================================================
# 节点 1: Visual Agent
# =============================================================================

async def visual_agent(state: PCBQualityState) -> dict:
    """
    视觉检测 Agent —— 缺陷检测 + 特征提取 + 源头解耦。

    Args:
        state: 全局状态，需读取 image_path

    Returns:
        {
            "defects": [...],          # 纯业务数据（无 vector）
            "feature_vectors": [...],  # 独立向量列表
            "visual_done": True,       # 关键标志位
        }

    或异常时:
        {"error": str, "next_step": "END", "visual_done": True}

    ✅ R39-A 核心改动：
        源头分流 —— 分别构建业务列表 (defects) 和向量列表 (feature_vectors)，
        不再把 vector 塞在 defects[i] 里。好处:
        1. defects 可 JSON 序列化（无 numpy/tensor 污染）
        2. 各下游按需取字段，Milvus 直取 feature_vectors
        3. Reducer 策略不同（defects 追加 vs vectors 覆盖）得以分别优化
    """
    logger.info("👁️ [Visual] 正在检测缺陷...")
    image_path = state.get("image_path")
    if not image_path:
        return {
            "error": "缺少 image_path，无法检测",
            "next_step": "END",
            "visual_done": True,
        }

    # 诊断日志：确认 extractor 实例和模型状态（生产可降到 DEBUG）
    logger.info(
        f"   🔧 [DIAG] extractor type={type(_extractor).__name__}, "
        f"model_path="
        f"{getattr(_extractor, 'model', {}).get('path', 'N/A') if isinstance(getattr(_extractor, 'model', None), dict) else str(getattr(_extractor.model, 'ckpt_path', 'N/A'))}"
    )

    try:
        # ✅ R42修复：统一使用 full_defects，并使用 _extractor 实例
        # async_extract_from_image 内部把同步推理放入线程池，不阻塞事件循环
        full_defects = await _extractor.async_extract_from_image(image_path)

        # 诊断日志：确认返回值
        logger.info(f"   🔧 [DIAG] extract_from_image returned {len(full_defects)} defects")
        if not full_defects:
            logger.info("   ✅ 检测完成，未发现有效缺陷")
            return {"defects": [], "feature_vectors": [], "visual_done": True}

        # 双重保险过滤 + 保底
        # （extractor 内部已经过滤过一次，此处再过滤一次防止版本不一致）
        clean_defects = [
            d for d in full_defects if d["confidence"] >= DISPLAY_CONF_THRESHOLD
        ]
        if not clean_defects and full_defects:
            best = max(full_defects, key=lambda x: x["confidence"])
            clean_defects = [best]
            logger.warning(
                f"   ⚠️ 全部 < {DISPLAY_CONF_THRESHOLD}，"
                f"保留最高分保底: {best['confidence']:.4f}"
            )

        # ---------- ✅ 核心：源头分流 ----------
        defects_output = []
        vectors_output = []

        for fd in clean_defects:
            # 1. 处理向量（独立出来）
            vec = fd.get("vector", [])
            # 兼容 list / tuple / numpy.ndarray 三种可能类型
            if hasattr(vec, "tolist"):
                vector_list = vec.tolist()
            elif isinstance(vec, (list, tuple)):
                vector_list = list(vec)
            else:
                vector_list = []
            vectors_output.append(vector_list)

            # 2. 兼容 label / type 两种字段名（核心修复）
            # YOLO 原生输出是 "label"，但下游统一用 "type"
            label = fd.get("label") or fd.get("type") or "unknown"
            confidence = float(fd.get("confidence", 0.0))
            bbox = fd.get("bbox", [0, 0, 0, 0])

            # 3. 处理业务数据（不再包含 vector）
            # 生成确定性 defect_id，便于日志追踪和去重
            bbox_str = ",".join(f"{v:.1f}" for v in bbox)
            defect_id = hashlib.md5(
                f"{image_path}_{label}_{bbox_str}".encode()
            ).hexdigest()[:16]

            defects_output.append({
                "defect_id": defect_id,
                "type": label,
                "confidence": confidence,
                "bbox": bbox,
            })

        logger.info(f"   ✅ 检测完成，{len(defects_output)} 个缺陷待入库")
        return {
            "defects": defects_output,
            "feature_vectors": vectors_output,
            "visual_done": True,  # 关键标志位：告诉 Decision 已执行过
        }
    except Exception as e:
        logger.error(f"   ❌ 检测失败: {e}", exc_info=True)
        # ⚠️ 即使异常也要设 visual_done=True，否则 Decision 会无限重试
        return {
            "error": f"Visual Agent 异常: {str(e)}",
            "next_step": "END",
            "visual_done": True,
        }


# =============================================================================
# 节点 2: DB Store Agent
# =============================================================================

async def db_store_agent(state: PCBQualityState) -> dict:
    """
    数据库写入 Agent —— 把特征向量写入 Milvus Lite。

    Args:
        state: 需读取 defects / feature_vectors / image_path

    Returns:
        {"db_write_count": int}
            > 0: 写入成功条数
            = 0: 无缺陷或无数据（跳过）
            = -1: 写入异常（Decision 会跳过 DB 防死循环，R43-fix）

    ✅ R42-async:
        用 asyncio.to_thread 将同步 I/O 放入线程池，避免阻塞 SSE 事件循环。
    ✅ R43-fix:
        写入失败时返回 -1（而非 0），让 Decision 能识别"已尝试但失败"
        并跳过重试，防止无限循环。
    """
    logger.info("💾 [DB Store] 正在写入缺陷特征到 Milvus...")
    defects = state.get("defects", [])
    feature_vectors = state.get("feature_vectors", [])
    image_path = state.get("image_path", "")

    if not defects or not feature_vectors:
        logger.info("   ℹ️ 无缺陷或无向量数据，跳过写入")
        return {"db_write_count": 0}

    try:
        # ✅ 核心：放入线程池执行，不阻塞主事件循环
        count = await asyncio.to_thread(
            upsert_defect_features, defects, feature_vectors, image_path
        )
        logger.info(f"   ✅ 成功写入 {count} 条缺陷特征")
        return {"db_write_count": count}
    except Exception as e:
        # 不抛出异常（写入失败不应导致整个流程失败）
        logger.exception(f"   ⚠️ DB 写入失败（不阻塞主流程）: {e}")
        # 返回 -1 而非 0，让 Decision 能识别"已尝试但失败"
        return {"db_write_count": -1}


# =============================================================================
# 节点 3: Standard Agent
# =============================================================================

def standard_agent(state: PCBQualityState) -> dict:
    """
    标准匹配 Agent —— 为每个缺陷匹配 IPC 标准。

    Args:
        state: 需读取 defects

    Returns:
        {"matched_standards": [
            {"defect_type": str, "confidence": float, "bbox": list, "standard": dict},
            ...
        ]}

    注意：本节点是**同步函数**（O(1) 内存字典查询，无需异步）。
    """
    logger.info("📖 [Standard] 正在匹配IPC标准...")
    defects = state.get("defects", [])
    if not defects:
        logger.warning("   ⚠️ 未检测到缺陷，跳过标准查询。")
        return {"matched_standards": []}

    matched = []
    for defect in defects:
        defect_type = defect.get("type", "")
        if not defect_type:
            continue
        standard = kb.query_standard(defect_type)
        evidence = {
            "defect_type": defect_type,
            "confidence": defect.get("confidence", 0.0),
            "bbox": defect.get("bbox"),
            "standard": standard,
        }
        matched.append(evidence)
        logger.info(f"   ✅ {defect_type} → {standard.get('defect_name', 'Unknown')}")

    logger.info(f"   📋 共匹配 {len(matched)} 条标准记录。")
    return {"matched_standards": matched}


# =============================================================================
# 节点 4: RootCause Agent
# =============================================================================

async def rootcause_agent(state: PCBQualityState) -> dict:
    """
    根因分析 Agent —— 调用 DeepSeek LLM 推理根因。

    Args:
        state: 需读取 defects / matched_standards

    Returns:
        {"root_cause": str} 或异常时 {"error": str, "next_step": "END"}

    Note:
        analyze_root_cause 内部已处理异常并返回友好字符串，
        此处 try-except 是二重保护。
    """
    logger.info("🧠 [RootCause] 正在调用 LLM 分析根因...")
    try:
        defects = state.get("defects", [])
        matched_standards = state.get("matched_standards", [])
        # ✅ R42修复：增加 await（analyze_root_cause 已改异步）
        root_cause = await analyze_root_cause(defects, matched_standards)
        logger.info(f"   ✅ 根因分析完成 ({len(root_cause)} 字)")
        return {"root_cause": root_cause}
    except Exception as e:
        logger.error(f"   ❌ 根因分析失败: {e}")
        return {"error": f"RootCause Agent 异常: {str(e)}", "next_step": "END"}


# =============================================================================
# 节点 5: Report Agent
# =============================================================================

_REPORT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一位资深的 PCB（印制电路板）质量控制工程师。
请根据提供的检测数据、IPC 标准匹配结果和根因分析，撰写一份专业、结构化的 Markdown 质检报告。

要求：
1. 报告必须包含：检测概要、缺陷明细（Markdown表格）、根因分析、综合处置意见。
2. 缺陷明细表格必须严格使用以下格式：

| # | 类型 | 置信度 | IPC 等级 | 处置建议 |
|---|------|--------|----------|----------|
3. 语气专业客观，直接输出 Markdown 内容，不要包含任何解释性前言或后缀。
4. 在报告末尾加上生成时间和系统签名。"""
    ),
    (
        "human",
        """## 输入数据

### 1. 检测概要
- 图片路径: `{image_path}`
- 缺陷数量: {defect_count} 个
- 特征入库: {db_status}

### 2. 缺陷与标准匹配数据
{defects_and_standards}

### 3. 根因分析
{root_cause}

请根据以上信息生成完整的 Markdown 质检报告。"""
    ),
])
# Prompt 要点:
#   - 强制 Markdown 表格格式，便于前端渲染
#   - 明确"不要解释性前言"，避免 LLM 输出"好的，以下是我为您生成的报告..."
#   - 要求生成时间 + 系统签名，方便追溯


async def report_agent(state: PCBQualityState) -> dict:
    """
    报告生成 Agent —— 用 LLM 生成结构化 Markdown 质检报告。

    Args:
        state: 需读取 defects / matched_standards / root_cause / image_path / db_write_count

    Returns:
        {"final_report": str, "next_step": "END"}

    ✅ R41-stream: 使用 await _llm.ainvoke(prompt) 触发 on_chat_model_stream，
                   前端可实时逐 Token 显示报告生成过程。
    ✅ R42: ainvoke 异步调用不阻塞事件循环。
    """
    logger.info("📝 [Report] 正在调用 LLM 生成结构化质检报告...")
    try:
        defects = state.get("defects", [])
        matched_standards = state.get("matched_standards", [])
        root_cause = state.get("root_cause", "未生成根因分析")
        image_path = state.get("image_path", "未知路径")
        db_count = state.get("db_write_count", 0)

        # ✅ R43-fix：兼容 -1 (写入失败) 的状态显示
        if db_count > 0:
            db_status = f"{db_count} 条已写入 Milvus"
        elif db_count == -1:
            db_status = "⚠️ Milvus 写入失败（已降级处理）"
        else:
            db_status = "无缺陷，未写入"

        # ---------- 构建缺陷与标准匹配的文本描述 ----------
        defects_and_standards = ""
        if not defects:
            defects_and_standards = "未检测到任何缺陷。"
        else:
            for i, d in enumerate(defects, 1):
                # 按 defect_type 关联 standard（缺陷类型相同的取第一条匹配）
                std_info = next(
                    (
                        s["standard"]
                        for s in matched_standards
                        if s["defect_type"] == d.get("type")
                    ),
                    {},
                )
                ipc_class = std_info.get("ipc_reference", "N/A")
                disposition = std_info.get("rework_suggestion", "待判定")
                defects_and_standards += (
                    f"缺陷 {i}: 类型={d.get('type', 'Unknown')}, "
                    f"置信度={d.get('confidence', 0):.2%}, "
                    f"IPC等级={ipc_class}, 处置建议={disposition}\n"
                )

        # ---------- 构建 Prompt 并调用 LLM ----------
        prompt = _REPORT_PROMPT.invoke({
            "image_path": image_path,
            "defect_count": len(defects),
            "db_status": db_status,
            "defects_and_standards": defects_and_standards,
            "root_cause": root_cause,
        })

        # ✅ R42修复：使用 ainvoke 异步调用，完美触发 on_chat_model_stream
        response = await _llm.ainvoke(prompt)
        report = response.content

        logger.info(f"   ✅ LLM 报告生成完成 ({len(report)} 字符)")
        return {"final_report": report, "next_step": "END"}

    except Exception as e:
        logger.error(f"   ❌ 报告生成失败: {e}", exc_info=True)
        # Fallback: LLM 调用失败时退化为错误提示，保证接口有响应
        return {
            "final_report": f"# ❌ 报告生成失败\n\nLLM 调用异常: {str(e)}",
            "next_step": "END",
        }


# =============================================================================
# 节点 6: Decision Agent（Supervisor）
# =============================================================================

def decision_agent(state: PCBQualityState) -> dict:
    """
    路由决策 Agent —— 根据当前 State 决定下一步走向。

    Args:
        state: 全局状态

    Returns:
        {"next_step": str}
            取值: "visual" | "db_store" | "standard" | "rootcause" | "report" | "END"

    路由优先级（短路顺序至关重要）:
        1. 有 error → END (最高优先级，立即终止)
        2. visual_done 为假 → visual
        3. 有 defects 且 db_write_count == 0 → db_store
        4. matched_standards 不在 state → standard
        5. root_cause 不在 state → rootcause
        6. final_report 不在 state → report
        7. 全部完成 → END

    ✅ R40-fix: 使用 visual_done 标志判断，而非 `defects not in state`。
    ✅ R43-fix: 识别 db_write_count == -1，跳过 DB 节点防止死循环。
    """
    logger.info("🎯 [Decision] 正在决策下一步...")

    # ---------- 1. 异常短路 ----------
    if state.get("error"):
        logger.error(f"   ⛔ 检测到错误，终止流程: {state['error']}")
        return {"next_step": "END"}

    # ---------- 2. Visual 未执行 ----------
    # ✅ R40-fix：用 visual_done 标志判断是否已执行检测，
    # 而非 defects 是否为空（空列表会被误判为"未执行"导致死循环）
    if not state.get("visual_done"):
        return {"next_step": "visual"}

    # ---------- 3. DB 路由防死循环（R43 核心修复）----------
    db_count = state.get("db_write_count", 0)
    if state.get("defects"):
        if db_count == 0:
            # 尚未写入过，尝试写入
            return {"next_step": "db_store"}
        elif db_count == -1:
            # 写入失败 → 记录警告并**直接跳过** db_store，进入 standard 节点
            # ⚠️ 关键：若不跳过，Decision 会反复返回 db_store 造成死循环
            logger.warning("   ⚠️ 检测到 DB 写入失败 (-1)，跳过 DB 节点继续后续流程")
            # 注意：这里不 return，直接往下走
        # 如果 db_count > 0，说明写入成功，也继续往下走

    # ---------- 4. 标准匹配未执行 ----------
    # 用 "in state" 而非 "值非空" —— 空列表也是已执行的有效结果
    if "matched_standards" not in state:
        return {"next_step": "standard"}

    # ---------- 5. 根因分析未执行 ----------
    if "root_cause" not in state:
        return {"next_step": "rootcause"}

    # ---------- 6. 报告未生成 ----------
    if "final_report" not in state:
        return {"next_step": "report"}

    # ---------- 7. 全部完成 ----------
    return {"next_step": "END"}


def route_decision(state: PCBQualityState) -> str:
    """
    条件边路由函数 —— 从 State 中提取 next_step，并做白名单校验。

    Args:
        state: 全局状态

    Returns:
        合法节点名（"visual"|"db_store"|"standard"|"rootcause"|"report"|"END"）

    Why 白名单校验？
        防止 LLM 幻觉或 bug 导致 next_step 被设置成非法值，
        让 LangGraph 抛 KeyError。非法值统一降级到 END，保证安全退出。
    """
    step = state.get("next_step", "END")
    valid_routes = {"visual", "db_store", "standard", "rootcause", "report", "END"}
    return step if step in valid_routes else "END"


# =============================================================================
# 图构建
# =============================================================================

def build_graph():
    """
    构建 LangGraph StateGraph。

    Returns:
        编译后的 CompiledGraph，可直接调用 .invoke() / .astream_events()

    【图结构】
        节点: decision, visual, db_store, standard, rootcause, report
        入口: decision
        条件边: decision → {visual|db_store|standard|rootcause|report|END}
        回边:   所有 Worker → decision（除 report → END）

    【为什么所有 Worker 都回 decision？】
        Orchestrator-Worker 模式的核心 —— Worker 只负责执行，不负责决策。
        执行完后回到 Supervisor 汇报，由 Supervisor 根据最新状态决定下一步。
        这样新增 Worker 无需修改现有节点，符合开闭原则。
    """
    builder = StateGraph(PCBQualityState)

    # ---------- 添加节点 ----------
    # LangGraph 完美支持 async def 节点，会自动 await
    builder.add_node("decision", decision_agent)
    builder.add_node("visual", visual_agent)
    builder.add_node("db_store", db_store_agent)
    builder.add_node("standard", standard_agent)
    builder.add_node("rootcause", rootcause_agent)
    builder.add_node("report", report_agent)

    # ---------- 设置入口 ----------
    builder.set_entry_point("decision")

    # ---------- 添加条件边（路由）----------
    builder.add_conditional_edges(
        "decision",
        route_decision,
        {
            "visual": "visual",
            "db_store": "db_store",
            "standard": "standard",
            "rootcause": "rootcause",
            "report": "report",
            "END": END,
        },
    )

    # ---------- 添加回边（Worker 完成后回 Decision）----------
    builder.add_edge("visual", "decision")
    builder.add_edge("db_store", "decision")
    builder.add_edge("standard", "decision")
    builder.add_edge("rootcause", "decision")
    # report 是最后一个节点，直接到 END（无需回 Decision）
    builder.add_edge("report", END)

    return builder.compile()


# =============================================================================
# 模块级编译实例
# =============================================================================
# 导入本模块时立即编译图（一次性开销，通常 <100ms）
# 之后 main.py 直接 `from app.graph import graph` 使用
graph = build_graph()