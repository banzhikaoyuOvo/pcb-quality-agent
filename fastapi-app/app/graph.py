# app/graph.py
"""
PCB质检多Agent系统 - LangGraph 工作流定义
✅ R39-A：企业级重构 —— 适配新State解耦设计 + 修复Reducer翻倍Bug + 统一配置读取
✅ R40-fix：修复 decision_agent 跳过 visual 节点的路由缺陷
✅ R41-stream：report_agent 改为 LLM 流式生成，支持 on_chat_model_stream 事件
✅ R42-async：全链路异步化改造 (visual/rootcause/report)，防止阻塞 SSE 事件循环
✅ R43-fix：修复 DB 写入失败导致的 GraphRecursionError 无限死循环 Bug
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

# ✅ 统一从环境变量读取 LLM 配置，移除硬编码假 Key
_llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    temperature=0.1,
    streaming=True,  # ✅ 确保开启流式输出
    api_key=os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    base_url=os.getenv("LLM_BASE_URL", os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1"))
)

_extractor = FeatureExtractor()
DISPLAY_CONF_THRESHOLD = 0.20  # 原值 0.30，导致 0.273 的缺陷被过滤


async def visual_agent(state: PCBQualityState) -> dict:
    """
    ✅ R39-A：源头解耦，直接输出独立的 defects 和 feature_vectors
    职责：检测 + 特征提取 + 过滤 + 保底 → 输出干净的业务数据和向量数据
    """
    logger.info("👁️ [Visual] 正在检测缺陷...")
    image_path = state.get("image_path")
    if not image_path:
        return {"error": "缺少 image_path，无法检测", "next_step": "END", "visual_done": True}

    # ✅ 诊断：确认 extractor 实例和模型状态
    logger.info(f"   🔧 [DIAG] extractor type={type(_extractor).__name__}, "
                f"model_path={getattr(_extractor, 'model', {}).get('path', 'N/A') if isinstance(getattr(_extractor, 'model', None), dict) else str(getattr(_extractor.model, 'ckpt_path', 'N/A'))}")

    try:
        # ✅ R42修复：统一使用 full_defects，并使用 _extractor 实例
        full_defects = await _extractor.async_extract_from_image(image_path)
        
        # ✅ 诊断：确认返回值
        logger.info(f"   🔧 [DIAG] extract_from_image returned {len(full_defects)} defects")
        if not full_defects:
            logger.info("   ✅ 检测完成，未发现有效缺陷")
            return {"defects": [], "feature_vectors": [], "visual_done": True}

        # 双重保险过滤 + 保底
        clean_defects = [d for d in full_defects if d["confidence"] >= DISPLAY_CONF_THRESHOLD]
        if not clean_defects and full_defects:
            best = max(full_defects, key=lambda x: x["confidence"])
            clean_defects = [best]
            logger.warning(f"   ⚠️ 全部 < {DISPLAY_CONF_THRESHOLD}，保留最高分保底: {best['confidence']:.4f}")

        # ✅ 核心改动：源头分流，分别构建业务列表和向量列表
        defects_output = []
        vectors_output = []

        for fd in clean_defects:
            # 1. 处理向量（独立出来）
            vec = fd.get("vector", [])
            if hasattr(vec, "tolist"):
                vector_list = vec.tolist()
            elif isinstance(vec, (list, tuple)):
                vector_list = list(vec)
            else:
                vector_list = []
            vectors_output.append(vector_list)

            # 2. 兼容 label / type 两种字段名（核心修复）
            label = fd.get("label") or fd.get("type") or "unknown"
            confidence = float(fd.get("confidence", 0.0))
            bbox = fd.get("bbox", [0, 0, 0, 0])

            # 3. 处理业务数据（不再包含 vector）
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
            "visual_done": True,
        }
    except Exception as e:
        logger.error(f"   ❌ 检测失败: {e}", exc_info=True)
        return {"error": f"Visual Agent 异常: {str(e)}", "next_step": "END", "visual_done": True}


async def db_store_agent(state: PCBQualityState) -> dict:
    """
    ✅ R42-async：Milvus 写入异步化
    使用 asyncio.to_thread 将同步 I/O 放入线程池，避免阻塞 SSE 事件循环
    ✅ R43-fix：写入失败时返回 -1，防止 decision_agent 无限重试
    """
    logger.info("💾 [DB Store] 正在写入缺陷特征到 Milvus...")
    defects = state.get("defects", [])
    feature_vectors = state.get("feature_vectors", [])
    image_path = state.get("image_path", "")

    if not defects or not feature_vectors:
        logger.info("   ℹ️ 无缺陷或无向量数据，跳过写入")
        return {"db_write_count": 0}

    try:
        # ✅ 核心改动：放入线程池执行，不阻塞主事件循环
        count = await asyncio.to_thread(
            upsert_defect_features, defects, feature_vectors, image_path
        )
        logger.info(f"   ✅ 成功写入 {count} 条缺陷特征")
        return {"db_write_count": count}
    except Exception as e:
        logger.exception(f"   ⚠️ DB 写入失败（不阻塞主流程）: {e}")
        return {"db_write_count": -1}


def standard_agent(state: PCBQualityState) -> dict:
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
            "standard": standard
        }
        matched.append(evidence)
        logger.info(f"   ✅ {defect_type} → {standard.get('defect_name', 'Unknown')}")

    logger.info(f"   📋 共匹配 {len(matched)} 条标准记录。")
    return {"matched_standards": matched}


async def rootcause_agent(state: PCBQualityState) -> dict:
    logger.info("🧠 [RootCause] 正在调用 LLM 分析根因...")
    try:
        defects = state.get("defects", [])
        matched_standards = state.get("matched_standards", [])
        # ✅ R42修复：增加 await
        root_cause = await analyze_root_cause(defects, matched_standards)
        logger.info(f"   ✅ 根因分析完成 ({len(root_cause)} 字)")
        return {"root_cause": root_cause}
    except Exception as e:
        logger.error(f"   ❌ 根因分析失败: {e}")
        return {"error": f"RootCause Agent 异常: {str(e)}", "next_step": "END"}


# ==========================================
# ✅ R41-stream：Report Agent 流式生成改造
# ==========================================

_REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深的 PCB（印制电路板）质量控制工程师。
请根据提供的检测数据、IPC 标准匹配结果和根因分析，撰写一份专业、结构化的 Markdown 质检报告。

要求：
1. 报告必须包含：检测概要、缺陷明细（Markdown表格）、根因分析、综合处置意见。
2. 缺陷明细表格必须严格使用以下格式：

| # | 类型 | 置信度 | IPC 等级 | 处置建议 |
|---|------|--------|----------|----------|
3. 语气专业客观，直接输出 Markdown 内容，不要包含任何解释性前言或后缀。
4. 在报告末尾加上生成时间和系统签名。"""),
    ("human", """## 输入数据

### 1. 检测概要
- 图片路径: `{image_path}`
- 缺陷数量: {defect_count} 个
- 特征入库: {db_status}

### 2. 缺陷与标准匹配数据
{defects_and_standards}

### 3. 根因分析
{root_cause}

请根据以上信息生成完整的 Markdown 质检报告。""")
])

async def report_agent(state: PCBQualityState) -> dict:
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

        # 构建缺陷与标准匹配的文本描述
        defects_and_standards = ""
        if not defects:
            defects_and_standards = "未检测到任何缺陷。"
        else:
            for i, d in enumerate(defects, 1):
                std_info = next(
                    (s["standard"] for s in matched_standards if s["defect_type"] == d.get("type")),
                    {}
                )
                ipc_class = std_info.get("ipc_reference", "N/A")
                disposition = std_info.get("rework_suggestion", "待判定")
                defects_and_standards += (
                    f"缺陷 {i}: 类型={d.get('type', 'Unknown')}, "
                    f"置信度={d.get('confidence', 0):.2%}, "
                    f"IPC等级={ipc_class}, 处置建议={disposition}\n"
                )

        # ✅ 核心改动：构建 Prompt 并调用 LLM
        prompt = _REPORT_PROMPT.invoke({
            "image_path": image_path,
            "defect_count": len(defects),
            "db_status": db_status,
            "defects_and_standards": defects_and_standards,
            "root_cause": root_cause
        })
        
        # ✅ R42修复：使用 ainvoke 异步调用，完美触发 on_chat_model_stream
        response = await _llm.ainvoke(prompt)
        report = response.content
        
        logger.info(f"   ✅ LLM 报告生成完成 ({len(report)} 字符)")
        return {"final_report": report, "next_step": "END"}

    except Exception as e:
        logger.error(f"   ❌ 报告生成失败: {e}", exc_info=True)
        # Fallback: 如果 LLM 调用失败，退化为简单的错误提示
        return {"final_report": f"# ❌ 报告生成失败\n\nLLM 调用异常: {str(e)}", "next_step": "END"}


def decision_agent(state: PCBQualityState) -> dict:
    """
    路由决策节点：根据当前 State 决定下一步走向
    ✅ R40-fix：使用 visual_done 标志判断是否已执行检测
    ✅ R43-fix：识别 db_write_count == -1，跳过 DB 节点防止死循环
    """
    logger.info("🎯 [Decision] 正在决策下一步...")

    if state.get("error"):
        logger.error(f"   ⛔ 检测到错误，终止流程: {state['error']}")
        return {"next_step": "END"}

    # ✅ 修复：用 visual_done 标志判断是否已执行检测，而非 defects 是否为空
    if not state.get("visual_done"):
        return {"next_step": "visual"}

    # ✅ R43 核心修复：DB 路由防死循环逻辑
    db_count = state.get("db_write_count", 0)
    if state.get("defects"):
        if db_count == 0:
            # 尚未写入过，尝试写入
            return {"next_step": "db_store"}
        elif db_count == -1:
            # 写入失败，记录警告并直接跳过，进入 standard 节点
            logger.warning("   ⚠️ 检测到 DB 写入失败 (-1)，跳过 DB 节点继续后续流程")
            # 注意：这里不返回 db_store，直接往下走
        # 如果 db_count > 0，说明写入成功，也继续往下走

    if "matched_standards" not in state:
        return {"next_step": "standard"}

    if "root_cause" not in state:
        return {"next_step": "rootcause"}

    if "final_report" not in state:
        return {"next_step": "report"}

    return {"next_step": "END"}


def route_decision(state: PCBQualityState) -> str:
    step = state.get("next_step", "END")
    valid_routes = {"visual", "db_store", "standard", "rootcause", "report", "END"}
    return step if step in valid_routes else "END"


def build_graph():
    builder = StateGraph(PCBQualityState)

    # 添加节点 (LangGraph 完美支持 async def 节点)
    builder.add_node("decision", decision_agent)
    builder.add_node("visual", visual_agent)
    builder.add_node("db_store", db_store_agent)
    builder.add_node("standard", standard_agent)
    builder.add_node("rootcause", rootcause_agent)
    builder.add_node("report", report_agent)

    # 设置入口
    builder.set_entry_point("decision")

    # 添加条件边（路由）
    builder.add_conditional_edges(
        "decision",
        route_decision,
        {
            "visual": "visual",
            "db_store": "db_store",
            "standard": "standard",
            "rootcause": "rootcause",
            "report": "report",
            "END": END
        }
    )

    # 所有业务节点执行完毕后，都回到 decision 节点进行下一步路由
    builder.add_edge("visual", "decision")
    builder.add_edge("db_store", "decision")
    builder.add_edge("standard", "decision")
    builder.add_edge("rootcause", "decision")
    builder.add_edge("report", END)

    return builder.compile()


# 模块级实例化
graph = build_graph()