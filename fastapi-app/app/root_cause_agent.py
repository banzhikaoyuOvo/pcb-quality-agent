# app/root_cause_agent.py
"""
RootCause Agent —— DeepSeek LLM 驱动的 PCB 缺陷根因分析
=======================================================

【职责】
    根据 Visual Agent 检出的缺陷列表 + Standard Agent 匹配的 IPC 标准，
    调用 DeepSeek LLM 生成结构化的根因分析文本（200-400 字中文）。

【技术演进】
    - B-C2-V: 从 OpenAI SDK 同步调用 → LangChain ChatOpenAI 异步流式
    - 关键收益: astream_events v2 能捕获 on_chat_model_stream 事件，
               使前端能实时逐 Token 显示分析过程

【为什么用 Prompt 而非规则引擎？】
    根因分析需要"跨缺陷推理"（多个缺陷指向同一根因时合并分析）和
    "证据链构建"（引用 IPC 标准编号），这类模糊推理是 LLM 的强项，
    规则引擎（if-else）难以覆盖。

【反幻觉设计】
    Prompt 中明确要求"严禁编造不存在的标准编号或工艺参数"，
    且分析只允许引用注入的 matched_standards 中真实存在的 IPC 条目。
"""
import os
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =============================================================================
# System Prompt
# =============================================================================
SYSTEM_PROMPT = """你是一位资深 PCB 质量工程师，擅长根据 IPC-A-610G 标准和 YOLOv8 缺陷检测结果进行根因分析。

## 任务
根据【缺陷列表】和对应的【IPC标准】，推理出最可能的根本原因。

## 输出要求
1. 使用中文，200-400字
2. 结构：先总结根因 → 再逐条列出证据链（引用IPC标准编号）
3. 多个缺陷指向同一根因时合并分析
4. 严禁编造不存在的标准编号或工艺参数
5. 若信息不足，明确说明"需人工复核"而非猜测"""
# Prompt 工程要点:
#   - 角色锚定 ("资深 PCB 质量工程师") 提升回答专业性
#   - 输出结构约束 (先总结 → 再证据链) 保证报告可读性
#   - 字数约束 (200-400字) 防止 LLM 废话或过于简短
#   - 反幻觉约束 (严禁编造) 是工业场景的硬性要求
#   - 兜底指令 (需人工复核) 避免模型"硬答"造成误导


# =============================================================================
# LLM 单例
# =============================================================================
# ✅ 模块级单例，避免每次调用都新建 HTTP 连接
_llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.2,
    # temperature 取低值 (0.2)：根因分析需要严谨推理而非创意发散，
    # 低温度能让输出更确定、更专业。
    max_tokens=1024,
    # 单次生成上限。200-400 字中文约需 300-600 token，
    # 1024 留足余量，同时防止 LLM 跑偏后长篇大论。
    streaming=True,
    # 🔑 关键：激活底层流式能力。
    # 这是 astream_events v2 捕获 on_chat_model_stream 事件的前提条件，
    # 没有它前端无法实现"打字机"效果。
)


# =============================================================================
# Prompt 模板 + Chain
# =============================================================================
_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    (
        "human",
        "## 缺陷列表 ({defect_count} 个)\n{defects}\n\n"
        "## 匹配的 IPC 标准\n{standards}",
    ),
])

# LCEL (LangChain Expression Language) 管道: Prompt → LLM → 字符串解析
# StrOutputParser 将 ChatOpenAI 返回的 AIMessage 转成纯 str
_chain = _prompt | _llm | StrOutputParser()


# =============================================================================
# 对外接口
# =============================================================================

async def analyze_root_cause(
    defects: List[Dict[str, Any]],
    matched_standards: List[Dict[str, Any]],
) -> str:
    """
    异步调用 DeepSeek 生成根因分析。

    Args:
        defects:          Visual Agent 检出的缺陷列表
        matched_standards: Standard Agent 匹配的 IPC 标准列表

    Returns:
        根因分析文本（纯字符串，供 State 存储）

    Early Return 分支（省一次 LLM 调用）:
        1. 无缺陷 → 返回"未检测到缺陷"
        2. 有缺陷但 IPC 标准库无匹配 → 返回"请人工复核"

    Note:
        流式 Token 会被 astream_events v2 自动捕获，
        本函数返回完整字符串供 State 存储（用于报告生成阶段）。
        两者是"同一份数据，两种消费方式"。
    """
    # ---------- Early Return 1: 无缺陷 ----------
    if not defects:
        return "未检测到缺陷，无需根因分析。"

    # ---------- Early Return 2: 无有效 IPC 标准 ----------
    # 检查 matched_standards 中至少有一条不含 "error" 键
    has_valid_standard = any(
        "error" not in s.get("standard", {}) for s in matched_standards
    )
    if not has_valid_standard:
        return (
            f"检测到 {len(defects)} 个缺陷，但 IPC 标准库无匹配记录，"
            f"无法自动推理根因。请人工复核。"
        )

    # ---------- 真实 LLM 调用 ----------
    try:
        result = await _chain.ainvoke({
            "defect_count": len(defects),
            "defects": str(defects),
            "standards": str(matched_standards),
        })
        return result.strip()
    except Exception as e:
        # 不抛出异常，返回带错误信息的字符串，让上层 Report Agent 继续生成报告
        # （降级优于崩溃 —— 用户至少能看到检测结果，只是缺少根因）
        print(f"❌ DeepSeek API 调用失败: {e}")
        return f"[RootCause Agent] LLM 调用失败: {str(e)}。请检查 API 配置。"