# app/root_cause_agent.py
"""
R42 RootCause Agent - DeepSeek LLM 驱动的 PCB 缺陷根因分析（原生流式版）
✅ B-C2-V：从 OpenAI SDK 同步调用升级为 LangChain ChatOpenAI 异步流式
✅ 确保 astream_events v2 能捕获 on_chat_model_stream 事件
"""
import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = """你是一位资深 PCB 质量工程师，擅长根据 IPC-A-610G 标准和 YOLOv8 缺陷检测结果进行根因分析。

## 任务
根据【缺陷列表】和对应的【IPC标准】，推理出最可能的根本原因。

## 输出要求
1. 使用中文，200-400字
2. 结构：先总结根因 → 再逐条列出证据链（引用IPC标准编号）
3. 多个缺陷指向同一根因时合并分析
4. 严禁编造不存在的标准编号或工艺参数
5. 若信息不足，明确说明"需人工复核"而非猜测"""

# ✅ 模块级单例，避免重复初始化
_llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.2,
    max_tokens=1024,
    streaming=True,  # 🔑 关键：激活底层流式能力
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "## 缺陷列表 ({defect_count} 个)\n{defects}\n\n## 匹配的 IPC 标准\n{standards}"),
])

_chain = _prompt | _llm | StrOutputParser()


async def analyze_root_cause(
    defects: List[Dict[str, Any]],
    matched_standards: List[Dict[str, Any]]
) -> str:
    """
    异步流式调用 DeepSeek 生成根因分析
    ✅ 返回完整字符串供 State 存储，同时流式 Token 会被 astream_events 自动捕获
    """
    if not defects:
        return "未检测到缺陷，无需根因分析。"

    has_valid_standard = any("error" not in s.get("standard", {}) for s in matched_standards)
    if not has_valid_standard:
        return f"检测到 {len(defects)} 个缺陷，但 IPC 标准库无匹配记录，无法自动推理根因。请人工复核。"

    try:
        result = await _chain.ainvoke({
            "defect_count": len(defects),
            "defects": str(defects),
            "standards": str(matched_standards),
        })
        return result.strip()
    except Exception as e:
        print(f"❌ DeepSeek API 调用失败: {e}")
        return f"[RootCause Agent] LLM 调用失败: {str(e)}。请检查 API 配置。"