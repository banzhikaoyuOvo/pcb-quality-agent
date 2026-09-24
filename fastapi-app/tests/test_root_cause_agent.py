# tests/test_root_cause_agent.py
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.root_cause_agent import analyze_root_cause


def test_no_defects():
    """无缺陷时应直接返回免分析提示"""
    result = analyze_root_cause([], [])
    assert "未检测到缺陷" in result
    print(f"\n[OK] 无缺陷短路: {result}")


def test_no_valid_standards():
    """有缺陷但无有效 IPC 标准时应返回人工复核提示"""
    defects = [{"type": "short", "confidence": 0.95, "bbox": [0, 0, 10, 10], "vector": []}]
    bad_standards = [{"defect_type": "short", "standard": {"error": "未找到"}}]

    result = analyze_root_cause(defects, bad_standards)
    assert "人工复核" in result
    print(f"[OK] 无有效标准短路: {result[:60]}...")


@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY 未设置，跳过 LLM 真实调用测试"
)
def test_llm_real_call():
    """真实调用 DeepSeek API 验证根因分析输出"""
    defects = [
        {"type": "missing_hole", "confidence": 0.92, "bbox": [100, 200, 150, 250], "vector": []}
    ]
    standards = [
        {
            "defect_type": "missing_hole",
            "confidence": 0.92,
            "bbox": [100, 200, 150, 250],
            "standard": {
                "defect_name": "Missing Hole",
                "ipc_reference": "IPC-A-610G 2.8.3",
                "description": "焊盘缺少通孔或孔径超差",
                "rework_suggestion": "报废或工程评审"
            }
        }
    ]

    result = analyze_root_cause(defects, standards)

    assert "[RootCause Agent] LLM 调用失败" not in result, f"API 调用失败: {result}"
    assert len(result) >= 50, f"根因分析过短 ({len(result)} 字)，可能未正常生成"
    assert "IPC" in result or "标准" in result or "根因" in result, "输出应包含专业术语"
    print(f"[OK] LLM 真实调用成功 ({len(result)} 字)")
    print(f"    预览: {result[:100]}...")