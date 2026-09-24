# tests/test_knowledge_base.py
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.knowledge_base import kb, StandardKnowledgeBase


def test_singleton():
    """验证知识库单例模式生效"""
    kb2 = StandardKnowledgeBase()
    assert kb is kb2, "StandardKnowledgeBase 应为单例"
    print("\n[OK] 单例模式验证通过")


def test_kb_loaded():
    """验证 YAML 知识库已成功加载且非空"""
    assert kb._is_loaded is True, "知识库应成功加载"
    assert len(kb._standards) > 0, "知识库不应为空"
    print(f"[OK] 知识库加载成功，共 {len(kb._standards)} 条规则")


def test_query_known_defect():
    """测试已知缺陷类型的 O(1) 精确匹配"""
    # 取知识库中第一个 key 作为测试用例，避免硬编码类别名
    first_key = next(iter(kb._standards))
    result = kb.query_standard(first_key)

    assert "error" not in result, f"已知缺陷 '{first_key}' 不应返回 error"
    assert isinstance(result, dict), "返回值应为 dict"
    print(f"[OK] 已知缺陷 '{first_key}' 匹配成功: {result.get('defect_name', 'N/A')}")


def test_query_unknown_defect():
    """测试未知缺陷类型的降级处理"""
    result = kb.query_standard("nonexistent_defect_xyz_12345")

    assert "error" in result, "未知缺陷应返回包含 error 的字典"
    assert result["defect_name"] == "Unknown Defect"
    assert "人工复核" in result["description"]
    print(f"[OK] 未知缺陷降级处理正确: {result['error']}")


def test_query_case_insensitive():
    """测试缺陷类型大小写容错"""
    first_key = next(iter(kb._standards))
    upper_key = first_key.upper()
    result = kb.query_standard(upper_key)

    assert "error" not in result, f"大写 '{upper_key}' 应能匹配到标准"
    print(f"[OK] 大小写容错验证通过: '{upper_key}' → '{first_key}'")