# tests/test_feature_extractor.py
import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.feature_extractor import FeatureExtractor, VECTOR_DIM

# 模块级单例，避免重复加载模型
@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor()


def test_model_loaded(extractor):
    """测试模型和 Hook 是否成功初始化"""
    assert extractor.model is not None
    assert extractor._hook_layer_idx > 9, f"Hook 层应在 Neck 区域 (idx>9)，实际: {extractor._hook_layer_idx}"
    print(f"\n[OK] 模型加载成功，Hook 层索引: {extractor._hook_layer_idx}")


def test_extract_single_vector_dim(extractor):
    """测试整图特征向量维度是否为 256 且 L2 归一化"""
    test_img = "uploaded_images/test_defect_01.jpg"
    if not os.path.exists(test_img):
        pytest.skip(f"测试图片 {test_img} 不存在")

    vec = extractor.extract_single_vector(test_img)
    
    assert isinstance(vec, np.ndarray), f"应返回 np.ndarray，实际: {type(vec)}"
    assert vec.shape == (VECTOR_DIM,), f"维度应为 ({VECTOR_DIM},)，实际: {vec.shape}"
    
    norm = np.linalg.norm(vec)
    assert 0.99 < norm < 1.01, f"L2 模长应≈1.0，实际: {norm}"
    print(f"\n[OK] 整图向量验证通过: shape={vec.shape}, norm={norm:.4f}")


def test_extract_from_image_structure(extractor):
    """测试 extract_from_image 返回结构完整性"""
    test_img = "uploaded_images/test_defect_01.jpg"
    if not os.path.exists(test_img):
        pytest.skip(f"测试图片 {test_img} 不存在")

    defects = extractor.extract_from_image(test_img)
    
    assert isinstance(defects, list), "应返回 list"
    if len(defects) == 0:
        print("\n[WARN] 测试图片未检测到缺陷，跳过结构验证")
        return
    
    d = defects[0]
    required_keys = {"label", "confidence", "bbox", "vector"}
    assert required_keys.issubset(d.keys()), f"缺少字段: {required_keys - set(d.keys())}"
    assert isinstance(d["vector"], np.ndarray), "vector 应为 np.ndarray"
    assert d["vector"].shape == (VECTOR_DIM,), f"vector 维度应为 {VECTOR_DIM}"
    print(f"\n[OK] extract_from_image 结构验证通过: {len(defects)} 个缺陷")