# app/db.py
"""
Milvus Lite 数据访问层
✅ R39-A：企业级重构 —— 接口适配新State + 确定性ID生成 + I/O抖动重试机制
"""
import os
import hashlib
import time
import logging
from pymilvus import MilvusClient, DataType

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("MILVUS_DB_PATH", "milvus_data.db")
COLLECTION_NAME = "pcb_defects"
VECTOR_DIM = 256

# 初始化 Milvus Lite 客户端（模块级单例）
client = MilvusClient(uri=DB_PATH)


def init_db():
    """
    初始化 pcb_defects 集合（幂等设计：已存在则跳过）
    """
    if client.has_collection(COLLECTION_NAME):
        logger.info(f"ℹ️ 集合 {COLLECTION_NAME} 已存在，跳过创建。")
        return

    logger.info(f"⚡ 正在创建集合: {COLLECTION_NAME} ...")
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    
    # 定义 Schema：主键 + 向量 + 标量过滤字段
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field(field_name="defect_class", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="confidence", datatype=DataType.FLOAT)
    schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=512)

    # 定义索引：Milvus Lite 小数据集推荐 FLAT (精确搜索) + COSINE (余弦相似度)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",       
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f"✅ 集合创建成功！维度={VECTOR_DIM}, 度量=COSINE")


def _generate_deterministic_id(image_path: str, index: int) -> str:
    """
    生成确定性 ID：基于图片路径和缺陷索引的 MD5 哈希。
    好处：同一张图片重复检测时，ID 不变，Milvus upsert 会自动覆盖旧数据，防止重复插入。
    """
    raw_str = f"{image_path}__defect_{index}"
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()[:16]  # 取前16位，足够唯一


def _upsert_with_retry(data: list[dict], max_retries: int = 3) -> bool:
    """
    带指数退避的重试写入机制（纯手写，无需额外安装 tenacity 库）
    解决 Milvus Lite 在容器内偶发的 I/O 文件锁冲突问题。
    """
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=COLLECTION_NAME, data=data)
            return True
        except Exception as e:
            wait_time = 2 ** attempt  # 指数退避：1s, 2s, 4s
            logger.warning(f"⚠️ Milvus Upsert 失败 (尝试 {attempt + 1}/{max_retries}): {e}. {wait_time}s 后重试...")
            time.sleep(wait_time)
    
    logger.error("❌ Milvus Upsert 最终失败，已达最大重试次数。")
    return False


def upsert_defect_features(
    defects: list[dict], 
    feature_vectors: list[list[float]], 
    image_path: str
) -> int:
    """
    ✅ R39-A：批量 Upsert 缺陷特征（适配新 State 独立向量字段）
    
    Args:
        defects: 业务缺陷列表 [{"type": str, "confidence": float, "bbox": list}, ...]
        feature_vectors: 独立的 256 维向量列表 [[0.1, ...], [0.2, ...], ...]
        image_path: 来源图片路径
    
    Returns:
        成功写入的条目数
    """
    if not defects or not feature_vectors:
        return 0

    if len(defects) != len(feature_vectors):
        logger.error(f"❌ 数据不一致: defects({len(defects)}) != vectors({len(feature_vectors)})")
        return 0

    data_to_insert = []
    for idx, (defect, vector) in enumerate(zip(defects, feature_vectors)):
        # 1. 严格校验向量维度
        if not isinstance(vector, list) or len(vector) != VECTOR_DIM:
            logger.warning(f"跳过无效向量: index={idx}, len={len(vector) if isinstance(vector, list) else 'N/A'}")
            continue
            
        # 2. 组装 Milvus 数据行
        data_to_insert.append({
            "id": _generate_deterministic_id(image_path, idx),
            "vector": vector,
            "defect_class": defect.get("type", "unknown"),  # 兼容 type/label
            "confidence": float(defect.get("confidence", 0.0)),
            "image_path": image_path,
        })

    if not data_to_insert:
        logger.warning("无有效缺陷数据可写入 Milvus")
        return 0

    # 3. 执行带重试的写入
    if _upsert_with_retry(data_to_insert):
        count = len(data_to_insert)
        logger.info(f"✅ Milvus 批量 Upsert 成功: {count} 条 (image={image_path})")
        return count
    
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()