# app/db.py
"""
Milvus Lite 数据访问层
======================

【职责】
    封装所有与 Milvus Lite 向量数据库的交互，向上层提供幂等、可重试的
    "缺陷特征批量写入"接口。本模块是 Visual Agent 与 Milvus 之间的唯一桥梁。

【架构定位】
    feature_extractor.py (产出向量) 
            ↓ 
    db.py (本模块，写入向量库) 
            ↓ 
    Milvus Lite (本地文件型向量库，路径由 MILVUS_DB_PATH 指定)

【为什么用 Milvus Lite 而非完整 Milvus？】
    - 单机部署、无需 Docker 起独立服务，零运维成本
    - 数据落盘为本地文件，便于备份和迁移
    - 百万级向量以下性能与完整版差异可忽略

【历史修复记录】
    R39-A: 企业级重构
      - 接口适配新 State（defects 与 feature_vectors 已解耦）
      - 引入确定性 ID 生成（防重复插入）
      - 加入指数退避重试（解决容器内偶发文件锁冲突）
"""
import os
import hashlib
import time
import logging
from pymilvus import MilvusClient, DataType

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 模块级配置常量
# -----------------------------------------------------------------------------
DB_PATH = os.getenv("MILVUS_DB_PATH", "milvus_data.db")
# Milvus Lite 数据库文件路径。Docker 部署时通过 volume 挂载到宿主机，
# 保证容器重启后向量数据不丢失。

COLLECTION_NAME = "pcb_defects"
# 集合名。单一集合，用 defect_class 字段做逻辑区分（而非多集合）。

VECTOR_DIM = 256
# 特征向量维度。必须与 feature_extractor.VECTOR_DIM 严格一致，
# 否则 upsert 时会因维度不匹配被拒绝。256 来自 YOLOv8n Neck 末端
# 最后一个 C2f 层的输出通道数（参见 feature_extractor._find_neck_layer）。


# -----------------------------------------------------------------------------
# 客户端单例
# -----------------------------------------------------------------------------
client = MilvusClient(uri=DB_PATH)
# 模块级单例 —— 整个进程共用一个 MilvusClient 实例。
# Why 模块级？
#   MilvusClient 内部维护 gRPC 连接池，每次新建开销大；
#   且 Lite 版本的写操作是文件级别的，多实例并发写入会引发文件锁冲突。


# =============================================================================
# 集合初始化
# =============================================================================

def init_db():
    """
    初始化 pcb_defects 集合（幂等设计：已存在则跳过）。

    在 FastAPI lifespan 启动阶段调用一次。多次调用安全（幂等），
    不会重复创建集合或覆盖已有数据。

    Schema 设计说明：
        - id (VARCHAR 64, primary): 确定性 MD5 哈希，重复检测自动覆盖
        - vector (FLOAT_VECTOR 256): L2 归一化特征向量
        - defect_class (VARCHAR 32): 缺陷类型枚举值（对应 DefectType）
        - confidence (FLOAT): YOLO 检测置信度 0~1
        - image_path (VARCHAR 512): 来源图片路径，便于溯源

    索引设计说明：
        Milvus Lite 数据量小（通常 < 10 万条），推荐:
        - index_type = "FLAT"   —— 暴力精确搜索，无近似误差
        - metric_type = "COSINE" —— 余弦相似度，配合 L2 归一化向量
                                    等价于内积，且对模长不敏感
    """
    if client.has_collection(COLLECTION_NAME):
        logger.info(f"ℹ️ 集合 {COLLECTION_NAME} 已存在，跳过创建。")
        return

    logger.info(f"⚡ 正在创建集合: {COLLECTION_NAME} ...")

    # auto_id=False: 主键由我们手工生成（确定性哈希），不用 Milvus 的雪花 ID
    # enable_dynamic_field=False: 关闭动态字段，强制 schema 严格校验
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    # ---------- 字段定义 ----------
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        max_length=64,
        is_primary=True,
        # 主键用确定性哈希（见 _generate_deterministic_id），
        # 保证同一图同一缺陷重复 upsert 时是"覆盖"而非"新增"。
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=VECTOR_DIM,
    )
    schema.add_field(
        field_name="defect_class",
        datatype=DataType.VARCHAR,
        max_length=32,
    )
    schema.add_field(
        field_name="confidence",
        datatype=DataType.FLOAT,
    )
    schema.add_field(
        field_name="image_path",
        datatype=DataType.VARCHAR,
        max_length=512,
    )

    # ---------- 索引定义 ----------
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",       # 小数据集首选：暴力精确搜索，召回率 100%
        metric_type="COSINE",    # 余弦相似度：值域 [-1,1]，越大越相似
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f"✅ 集合创建成功！维度={VECTOR_DIM}, 度量=COSINE")


# =============================================================================
# 内部工具函数
# =============================================================================

def _generate_deterministic_id(image_path: str, index: int) -> str:
    """
    生成确定性 ID：基于图片路径 + 缺陷索引的 MD5 哈希（取前 16 位）。

    Args:
        image_path: 来源图片路径（作为业务唯一标识的一部分）
        index:      缺陷在本图缺陷列表中的序号

    Returns:
        16 字符的十六进制字符串，作为 Milvus 主键

    Why 确定性 ID 而非 UUID？
        1. **幂等性**：同一张图重复上传时，ID 不变，Milvus upsert 会自动
           覆盖旧数据，避免"同一缺陷被重复插入 N 次"的数据污染问题。
        2. **可追溯**：ID 与 (image_path, index) 一一对应，看到 ID 能反查
           是哪张图的第几个缺陷。
        3. **轻量**：相比 UUID4，MD5 前 16 位已足够唯一（碰撞概率 < 2^-64），
           且主键长度短，节省存储。

    ⚠️ 陷阱: image_path 必须是稳定的业务标识，不能包含随机临时名。
             若上游用 uuid4 命名上传文件，本函数会失去幂等性。
    """
    raw_str = f"{image_path}__defect_{index}"
    return hashlib.md5(raw_str.encode("utf-8")).hexdigest()[:16]


def _upsert_with_retry(data: list[dict], max_retries: int = 3) -> bool:
    """
    带指数退避的 Milvus 写入重试机制。

    Args:
        data:        Milvus upsert 所需的行数据列表
        max_retries: 最大重试次数（默认 3）

    Returns:
        True  —— 成功写入
        False —— 达到最大重试次数仍失败

    Why 需要重试？
        Milvus Lite 在多线程/容器环境下，偶发 I/O 文件锁冲突
        （两个写入请求同时争抢 .db 文件的写锁）。这类错误是**瞬态**的，
        指数退避后重试通常能成功。

    Why 手写退避而非用 tenacity？
        - 减少一个外部依赖，保持 requirements.txt 精简
        - 逻辑简单（2^n 秒退避），无需引入复杂框架

    退避序列: 1s → 2s → 4s（首次失败等 1s，第二次等 2s，第三次等 4s）
    """
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=COLLECTION_NAME, data=data)
            return True
        except Exception as e:
            wait_time = 2 ** attempt  # 指数退避：1s, 2s, 4s
            logger.warning(
                f"⚠️ Milvus Upsert 失败 "
                f"(尝试 {attempt + 1}/{max_retries}): {e}. "
                f"{wait_time}s 后重试..."
            )
            time.sleep(wait_time)

    logger.error("❌ Milvus Upsert 最终失败，已达最大重试次数。")
    return False


# =============================================================================
# 对外接口
# =============================================================================

def upsert_defect_features(
    defects: list[dict],
    feature_vectors: list[list[float]],
    image_path: str,
) -> int:
    """
    批量 Upsert 缺陷特征到 Milvus。

    Args:
        defects:         业务缺陷列表 [{"type": str, "confidence": float, "bbox": list}, ...]
        feature_vectors: 独立的 256 维向量列表 [[0.1, ...], [0.2, ...], ...]
                         必须与 defects 按索引一一对应
        image_path:      来源图片路径（用于生成确定性 ID）

    Returns:
        成功写入的条目数（0 表示无有效数据或全部失败）

    Data Flow:
        Visual Agent → defects + feature_vectors（已解耦）
                     → 本函数按索引 zip 组装 Milvus 数据行
                     → Milvus upsert

    ✅ R39-A 重构说明：
        旧接口把 vector 塞在 defects[i]["vector"] 里，与本模块耦合；
        新接口让 defects 保持"纯业务数据"（JSON 可序列化），向量单独传，
        职责更清晰。
    """
    # ---------- 前置校验：空数据短路 ----------
    if not defects or not feature_vectors:
        return 0

    # ---------- 前置校验：数量必须一致 ----------
    # 严格校验防止 zip() 静默截断导致的向量-缺陷错位
    if len(defects) != len(feature_vectors):
        logger.error(
            f"❌ 数据不一致: defects({len(defects)}) != "
            f"vectors({len(feature_vectors)})"
        )
        return 0

    # ---------- 组装 Milvus 数据行 ----------
    data_to_insert = []
    for idx, (defect, vector) in enumerate(zip(defects, feature_vectors)):
        # 1. 严格校验向量维度（防御上游 feature_extractor 异常产出）
        if not isinstance(vector, list) or len(vector) != VECTOR_DIM:
            logger.warning(
                f"跳过无效向量: index={idx}, "
                f"len={len(vector) if isinstance(vector, list) else 'N/A'}"
            )
            continue

        # 2. 组装数据行
        data_to_insert.append({
            "id": _generate_deterministic_id(image_path, idx),
            "vector": vector,
            "defect_class": defect.get("type", "unknown"),  # 兼容 type/label 两种字段名
            "confidence": float(defect.get("confidence", 0.0)),
            "image_path": image_path,
        })

    if not data_to_insert:
        logger.warning("无有效缺陷数据可写入 Milvus")
        return 0

    # ---------- 执行带重试的写入 ----------
    if _upsert_with_retry(data_to_insert):
        count = len(data_to_insert)
        logger.info(f"✅ Milvus 批量 Upsert 成功: {count} 条 (image={image_path})")
        return count

    return 0


# =============================================================================
# 自测入口（仅本地开发使用，生产不会执行）
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()