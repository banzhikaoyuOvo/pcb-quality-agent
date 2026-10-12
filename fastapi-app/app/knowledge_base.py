# app/knowledge_base.py
"""
IPC 标准与企业 SOP 知识库管理模块
==================================

【职责】
    加载 data/ipc_standards.yaml 中的结构化标准，提供 O(1) 精确匹配查询。

【为什么用 YAML 精确匹配而非向量 RAG？】
    ┌──────────────┬──────────────┬────────────┐
    │   维度       │  向量 RAG    │  YAML 字典 │
    ├──────────────┼──────────────┼────────────┤
    │ 适用场景     │ 非结构化文档  │ 结构化条目 │
    │ 检索方式     │ 语义相似      │ 精确匹配   │
    │ 检索延迟     │ 10-100ms     │ <1ms       │
    │ 准确率       │ 85-95%       │ 100%       │
    │ 幻觉风险     │ 有            │ 零         │
    └──────────────┴──────────────┴────────────┘

    PCB 缺陷类型是**确定的 6 类枚举值**，IPC 标准按缺陷类型**一一对应**，
    无需语义检索。选型原则：
        查询键确定性 + 结果唯一性  →  精确匹配（YAML/字典）
        查询键模糊   + 结果多样性  →  语义检索（向量 RAG）

    反例：若用向量 RAG，"short"（短路）与"open_circuit"（开路）在向量空间
          可能高相似（都是电路异常），导致误匹配。
"""
import os
import yaml
from typing import Dict, Any, Optional

# 知识库文件路径（相对于项目根目录）
# 计算方式: __file__ = .../app/knowledge_base.py
#           abspath → dirname → .../app
#           dirname → .../fastapi-app
#           join("data", "ipc_standards.yaml")
_KB_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "ipc_standards.yaml",
)


class StandardKnowledgeBase:
    """
    IPC 标准知识库单例类。

    【单例模式的实现细节】
        - __new__ 中维护类级别 _instance，保证全局唯一实例
        - __init__ 中用 _is_loaded 标志避免重复加载 YAML
        - _standards 与 _is_loaded 声明为**类变量**而非实例变量，
          保证所有"伪实例"看到同一份数据

    【线程安全说明】
        FastAPI 是 async 单进程多协程模型，__new__ 中无 await 点，
        协程切换不会发生在 if cls._instance is None 判断与赋值之间，
        因此无需加锁。若将来引入多线程 worker，需补 threading.Lock。
    """

    _instance: Optional["StandardKnowledgeBase"] = None
    _is_loaded: bool = False
    _standards: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StandardKnowledgeBase, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # 只在首次实例化时加载 YAML，后续实例化直接跳过
        if not self._is_loaded:
            self._load_knowledge_base()

    def _load_knowledge_base(self):
        """
        从 YAML 文件加载知识库到内存字典。

        异常处理策略：**降级但不崩溃**
            - 文件不存在 → 记空字典，上游查询会返回"未找到"提示
            - YAML 解析错误 → 同上
            - 这样即使知识库损坏，服务仍能启动（只是标准匹配失效）

        Why 加载到内存而非每次读文件？
            文件 I/O 延迟 1-10ms，内存字典查询 <1μs，差 3-4 个数量级。
            知识库是只读数据，启动时加载一次，后续零成本。
        """
        try:
            with open(_KB_FILE_PATH, "r", encoding="utf-8") as f:
                self._standards = yaml.safe_load(f)
                self._is_loaded = True
                print(f"[KnowledgeBase] ✅ 成功加载 {len(self._standards)} 条 IPC 标准规则。")
        except FileNotFoundError:
            print(f"[KnowledgeBase] ❌ 致命错误：找不到知识库文件 {_KB_FILE_PATH}")
            self._standards = {}
        except yaml.YAMLError as e:
            print(f"[KnowledgeBase] ❌ YAML 解析错误：{e}")
            self._standards = {}

    def query_standard(self, defect_type: str) -> Dict[str, Any]:
        """
        根据缺陷类型精确查询 IPC 标准。

        Args:
            defect_type: 缺陷类型枚举值（如 'missing_hole', 'short'）

        Returns:
            - 命中: IPC 标准详情字典
            - 未命中: 包含 "error" 键的降级字典（调用方应检查此键）

        Example:
            >>> kb = StandardKnowledgeBase()
            >>> std = kb.query_standard("missing_hole")
            >>> std["ipc_reference"]
            'IPC-A-610G Class 2 / Class 3'

        Note:
            大小写容错：统一转小写后查询，兼容 "Missing_Hole"、"MISSING_HOLE"
            等 YOLO 输出中可能出现的大小写变体。
        """
        # 统一转小写 + 去空格，增加容错性
        defect_key = defect_type.strip().lower()

        standard = self._standards.get(defect_key)
        if standard:
            return standard

        # 降级返回：结构化 error 而非抛异常，让调用方决定如何处理
        return {
            "error": f"未找到缺陷类型 '{defect_type}' 的 IPC 标准记录。",
            "defect_name": "Unknown Defect",
            "description": "知识库中无此缺陷的定义，请人工复核。",
        }


# =============================================================================
# 模块级单例
# =============================================================================
# 供 graph.py 直接 `from app.knowledge_base import kb` 使用。
# 模块导入时立即加载 YAML（一次性成本，通常 <10ms）。
kb = StandardKnowledgeBase()