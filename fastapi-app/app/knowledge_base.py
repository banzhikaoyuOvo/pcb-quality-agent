"""
IPC 标准与企业 SOP 知识库管理模块
采用结构化 YAML 存储 + 内存字典精确匹配，确保工业级零幻觉查询。
"""
import os
import yaml
from typing import Dict, Any, Optional

# 知识库文件路径（相对于项目根目录）
_KB_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "data", 
    "ipc_standards.yaml"
)

class StandardKnowledgeBase:
    """IPC 标准知识库单例类"""
    
    _instance: Optional['StandardKnowledgeBase'] = None
    _is_loaded: bool = False
    _standards: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StandardKnowledgeBase, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_loaded:
            self._load_knowledge_base()

    def _load_knowledge_base(self):
        """从 YAML 文件加载知识库到内存"""
        try:
            with open(_KB_FILE_PATH, 'r', encoding='utf-8') as f:
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
            defect_type: 缺陷类型枚举值 (如 'missing_hole', 'short')
            
        Returns:
            包含标准详情的字典；若未找到则返回包含 error 信息的字典。
        """
        # 统一转小写，增加容错性
        defect_key = defect_type.strip().lower()
        
        standard = self._standards.get(defect_key)
        if standard:
            return standard
        
        return {
            "error": f"未找到缺陷类型 '{defect_type}' 的 IPC 标准记录。",
            "defect_name": "Unknown Defect",
            "description": "知识库中无此缺陷的定义，请人工复核。"
        }

# 模块级单例，供 graph.py 直接 import 使用
kb = StandardKnowledgeBase()