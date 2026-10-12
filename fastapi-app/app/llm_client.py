# app/llm_client.py
"""
统一的 LLM 客户端配置模块
==========================

【职责】
    基于 pydantic-settings 从 .env 读取配置，初始化一个 DeepSeek 兼容的
    OpenAI 客户端。供 root_cause_agent.py 等模块复用。

【为什么用 OpenAI SDK 调用 DeepSeek？】
    DeepSeek 官方 API **完全兼容 OpenAI SDK 协议**（路径、请求体、响应体
    格式一致），只需改 base_url 和 api_key 即可无缝切换，无需自研 HTTP 封装。

【为什么用 pydantic-settings 而非 os.getenv？】
    - 自动从 .env 加载，无需 python-dotenv 手动调用
    - 类型校验：配置缺字段时启动即报错，而非运行时才暴露
    - 集中管理：所有配置项在 Settings 类一目了然

【✅ R38-C 说明】
    本模块是历史重构的产物，主流程已迁移到 root_cause_agent.py 中的
    LangChain ChatOpenAI 封装。保留本模块是为了向后兼容和配置说明。
"""
from pydantic_settings import BaseSettings
from openai import OpenAI


class Settings(BaseSettings):
    """
    应用配置类 —— 从环境变量 / .env 加载。

    字段名小写，对应 .env 中的大写变量（pydantic-settings 自动大小写不敏感）。

    ⚠️ 启动时校验：缺少任一必填字段会抛 ValidationError，服务无法启动。
       这是"Fail Fast"原则的体现 —— 配置问题要在启动时暴露，而不是
       等到用户上传图片才报 500。
    """
    deepseek_api_key: str    # 对应 .env: DEEPSEEK_API_KEY
    llm_base_url: str        # 对应 .env: LLM_BASE_URL（如 https://api.deepseek.com）
    llm_model: str           # 对应 .env: LLM_MODEL（如 deepseek-chat）

    class Config:
        env_file = ".env"    # 从当前工作目录的 .env 加载


# 模块级单例 —— 导入时立即读取配置并校验
settings = Settings()

# ✅ 初始化 OpenAI 客户端（DeepSeek API 完美兼容 OpenAI SDK）
# 之后所有 LLM 调用都复用此客户端，共享底层 HTTP 连接池
llm_client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.llm_base_url,
)


def get_llm_model_name() -> str:
    """返回配置的 LLM 模型名，供需要显式指定 model 参数的场景调用。"""
    return settings.llm_model