# app/llm_client.py
"""
✅ R38-C：统一的 LLM 客户端配置
基于 pydantic-settings 读取 .env，初始化 DeepSeek 兼容的 OpenAI 客户端
"""
from pydantic_settings import BaseSettings
from openai import OpenAI

class Settings(BaseSettings):
    deepseek_api_key: str
    llm_base_url: str
    llm_model: str

    class Config:
        env_file = ".env"

# 实例化配置
settings = Settings()

# ✅ 初始化 OpenAI 客户端（DeepSeek API 完美兼容 OpenAI SDK）
llm_client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.llm_base_url
)

def get_llm_model_name() -> str:
    return settings.llm_model