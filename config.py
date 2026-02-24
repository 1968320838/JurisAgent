# Configuration Module
"""
JurisAgent 全局配置
"""

import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# API配置
API_CONFIG = {
    "api_key": os.environ.get("ZHIPU_API_KEY", ""),  # 从环境变量读取，或直接填写
    "model": "GLM-4.7",  # GLM-4 Plus 模型
    "timeout": 120,         # 请求超时时间（秒）
    "max_retries": 3,       # 最大重试次数
    "retry_delay": 2        # 重试间隔（秒）
}

# 应用配置
APP_CONFIG = {
    "app_name": "JurisAgent",
    "version": "0.1.0",
    "debug": True,
    "max_file_size": 10 * 1024 * 1024,  # 最大文件大小 10MB
    "supported_formats": [".pdf", ".docx", ".doc", ".txt"]
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
