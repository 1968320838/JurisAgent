# API Client Module
"""
GLM-4.7 API客户端封装
使用requests直接调用智谱AI API
"""

import time
import json
from typing import List, Dict, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from config import API_CONFIG


class GLMClient:
    """GLM-4.7 API客户端"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化GLM客户端

        Args:
            api_key: API密钥，如果不提供则使用配置文件中的默认值
        """
        self.api_key = api_key or API_CONFIG["api_key"]
        self.model = API_CONFIG["model"]
        self.timeout = API_CONFIG["timeout"]
        self.max_retries = API_CONFIG["max_retries"]
        self.retry_delay = API_CONFIG["retry_delay"]

        self._total_tokens = 0
        self._total_calls = 0

        # API配置
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        同步聊天接口

        Args:
            messages: 消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]
            temperature: 温度参数，控制随机性

        Returns:
            模型回复内容
        """
        retry_count = 0
        last_error = None

        while retry_count < self.max_retries:
            try:
                self._total_calls += 1

                # 构建请求
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature
                }

                # 发送请求
                if HAS_REQUESTS:
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=data,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    result = response.json()
                else:
                    # 如果没有requests，使用urllib
                    import urllib.request
                    req = urllib.request.Request(
                        self.api_url,
                        data=json.dumps(data).encode('utf-8'),
                        headers=headers,
                        method='POST'
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        result = json.loads(resp.read().decode('utf-8'))

                # 解析响应
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']

                    # 估算token数量
                    self._total_tokens += self.estimate_tokens(content)

                    return content
                else:
                    raise Exception(f"API响应格式错误: {result}")

            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** retry_count))  # 指数退避
                else:
                    raise Exception(f"API调用失败（重试{self.max_retries}次后）: {last_error}")

    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量

        Args:
            text: 待估算的文本

        Returns:
            估算的token数量
        """
        # 粗略估算：中文字符约1.5 token，英文约0.25 token
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.25)

    def get_stats(self) -> Dict[str, int]:
        """
        获取使用统计

        Returns:
            包含总调用次数和总token数的字典
        """
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
        }

    def reset_stats(self):
        """重置统计数据"""
        self._total_tokens = 0
        self._total_calls = 0


# 全局单例
_glm_client: Optional[GLMClient] = None


def get_glm_client() -> GLMClient:
    """获取全局GLM客户端实例"""
    global _glm_client
    if _glm_client is None:
        _glm_client = GLMClient()
    return _glm_client
