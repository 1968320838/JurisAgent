# Base Agent Module
"""
Agent基类定义
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from enum import Enum


class AgentType(Enum):
    """Agent类型枚举"""
    CONTRACT_REVIEW = "contract_review"
    LEGAL_QA = "legal_qa"
    CASE_ANALYSIS = "case_analysis"


class AgentResponse(BaseModel):
    """Agent响应标准格式"""
    success: bool = True
    content: str = ""
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    confidence: float = 0.0
    error_message: Optional[str] = None


class BaseAgent(ABC):
    """
    Agent基类

    所有具体Agent都应继承此类并实现execute方法
    """

    def __init__(self, llm_client, name: str = "BaseAgent"):
        """
        初始化Agent

        Args:
            llm_client: LLM客户端实例
            name: Agent名称
        """
        self.llm = llm_client
        self.name = name
        self.memory: List[Dict[str, str]] = []

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        执行Agent任务

        Args:
            input_data: 输入数据字典

        Returns:
            AgentResponse: 标准响应格式
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        获取系统提示词

        Returns:
            str: 系统提示词
        """
        pass

    def add_to_memory(self, role: str, content: str) -> None:
        """
        添加消息到对话记忆

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
        """
        self.memory.append({"role": role, "content": content})

    def clear_memory(self) -> None:
        """清空对话记忆"""
        self.memory = []

    def get_memory_context(self, max_turns: int = 5) -> List[Dict[str, str]]:
        """
        获取最近N轮对话记忆

        Args:
            max_turns: 最大轮数

        Returns:
            List[Dict]: 对话历史
        """
        return self.memory[-max_turns * 2:] if len(self.memory) > max_turns * 2 else self.memory

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        调用LLM进行对话

        Args:
            messages: 消息列表
            temperature: 温度参数

        Returns:
            str: LLM回复
        """
        return self.llm.chat(messages, temperature=temperature)

    def chat_with_memory(self, user_input: str, temperature: float = 0.7) -> str:
        """
        带记忆的对话

        Args:
            user_input: 用户输入
            temperature: 温度参数

        Returns:
            str: LLM回复
        """
        # 构建消息列表
        messages = [{"role": "system", "content": self.get_system_prompt()}]

        # 添加对话历史
        messages.extend(self.get_memory_context())

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})

        # 调用LLM
        response = self.chat(messages, temperature)

        # 保存到记忆
        self.add_to_memory("user", user_input)
        self.add_to_memory("assistant", response)

        return response
