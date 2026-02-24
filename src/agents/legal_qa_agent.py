# Legal QA Agent Module
"""
法律咨询智能体
支持RAG检索增强
"""

import json
import re
from typing import Dict, Any, List, Optional

from ..core.base_agent import BaseAgent, AgentResponse
from ..core.exceptions import AgentExecutionError
from ..prompts.legal_qa import (
    LEGAL_QA_SYSTEM_PROMPT,
    LEGAL_QA_USER_PROMPT,
    LEGAL_QA_SIMPLE_PROMPT,
    LEGAL_QA_LAW_SEARCH_PROMPT
)


class LegalQAAgent(BaseAgent):
    """
    法律咨询Agent

    功能：
    1. 解答法律问题
    2. 引用相关法条
    3. 提供操作建议
    4. 风险提示
    5. RAG检索增强（可选）
    """

    def __init__(self, llm_client, retriever=None):
        """
        初始化法律咨询Agent

        Args:
            llm_client: LLM客户端实例
            retriever: 法律检索器实例（可选，用于RAG增强）
        """
        super().__init__(llm_client, name="LegalQAAgent")
        self.retriever = retriever

    def set_retriever(self, retriever):
        """设置检索器"""
        self.retriever = retriever

    def has_rag(self) -> bool:
        """检查是否启用了RAG"""
        return self.retriever is not None

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return LEGAL_QA_SYSTEM_PROMPT

    def execute(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        执行法律咨询

        Args:
            input_data: 输入数据
                - question: 法律问题（必需）
                - simple_mode: 是否使用简化模式（可选）
                - context: 额外上下文（可选）
                - use_rag: 是否使用RAG增强（可选，默认True如果检索器可用）

        Returns:
            AgentResponse: 咨询结果
        """
        try:
            question = input_data.get("question", "").strip()

            if not question:
                return AgentResponse(
                    success=False,
                    error_message="请输入您的法律问题"
                )

            # 判断回答模式
            simple_mode = input_data.get("simple_mode", False)
            use_rag = input_data.get("use_rag", True)

            # 判断是否使用RAG
            rag_context = None
            rag_sources = []
            if use_rag and self.has_rag():
                rag_context, rag_sources = self._retrieve_context(question)

            # 获取回答
            if simple_mode:
                answer = self._quick_answer(question)
            else:
                # 合并用户提供的context和RAG检索的context
                context = input_data.get("context")
                if rag_context:
                    if context:
                        context = rag_context + "\n\n" + context
                    else:
                        context = rag_context
                answer = self._full_answer(question, context)

            # 解析回答
            parsed = self._parse_answer(answer)

            # 将法条引用转换为字典格式
            # 优先使用RAG检索的来源
            if rag_sources:
                sources = [{"law": s.get("metadata", {}).get("law_name", "") + " " +
                           s.get("metadata", {}).get("article_number", ""),
                           "text": s.get("text", "")[:100] + "..."} for s in rag_sources]
            else:
                sources = [{"law": law} for law in parsed.get("法律依据", [])]

            return AgentResponse(
                success=True,
                content=answer,
                sources=sources,
                metadata={
                    "question": question,
                    "has_law_reference": len(sources) > 0,
                    "rag_enabled": self.has_rag() and use_rag
                }
            )

        except Exception as e:
            raise AgentExecutionError(self.name, str(e))

    def _retrieve_context(self, question: str, n_results: int = 5) -> tuple:
        """
        检索相关法条作为上下文

        Args:
            question: 用户问题
            n_results: 检索结果数量

        Returns:
            tuple: (上下文文本, 来源列表)
        """
        if not self.has_rag():
            return None, []

        try:
            results = self.retriever.retrieve(question, n_results=n_results)

            if not results:
                return None, []

            context_parts = []
            for i, result in enumerate(results, 1):
                metadata = result.get("metadata", {})
                law_name = metadata.get("law_name", "")
                article = metadata.get("article_number", "")
                text = result.get("text", "")

                context_parts.append(f"[{i}] 《{law_name}》{article}\n{text}\n")

            context = "\n".join(context_parts)
            return context, results

        except Exception as e:
            # RAG失败时不影响正常回答
            print(f"RAG检索失败: {e}")
            return None, []

    def _full_answer(self, question: str, context: str = None) -> str:
        """
        生成完整回答

        Args:
            question: 法律问题
            context: 额外上下文

        Returns:
            str: 完整回答
        """
        # 构建提示词
        prompt = LEGAL_QA_USER_PROMPT.format(question=question)

        if context:
            prompt = f"【背景信息】\n{context}\n\n" + prompt

        # 构建消息
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        # 调用LLM
        response = self.chat(messages, temperature=0.5)

        # 保存到记忆
        self.add_to_memory("user", question)
        self.add_to_memory("assistant", response)

        return response

    def _quick_answer(self, question: str) -> str:
        """
        生成快速回答

        Args:
            question: 法律问题

        Returns:
            str: 简短回答
        """
        prompt = LEGAL_QA_SIMPLE_PROMPT.format(question=question)

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        response = self.chat(messages, temperature=0.5)

        self.add_to_memory("user", question)
        self.add_to_memory("assistant", response)

        return response

    def _parse_answer(self, answer: str) -> Dict:
        """
        解析回答，提取法律依据等结构化信息

        Args:
            answer: LLM回答

        Returns:
            Dict: 解析后的结构化数据
        """
        result = {
            "结论": "",
            "法律依据": [],
            "操作建议": [],
            "风险提示": []
        }

        # 提取法条引用
        # 匹配格式：《法律名称》第X条 或 《法律名称》第X条第X款
        law_pattern = r'《([^》]+)》第([一二三四五六七八九十百零\d]+)条(?:第([一二三四五六七八九十百零\d]+)款)?'
        law_refs = re.findall(law_pattern, answer)

        for ref in law_refs:
            law_name = ref[0]
            article = ref[1]
            clause = ref[2] if ref[2] else ""
            law_ref = f"《{law_name}》第{article}条"
            if clause:
                law_ref += f"第{clause}款"
            if law_ref not in result["法律依据"]:
                result["法律依据"].append(law_ref)

        # 尝试提取各部分内容
        sections = {
            "结论": r'###\s*结论\s*\n([\s\S]*?)(?=###|$)',
            "操作建议": r'###\s*操作建议\s*\n([\s\S]*?)(?=###|$)',
            "风险提示": r'###\s*风险提示\s*\n([\s\S]*?)(?=###|$)'
        }

        for key, pattern in sections.items():
            match = re.search(pattern, answer)
            if match:
                result[key] = match.group(1).strip()

        return result

    def ask(self, question: str, context: str = None) -> AgentResponse:
        """
        便捷方法：提问法律问题

        Args:
            question: 法律问题
            context: 额外上下文

        Returns:
            AgentResponse: 回答结果
        """
        return self.execute({
            "question": question,
            "context": context
        })

    def quick_ask(self, question: str) -> AgentResponse:
        """
        便捷方法：快速提问

        Args:
            question: 法律问题

        Returns:
            AgentResponse: 简短回答
        """
        return self.execute({
            "question": question,
            "simple_mode": True
        })

    def explain_law(self, law_name: str, article: str) -> AgentResponse:
        """
        解释特定法条

        Args:
            law_name: 法律名称（如"民法典"）
            article: 条款号（如"577"）

        Returns:
            AgentResponse: 法条解释
        """
        prompt = LEGAL_QA_LAW_SEARCH_PROMPT.format(
            law_name=law_name,
            article=article
        )

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        response = self.chat(messages, temperature=0.3)

        return AgentResponse(
            success=True,
            content=response,
            sources=[f"《{law_name}》第{article}条"],
            metadata={
                "law_name": law_name,
                "article": article
            }
        )

    def continue_conversation(self, follow_up: str) -> AgentResponse:
        """
        继续对话（基于历史记忆）

        Args:
            follow_up: 追问内容

        Returns:
            AgentResponse: 回答结果
        """
        if not self.memory:
            return self.ask(follow_up)

        # 使用带记忆的对话
        response = self.chat_with_memory(follow_up, temperature=0.5)

        return AgentResponse(
            success=True,
            content=response,
            metadata={"turn": len(self.memory) // 2}
        )

    def clear_conversation(self):
        """清空对话历史"""
        self.clear_memory()
