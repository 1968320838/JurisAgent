# Case Analysis Agent Module
"""
案例分析智能体
支持案件分析、争议焦点提取、裁判预测
"""

import json
import re
from typing import Dict, Any, List, Optional

from ..core.base_agent import BaseAgent, AgentResponse
from ..core.exceptions import AgentExecutionError
from ..prompts.case_analysis import (
    CASE_ANALYSIS_SYSTEM_PROMPT,
    CASE_ANALYSIS_USER_PROMPT,
    CASE_ANALYSIS_SIMPLE_PROMPT,
    FOCUS_EXTRACTION_PROMPT,
    JUDGMENT_PREDICTION_PROMPT
)


class CaseAnalysisAgent(BaseAgent):
    """
    案例分析Agent

    功能：
    1. 案件事实梳理
    2. 法律关系分析
    3. 争议焦点提取
    4. 法律适用分析
    5. 裁判倾向预测
    6. 风险提示与建议
    7. RAG检索增强（可选）
    """

    def __init__(self, llm_client, retriever=None):
        """
        初始化案例分析Agent

        Args:
            llm_client: LLM客户端实例
            retriever: 法律检索器实例（可选，用于RAG增强）
        """
        super().__init__(llm_client, name="CaseAnalysisAgent")
        self.retriever = retriever

    def set_retriever(self, retriever):
        """设置检索器"""
        self.retriever = retriever

    def has_rag(self) -> bool:
        """检查是否启用了RAG"""
        return self.retriever is not None

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return CASE_ANALYSIS_SYSTEM_PROMPT

    def execute(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        执行案例分析

        Args:
            input_data: 输入数据
                - case_content: 案件内容（必需）
                - simple_mode: 是否使用简化模式（可选）
                - analysis_type: 分析类型（可选：full, focus, prediction）
                - use_rag: 是否使用RAG增强（可选，默认True如果检索器可用）

        Returns:
            AgentResponse: 分析结果
        """
        try:
            case_content = input_data.get("case_content", "").strip()

            if not case_content:
                return AgentResponse(
                    success=False,
                    error_message="请输入案件内容"
                )

            # 判断分析模式
            simple_mode = input_data.get("simple_mode", False)
            analysis_type = input_data.get("analysis_type", "full")
            use_rag = input_data.get("use_rag", True)

            # RAG检索相关法条
            rag_context = None
            rag_sources = []
            if use_rag and self.has_rag():
                rag_context, rag_sources = self._retrieve_context(case_content)

            # 根据分析类型执行不同分析
            if analysis_type == "focus":
                result = self._extract_focus(case_content)
            elif analysis_type == "prediction":
                result = self._predict_judgment(case_content, rag_context)
            else:
                result = self._full_analysis(case_content, rag_context, simple_mode)

            # 解析结果
            parsed = self._parse_result(result)

            # 构建来源列表
            if rag_sources:
                sources = [{
                    "law": s.get("metadata", {}).get("law_name", "") + " " +
                           s.get("metadata", {}).get("article_number", ""),
                    "text": s.get("text", "")[:100] + "..."
                } for s in rag_sources]
            else:
                sources = [{"law": law} for law in parsed.get("法律依据", [])]

            return AgentResponse(
                success=True,
                content=result,
                sources=sources,
                metadata={
                    "case_type": parsed.get("案件类型", ""),
                    "focus_count": len(parsed.get("争议焦点", [])),
                    "rag_enabled": self.has_rag() and use_rag,
                    "analysis_type": analysis_type
                }
            )

        except Exception as e:
            raise AgentExecutionError(self.name, str(e))

    def _retrieve_context(self, case_content: str, n_results: int = 5) -> tuple:
        """
        检索相关法条作为上下文

        Args:
            case_content: 案件内容
            n_results: 检索结果数量

        Returns:
            tuple: (上下文文本, 来源列表)
        """
        if not self.has_rag():
            return None, []

        try:
            # 提取关键词进行检索
            keywords = self._extract_keywords(case_content)
            query = " ".join(keywords[:5]) if keywords else case_content[:200]

            results = self.retriever.retrieve(query, n_results=n_results)

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
            print(f"RAG检索失败: {e}")
            return None, []

    def _extract_keywords(self, text: str) -> List[str]:
        """提取案件关键词"""
        # 简单的关键词提取
        keywords = []

        # 常见案件类型关键词
        type_keywords = [
            "合同", "借贷", "买卖", "租赁", "劳动", "工伤", "交通事故",
            "离婚", "继承", "侵权", "债务", "违约", "赔偿", "纠纷"
        ]

        for kw in type_keywords:
            if kw in text:
                keywords.append(kw)

        return keywords

    def _full_analysis(self, case_content: str, context: str = None,
                       simple_mode: bool = False) -> str:
        """
        完整案件分析

        Args:
            case_content: 案件内容
            context: RAG检索的上下文
            simple_mode: 是否简化模式

        Returns:
            str: 分析结果
        """
        if simple_mode:
            prompt = CASE_ANALYSIS_SIMPLE_PROMPT.format(case_content=case_content)
        else:
            prompt = CASE_ANALYSIS_USER_PROMPT.format(case_content=case_content)

        if context:
            prompt = f"【相关法律条文】\n{context}\n\n" + prompt

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        response = self.chat(messages, temperature=0.5)

        # 保存到记忆
        self.add_to_memory("user", case_content[:500])
        self.add_to_memory("assistant", response)

        return response

    def _extract_focus(self, case_content: str) -> str:
        """
        提取争议焦点

        Args:
            case_content: 案件内容

        Returns:
            str: 争议焦点分析
        """
        prompt = FOCUS_EXTRACTION_PROMPT.format(case_content=case_content)

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        return self.chat(messages, temperature=0.3)

    def _predict_judgment(self, case_content: str, context: str = None) -> str:
        """
        裁判预测

        Args:
            case_content: 案件内容
            context: RAG检索的上下文

        Returns:
            str: 预测结果
        """
        prompt = JUDGMENT_PREDICTION_PROMPT.format(
            case_facts=case_content,
            legal_basis=context or "根据案件事实自行分析"
        )

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        return self.chat(messages, temperature=0.5)

    def _parse_result(self, result: str) -> Dict:
        """
        解析分析结果

        Args:
            result: 分析结果文本

        Returns:
            Dict: 解析后的结构化数据
        """
        parsed = {
            "案件类型": "",
            "争议焦点": [],
            "法律依据": [],
            "裁判预测": ""
        }

        # 提取法条引用
        law_pattern = r'《([^》]+)》第([一二三四五六七八九十百零\d]+)条'
        law_refs = re.findall(law_pattern, result)

        for ref in law_refs:
            law_ref = f"《{ref[0]}》第{ref[1]}条"
            if law_ref not in parsed["法律依据"]:
                parsed["法律依据"].append(law_ref)

        # 提取争议焦点
        focus_pattern = r'(?:焦点|争议点)[一二三四五六七八九十\d]*[、:：]\s*([^\n]+)'
        focuses = re.findall(focus_pattern, result)
        parsed["争议焦点"] = [f.strip() for f in focuses if f.strip()]

        return parsed

    def analyze(self, case_content: str, simple_mode: bool = False) -> AgentResponse:
        """
        便捷方法：分析案件

        Args:
            case_content: 案件内容
            simple_mode: 是否简化模式

        Returns:
            AgentResponse: 分析结果
        """
        return self.execute({
            "case_content": case_content,
            "simple_mode": simple_mode
        })

    def extract_focus(self, case_content: str) -> AgentResponse:
        """
        便捷方法：提取争议焦点

        Args:
            case_content: 案件内容

        Returns:
            AgentResponse: 争议焦点
        """
        return self.execute({
            "case_content": case_content,
            "analysis_type": "focus"
        })

    def predict_judgment(self, case_content: str) -> AgentResponse:
        """
        便捷方法：裁判预测

        Args:
            case_content: 案件内容

        Returns:
            AgentResponse: 预测结果
        """
        return self.execute({
            "case_content": case_content,
            "analysis_type": "prediction"
        })
