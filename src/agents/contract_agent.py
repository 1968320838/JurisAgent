# Contract Review Agent Module
"""
合同审查智能体
"""

import json
import re
from typing import Dict, Any, List, Optional

from ..core.base_agent import BaseAgent, AgentResponse
from ..core.exceptions import AgentExecutionError, DocumentParseError
from ..prompts.contract_review import (
    CONTRACT_REVIEW_SYSTEM_PROMPT,
    CONTRACT_REVIEW_USER_PROMPT,
    CONTRACT_REVIEW_SIMPLE_PROMPT
)
from ..parsers.pdf_parser import PDFParser
from ..parsers.docx_parser import DocxParser


class ContractReviewAgent(BaseAgent):
    """
    合同审查Agent

    功能：
    1. 解析PDF/Word格式合同
    2. 识别风险条款
    3. 提供修改建议
    4. 生成审查报告
    """

    def __init__(self, llm_client):
        """
        初始化合同审查Agent

        Args:
            llm_client: LLM客户端实例
        """
        super().__init__(llm_client, name="ContractReviewAgent")

        # 初始化文档解析器
        self.pdf_parser = PDFParser()
        self.docx_parser = DocxParser()

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return CONTRACT_REVIEW_SYSTEM_PROMPT

    def execute(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        执行合同审查

        Args:
            input_data: 输入数据，支持以下格式：
                - {"contract_text": "..."}  直接传入合同文本
                - {"file_path": "..."}      传入文件路径
                - {"file_bytes": ..., "file_type": "pdf/docx"}  传入文件字节

        Returns:
            AgentResponse: 审查结果
        """
        try:
            # Step 1: 获取合同文本
            contract_text = self._get_contract_text(input_data)

            if not contract_text or len(contract_text.strip()) < 50:
                return AgentResponse(
                    success=False,
                    error_message="合同文本过短或为空，无法进行审查"
                )

            # Step 2: 调用LLM进行审查
            review_result = self._review_contract(contract_text, input_data)

            # Step 3: 解析结果
            parsed_result = self._parse_review_result(review_result)

            return AgentResponse(
                success=True,
                content=review_result,
                sources=parsed_result.get("风险条款", []),
                metadata={
                    "contract_length": len(contract_text),
                    "risk_count": len(parsed_result.get("风险条款", [])),
                    "compliance_score": parsed_result.get("总体评估", {}).get("合规评分", "N/A")
                }
            )

        except DocumentParseError as e:
            return AgentResponse(
                success=False,
                error_message=str(e)
            )
        except Exception as e:
            raise AgentExecutionError(self.name, str(e))

    def _get_contract_text(self, input_data: Dict[str, Any]) -> str:
        """
        从输入数据中获取合同文本

        Args:
            input_data: 输入数据

        Returns:
            str: 合同文本
        """
        # 方式1：直接传入文本
        if "contract_text" in input_data:
            return input_data["contract_text"]

        # 方式2：传入文件路径
        if "file_path" in input_data:
            file_path = input_data["file_path"]
            return self._parse_file(file_path)

        # 方式3：传入文件字节
        if "file_bytes" in input_data:
            file_bytes = input_data["file_bytes"]
            file_type = input_data.get("file_type", "pdf")
            return self._parse_bytes(file_bytes, file_type)

        raise ValueError("输入数据必须包含 contract_text、file_path 或 file_bytes")

    def _parse_file(self, file_path: str) -> str:
        """解析文件并返回文本"""
        file_ext = file_path.lower().split(".")[-1]

        if file_ext == "pdf":
            result = self.pdf_parser.parse(file_path)
        elif file_ext in ["docx", "doc"]:
            result = self.docx_parser.parse(file_path)
        elif file_ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise DocumentParseError(f"不支持的文件格式: {file_ext}", file_path)

        return result["text"]

    def _parse_bytes(self, file_bytes: bytes, file_type: str) -> str:
        """解析字节数据并返回文本"""
        if file_type == "pdf":
            result = self.pdf_parser.parse_from_bytes(file_bytes)
        elif file_type in ["docx", "doc"]:
            result = self.docx_parser.parse_from_bytes(file_bytes)
        else:
            raise DocumentParseError(f"不支持的文件格式: {file_type}")

        return result["text"]

    def _review_contract(self, contract_text: str, input_data: Dict) -> str:
        """
        调用LLM审查合同

        Args:
            contract_text: 合同文本
            input_data: 原始输入数据

        Returns:
            str: 审查结果
        """
        # 判断是否使用简化模式
        simple_mode = input_data.get("simple_mode", False)

        if simple_mode:
            prompt = CONTRACT_REVIEW_SIMPLE_PROMPT.format(contract_text=contract_text)
        else:
            prompt = CONTRACT_REVIEW_USER_PROMPT.format(contract_text=contract_text)

        # 构建消息
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        # 调用LLM（使用较低温度保证稳定性）
        response = self.chat(messages, temperature=0.3)

        return response

    def _parse_review_result(self, result: str) -> Dict:
        """
        解析审查结果

        Args:
            result: LLM返回的审查结果

        Returns:
            Dict: 解析后的结构化结果
        """
        # 尝试提取JSON部分
        json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试直接解析整个响应
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            pass

        # 无法解析为JSON，返回原始文本
        return {
            "原始审查结果": result,
            "风险条款": [],
            "总体评估": {"合规评分": "N/A"}
        }

    def quick_review(self, contract_text: str) -> AgentResponse:
        """
        快速审查合同（简化版）

        Args:
            contract_text: 合同文本

        Returns:
            AgentResponse: 审查结果
        """
        return self.execute({
            "contract_text": contract_text,
            "simple_mode": True
        })

    def review_from_file(self, file_path: str) -> AgentResponse:
        """
        从文件审查合同

        Args:
            file_path: 合同文件路径

        Returns:
            AgentResponse: 审查结果
        """
        return self.execute({"file_path": file_path})
