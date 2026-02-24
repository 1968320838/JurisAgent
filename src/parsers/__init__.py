# Parsers Module
"""
文档解析模块
"""

from .pdf_parser import PDFParser
from .docx_parser import DocxParser

__all__ = ["PDFParser", "DocxParser"]
