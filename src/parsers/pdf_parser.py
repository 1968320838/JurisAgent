# PDF Parser Module
"""
PDF文档解析器
"""

from typing import Optional, Dict, Any
from pathlib import Path

from ..core.exceptions import DocumentParseError


class PDFParser:
    """PDF文档解析器"""

    def __init__(self):
        """初始化PDF解析器"""
        self._check_dependencies()

    def _check_dependencies(self):
        """检查依赖库是否可用"""
        try:
            import pdfplumber
            self._has_pdfplumber = True
        except ImportError:
            self._has_pdfplumber = False

        try:
            from PyPDF2 import PdfReader
            self._has_pypdf2 = True
        except ImportError:
            self._has_pypdf2 = False

        if not self._has_pdfplumber and not self._has_pypdf2:
            raise ImportError(
                "PDF解析需要安装 pdfplumber 或 PyPDF2。"
                "请运行: pip install pdfplumber 或 pip install PyPDF2"
            )

    def parse(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        解析PDF文件

        Args:
            file_path: PDF文件路径
            **kwargs: 额外参数

        Returns:
            Dict: 包含文本内容和元数据的字典
        """
        path = Path(file_path)
        if not path.exists():
            raise DocumentParseError(f"文件不存在", file_path)

        if path.suffix.lower() != ".pdf":
            raise DocumentParseError(f"不是PDF文件", file_path)

        # 优先使用pdfplumber（效果更好）
        if self._has_pdfplumber:
            return self._parse_with_pdfplumber(file_path)
        else:
            return self._parse_with_pypdf2(file_path)

    def _parse_with_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        """使用pdfplumber解析PDF"""
        import pdfplumber

        text_parts = []
        page_count = 0

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)

        return {
            "text": full_text,
            "page_count": page_count,
            "char_count": len(full_text),
            "parser": "pdfplumber"
        }

    def _parse_with_pypdf2(self, file_path: str) -> Dict[str, Any]:
        """使用PyPDF2解析PDF"""
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        page_count = len(reader.pages)

        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)

        return {
            "text": full_text,
            "page_count": page_count,
            "char_count": len(full_text),
            "parser": "PyPDF2"
        }

    def parse_from_bytes(self, file_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """
        从字节数据解析PDF

        Args:
            file_bytes: PDF文件的字节数据
            **kwargs: 额外参数

        Returns:
            Dict: 包含文本内容和元数据的字典
        """
        import io

        if self._has_pdfplumber:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                page_count = len(pdf.pages)
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

                full_text = "\n\n".join(text_parts)
                return {
                    "text": full_text,
                    "page_count": page_count,
                    "char_count": len(full_text),
                    "parser": "pdfplumber"
                }
        else:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            full_text = "\n\n".join(text_parts)
            return {
                "text": full_text,
                "page_count": page_count,
                "char_count": len(full_text),
                "parser": "PyPDF2"
            }
