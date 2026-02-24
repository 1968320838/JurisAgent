# Retriever Module
"""
法条检索器
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from .embeddings import EmbeddingEngine, get_embedding_engine
from .vector_store import VectorStore


class LegalRetriever:
    """
    法律文档检索器

    功能：
    1. 法条检索
    2. 案例检索
    3. 混合检索（向量 + 关键词）
    """

    def __init__(
        self,
        vector_store: VectorStore = None,
        embedding_engine: EmbeddingEngine = None,
        api_key: str = None,
        persist_directory: str = "./data/vector_db"
    ):
        """
        初始化检索器

        Args:
            vector_store: 向量存储实例
            embedding_engine: 嵌入引擎实例
            api_key: API密钥
            persist_directory: 向量存储目录
        """
        self.api_key = api_key

        # 初始化嵌入引擎
        if embedding_engine:
            self.embedding_engine = embedding_engine
        else:
            self.embedding_engine = get_embedding_engine(
                api_key=api_key,
                engine_type="zhipu"
            )

        # 初始化向量存储
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore(persist_directory=persist_directory)

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        doc_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        检索相关法律文档

        Args:
            query: 查询文本
            n_results: 返回结果数量
            doc_type: 文档类型过滤 (law/case)

        Returns:
            List[Dict]: 检索结果列表
        """
        # 生成查询向量
        query_embedding = self.embedding_engine.embed_single(query)

        # 构建过滤条件
        where_filter = None
        if doc_type:
            where_filter = {"doc_type": doc_type}

        # 执行检索
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where_filter
        )

        # 格式化结果
        formatted_results = []
        for i in range(len(results["ids"])):
            formatted_results.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
                "score": 1 - results["distances"][i]  # 转换为相似度
            })

        return formatted_results

    def retrieve_laws(self, query: str, n_results: int = 5) -> List[Dict]:
        """检索法条"""
        return self.retrieve(query, n_results, doc_type="law")

    def retrieve_cases(self, query: str, n_results: int = 3) -> List[Dict]:
        """检索案例"""
        return self.retrieve(query, n_results, doc_type="case")

    def retrieve_with_context(
        self,
        query: str,
        n_results: int = 5,
        max_chars: int = 2000
    ) -> str:
        """
        检索并组装上下文

        Args:
            query: 查询文本
            n_results: 返回结果数量
            max_chars: 最大字符数

        Returns:
            str: 组装后的上下文文本
        """
        results = self.retrieve(query, n_results)

        context_parts = []
        total_chars = 0

        for i, result in enumerate(results, 1):
            text = result["text"]
            metadata = result["metadata"]

            # 构建上下文片段
            source = metadata.get("law_name", metadata.get("title", "未知来源"))
            article = metadata.get("article_number", "")

            part = f"[{i}] {source}"
            if article:
                part += f" {article}"
            part += f"\n{text}\n"

            # 检查长度限制
            if total_chars + len(part) > max_chars:
                break

            context_parts.append(part)
            total_chars += len(part)

        return "\n".join(context_parts)

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        索引文档

        Args:
            documents: 文档列表，每个包含 text, metadata, id
            batch_size: 批处理大小

        Returns:
            int: 索引的文档数量
        """
        indexed = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # 生成嵌入
            texts = [doc["text"] for doc in batch]
            embeddings = self.embedding_engine.embed(texts)

            # 准备数据
            ids = [doc["id"] for doc in batch]

            # 添加到向量存储
            self.vector_store.add_documents(
                documents=[{"text": d["text"], "metadata": d.get("metadata", {})} for d in batch],
                embeddings=embeddings,
                ids=ids
            )

            indexed += len(batch)

        return indexed

    def get_stats(self) -> Dict:
        """获取检索器统计信息"""
        return {
            "total_documents": self.vector_store.count(),
            "embedding_dimension": self.embedding_engine.embedding_dim
        }


class RAGEnhancedQA:
    """
    RAG增强的问答系统

    将检索结果与LLM结合，生成基于事实的回答
    """

    def __init__(
        self,
        retriever: LegalRetriever,
        llm_client
    ):
        """
        初始化RAG问答系统

        Args:
            retriever: 法律检索器
            llm_client: LLM客户端
        """
        self.retriever = retriever
        self.llm = llm_client

    def answer(
        self,
        question: str,
        n_context: int = 5,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        基于检索结果回答问题

        Args:
            question: 用户问题
            n_context: 检索的上下文数量
            temperature: LLM温度

        Returns:
            Dict: 包含回答和来源的结果
        """
        # 1. 检索相关法条
        context = self.retriever.retrieve_with_context(question, n_context)

        if not context:
            return {
                "answer": "抱歉，未找到相关的法律依据。请尝试更具体的问题描述。",
                "sources": [],
                "has_context": False
            }

        # 2. 构建提示词
        prompt = f"""请基于以下法律条文回答用户问题。

【重要规则】
1. 只使用提供的法律条文作为依据
2. 每个观点必须标注来源，格式：[来源编号]
3. 如果提供的法律条文不足以回答问题，请明确说明
4. 不要编造法律条文

【相关法律条文】
{context}

【用户问题】
{question}

【回答】
请先给出结论，然后说明法律依据，最后提供具体建议。"""

        # 3. 调用LLM
        messages = [
            {"role": "system", "content": "你是一位专业的法律顾问，请基于提供的法律条文准确回答问题。"},
            {"role": "user", "content": prompt}
        ]

        answer = self.llm.chat(messages, temperature=temperature)

        # 4. 获取来源信息
        sources = self.retriever.retrieve(question, n_context)

        return {
            "answer": answer,
            "sources": sources,
            "has_context": True
        }
