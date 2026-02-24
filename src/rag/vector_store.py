# Vector Store Module
"""
向量存储模块
使用Chroma作为向量数据库
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json


class VectorStore:
    """
    向量存储

    使用Chroma进行向量存储和检索
    """

    def __init__(
        self,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "legal_documents"
    ):
        """
        初始化向量存储

        Args:
            persist_directory: 持久化目录
            collection_name: 集合名称
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self._init_chroma()

    def _init_chroma(self):
        """初始化Chroma客户端"""
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )

            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "法律文档向量存储"}
            )

        except ImportError:
            raise ImportError(
                "向量存储需要安装 chromadb。"
                "请运行: pip install chromadb"
            )

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: List[str]
    ) -> None:
        """
        添加文档到向量存储

        Args:
            documents: 文档列表，每个文档包含 text 和 metadata
            embeddings: 嵌入向量列表
            ids: 文档ID列表
        """
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def add_single_document(
        self,
        text: str,
        embedding: List[float],
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        添加单个文档

        Args:
            text: 文档文本
            embedding: 嵌入向量
            doc_id: 文档ID
            metadata: 元数据
        """
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Dict = None
    ) -> Dict[str, Any]:
        """
        搜索相似文档

        Args:
            query_embedding: 查询向量
            n_results: 返回结果数量
            where: 元数据过滤条件

        Returns:
            Dict: 搜索结果
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def search_by_text(
        self,
        query_text: str,
        embedding_engine,
        n_results: int = 5,
        where: Dict = None
    ) -> Dict[str, Any]:
        """
        通过文本搜索

        Args:
            query_text: 查询文本
            embedding_engine: 嵌入引擎
            n_results: 返回结果数量
            where: 元数据过滤条件

        Returns:
            Dict: 搜索结果
        """
        query_embedding = embedding_engine.embed_single(query_text)
        return self.search(query_embedding, n_results, where)

    def delete(self, ids: List[str] = None, where: Dict = None) -> None:
        """
        删除文档

        Args:
            ids: 要删除的文档ID列表
            where: 元数据过滤条件
        """
        self.collection.delete(ids=ids, where=where)

    def count(self) -> int:
        """返回文档数量"""
        return self.collection.count()

    def get_collection_info(self) -> Dict:
        """获取集合信息"""
        return {
            "name": self.collection_name,
            "count": self.count(),
            "metadata": self.collection.metadata
        }

    def clear(self) -> None:
        """清空集合"""
        # 删除并重建集合
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "法律文档向量存储"}
        )


class SimpleVectorStore:
    """
    简单向量存储（不依赖外部库，用于测试）
    """

    def __init__(self):
        self.documents: List[Dict] = []
        self.embeddings: List[List[float]] = []
        self.ids: List[str] = []

    def add_documents(
        self,
        documents: List[Dict],
        embeddings: List[List[float]],
        ids: List[str]
    ) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.ids.extend(ids)

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> Dict[str, Any]:
        """使用余弦相似度搜索"""
        import numpy as np

        query = np.array(query_embedding)
        similarities = []

        for i, emb in enumerate(self.embeddings):
            emb_array = np.array(emb)
            # 余弦相似度
            similarity = np.dot(query, emb_array) / (
                np.linalg.norm(query) * np.linalg.norm(emb_array)
            )
            similarities.append((i, similarity))

        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回top-k
        top_k = similarities[:n_results]

        return {
            "ids": [self.ids[i] for i, _ in top_k],
            "documents": [self.documents[i]["text"] for i, _ in top_k],
            "metadatas": [self.documents[i].get("metadata", {}) for i, _ in top_k],
            "distances": [s for _, s in top_k]
        }

    def count(self) -> int:
        return len(self.documents)
