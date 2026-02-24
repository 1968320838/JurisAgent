# Embeddings Module
"""
向量嵌入模块
支持多种嵌入引擎
"""

from typing import List, Optional
import hashlib


class EmbeddingEngine:
    """
    向量嵌入引擎

    支持多种后端：
    - zhipu: 智谱AI嵌入API
    - local: 本地TF/PyTorch模型
    - mock: 用于测试的模拟嵌入
    """

    def __init__(self, engine_type: str = "zhipu", api_key: str = None):
        """
        初始化嵌入引擎

        Args:
            engine_type: 引擎类型 (zhipu/local/mock)
            api_key: API密钥（用于智谱AI）
        """
        self.engine_type = engine_type
        self.api_key = api_key
        self._embedding_dim = 1024  # 智谱AI嵌入维度

        if engine_type == "zhipu":
            self._init_zhipu()
        elif engine_type == "local":
            self._init_local()
        elif engine_type == "mock":
            self._init_mock()
        else:
            raise ValueError(f"不支持的嵌入引擎类型: {engine_type}")

    def _init_zhipu(self):
        """初始化智谱AI嵌入引擎"""
        try:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=self.api_key)
            self._model = "embedding-3"
        except ImportError:
            # 如果没有安装zhipuai，回退到requests
            self.client = None
            self._api_url = "https://open.bigmodel.cn/api/paas/v4/embeddings"

    def _init_local(self):
        """初始化本地嵌入引擎"""
        try:
            from sentence_transformers import SentenceTransformer
            # 使用中文优化的模型
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self._embedding_dim = 384
        except ImportError:
            raise ImportError(
                "本地嵌入需要安装 sentence-transformers。"
                "请运行: pip install sentence-transformers"
            )

    def _init_mock(self):
        """初始化模拟嵌入引擎（用于测试）"""
        import numpy as np
        self._np = np
        self._embedding_dim = 128

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        生成文本嵌入向量

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if self.engine_type == "zhipu":
            return self._embed_zhipu(texts)
        elif self.engine_type == "local":
            return self._embed_local(texts)
        else:
            return self._embed_mock(texts)

    def embed_single(self, text: str) -> List[float]:
        """
        生成单个文本的嵌入向量

        Args:
            text: 文本

        Returns:
            List[float]: 嵌入向量
        """
        return self.embed([text])[0]

    def _embed_zhipu(self, texts: List[str]) -> List[List[float]]:
        """使用智谱AI生成嵌入"""
        if self.client:
            # 使用SDK
            response = self.client.embeddings.create(
                model=self._model,
                input=texts
            )
            return [item.embedding for item in response.data]
        else:
            # 使用requests
            import requests
            import json

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self._model,
                "input": texts
            }

            response = requests.post(
                self._api_url,
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            return [item["embedding"] for item in result["data"]]

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """使用本地模型生成嵌入"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def _embed_mock(self, texts: List[str]) -> List[List[float]]:
        """生成模拟嵌入（用于测试）"""
        # 基于文本哈希生成确定性向量
        vectors = []
        for text in texts:
            # 使用哈希生成确定性向量
            hash_bytes = hashlib.md5(text.encode()).digest()
            np_random = self._np.random.RandomState(int.from_bytes(hash_bytes[:4], 'big'))
            vector = np_random.randn(self._embedding_dim).tolist()
            vectors.append(vector)
        return vectors

    @property
    def embedding_dim(self) -> int:
        """返回嵌入向量维度"""
        return self._embedding_dim


def get_embedding_engine(api_key: str = None, engine_type: str = "zhipu") -> EmbeddingEngine:
    """
    获取嵌入引擎实例

    Args:
        api_key: API密钥
        engine_type: 引擎类型

    Returns:
        EmbeddingEngine: 嵌入引擎实例
    """
    return EmbeddingEngine(engine_type=engine_type, api_key=api_key)
