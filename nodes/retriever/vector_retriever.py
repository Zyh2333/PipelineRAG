# 1. nodes/retriever/vector_retriever.py
from typing import Dict, Optional, List, Tuple
import numpy as np
from nodes.base import BaseComponent
from storage.base import BaseStorage
from nodes.embedding.text_embedding import TextEmbeddingNode

class VectorRetrieverNode(BaseComponent):
    """向量检索节点"""
    def __init__(self,
                 vector_store_node,
                 top_k: int = 3,
                 storage: Optional[BaseStorage] = None):
        """
        初始化向量检索节点

        Args:
            vector_store_node: 向量存储节点实例
            top_k: 返回的最相关文档数量
            storage: 可选的存储实例
        """
        super().__init__(storage=storage)
        self.vector_store = vector_store_node
        self.embedding_model = TextEmbeddingNode()
        self.top_k = top_k

    def run(self, query: str, **kwargs) -> Tuple[Dict, Optional[str]]:
        """
        检索与查询最相关的文档

        Args:
            query: 查询文本

        Returns:
            Tuple[Dict, Optional[str]]: (输出字典, None)
        """
        # 生成查询的嵌入向量
        query_embedding_output, _ = self.embedding_model.run([query])
        query_embedding = query_embedding_output["embeddings"][0]["embedding"]

        # 搜索最相关的文档
        results = self.vector_store.search(query_embedding, self.top_k)

        if not results:
            context = "未找到相关文档。"
        else:
            # 组合相关文档文本
            context_parts = []
            for i, result in enumerate(results, 1):
                score = result.get("score", 0.0)
                text = result.get("text", "").strip()
                if text:
                    context_parts.append(f"{i}. (相关度: {score:.3f}) {text}")
            context = "\n".join(context_parts)

        # 构建提示
        prompt = self._build_prompt(query, context)

        output = {
            "original_query": query,
            "context": context,
            "search_results": results,
            "query": prompt
        }

        # 保存输出
        self.save_output(output)

        return output, None

    def _build_prompt(self, query: str, context: str) -> str:
        """
        构建完整的提示文本

        Args:
            query: 原始查询
            context: 检索到的上下文

        Returns:
            str: 格式化的提示文本
        """
        return f"""请基于以下参考信息回答问题。
如果参考信息中没有相关内容，请直接说明无法从给定信息中找到答案。
请保持回答的客观性，不要添加参考信息以外的内容。

参考信息:
{context}

问题: {query}
"""