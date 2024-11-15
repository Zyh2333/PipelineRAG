from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from nodes.base import BaseComponent
from storage.base import BaseStorage

logger = logging.getLogger(__name__)


class TextEmbeddingNode(BaseComponent):
    def __init__(self,
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 storage: Optional[BaseStorage] = None):
        """
        初始化文本嵌入节点

        Args:
            model_name: Sentence Transformer模型名称
            storage: 可选的存储实例
        """
        super().__init__(storage=storage)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def run(self,
            chunks: List[Dict[str, Any]] = None,
            total_chunks: int = 0,
            processed_documents: int = 0,
            chunk_stats: Dict[str, float] = None,
            processing_stats: Dict[str, Any] = None,
            query: Optional[str] = None,
            **kwargs) -> Tuple[Dict, Optional[str]]:
        """
        为文本生成嵌入向量
        支持两种模式:
        1. 查询模式: 输入query, 生成单个查询向量
        2. 索引模式: 输入chunks, 生成多个文档向量

        Args:
            chunks: 文本块列表，每个块包含:
                - text: 文本内容
                - source: 来源文件
                - metadata: 文档元数据
                - chunk_size: 块大小
            total_chunks: 总块数
            processed_documents: 处理的文档数
            chunk_stats: 块统计信息
            processing_stats: 处理统计信息
            query: 查询文本（用于查询模式）
            **kwargs: 其他参数

        Returns:
            Tuple[Dict, Optional[str]]: (输出字典, None)
        """
        if query is not None:
            # 查询模式 - 生成单个查询向量
            logger.info("进入查询模式")
            embedding = self.model.encode([query])[0]
            output = {
                "query": query,
                "query_embedding": embedding,
                "model_name": self.model_name,
                "dimension": len(embedding)
            }
        else:
            # 索引模式 - 为文档块生成向量
            if not chunks:
                logger.error("没有收到要处理的文本块")
                raise ValueError("没有收到要处理的文本块")

            logger.info(f"进入索引模式，处理 {len(chunks)} 个文本块")

            # 提取要编码的文本
            texts = [chunk['text'] for chunk in chunks]

            # 生成向量
            embeddings = self.model.encode(texts, show_progress_bar=True)

            # 组合结果
            result = []
            for chunk, embedding in zip(chunks, embeddings):
                enhanced_chunk = chunk.copy()  # 保留原始chunk的所有信息
                enhanced_chunk.update({
                    "embedding": embedding,
                    "embedding_dimension": len(embedding)
                })
                result.append(enhanced_chunk)

            # 准备输出
            output = {
                "embeddings": result,
                "model_name": self.model_name,
                "dimension": len(embeddings[0]) if len(result) > 0 else None,
                "total_vectors": len(result),
                "original_stats": {
                    "total_chunks": total_chunks,
                    "processed_documents": processed_documents,
                    "chunk_stats": chunk_stats,
                    "processing_stats": processing_stats
                }
            }

            logger.info(f"向量生成完成:")
            logger.info(f"- 总向量数: {len(result)}")
            logger.info(f"- 向量维度: {output['dimension']}")

        # 保存输出
        self.save_output(output)

        return output, None