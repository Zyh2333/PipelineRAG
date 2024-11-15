import os
import faiss
import numpy as np
import pickle
import logging
from typing import Dict, Optional, Tuple, List, Any
from nodes.base import BaseComponent
from storage.base import BaseStorage

logger = logging.getLogger(__name__)


class VectorStoreNode(BaseComponent):
    def __init__(self,
                 index_file_path: str,
                 mapping_file_path: str,
                 dimension: int = 384,
                 top_k: int = 3,
                 storage: Optional[BaseStorage] = None):
        """
        初始化向量存储节点

        Args:
            index_file_path: FAISS索引文件路径
            mapping_file_path: ID到文本的映射文件路径
            dimension: 向量维度
            top_k: 检索时返回的最相关文档数量
            storage: 可选的存储实例
        """
        super().__init__(storage=storage)
        self.index_file_path = index_file_path
        self.mapping_file_path = mapping_file_path
        self.dimension = dimension
        self.top_k = top_k
        self._initialize_index()

    def run(self,
            embeddings: Optional[List[Dict[str, Any]]] = None,
            model_name: Optional[str] = None,
            dimension: Optional[int] = None,
            total_vectors: Optional[int] = None,
            original_stats: Optional[Dict] = None,
            query: Optional[str] = None,
            query_embedding: Optional[np.ndarray] = None,
            **kwargs) -> Tuple[Dict, Optional[str]]:
        """
        处理输入数据。支持两种模式:
        1. 索引模式: 接收embeddings并存储
        2. 查询模式: 接收query_embedding并检索

        Args:
            embeddings: 包含向量和元数据的列表，每项包含:
                - text: 文本内容
                - source: 来源文件
                - metadata: 文档元数据
                - chunk_size: 块大小
                - embedding: 向量
                - embedding_dimension: 向量维度
            model_name: 使用的嵌入模型名称
            dimension: 向量维度
            total_vectors: 总向量数
            original_stats: 原始文档统计信息
            query: 查询文本
            query_embedding: 查询向量
            **kwargs: 其他参数

        Returns:
            Tuple[Dict, Optional[str]]: (输出字典, None)
        """
        # 检查输入模式
        if embeddings is not None:
            # 索引模式
            logger.info("进入索引模式")
            return self._handle_storage(embeddings, model_name, original_stats)
        elif query_embedding is not None:
            # 查询模式
            logger.info("进入查询模式")
            return self._handle_query(query, query_embedding)
        else:
            raise ValueError("无效的输入: 需要提供embeddings用于存储或query_embedding用于检索")

    def _handle_query(self, query: str, query_embedding: np.ndarray) -> Tuple[Dict, Optional[str]]:
        """处理查询请求"""
        try:
            if self.index.ntotal == 0:
                return {"status": "error", "message": "索引为空,请先添加文档"}, None

            # 准备查询向量
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_vector)

            # 执行检索
            scores, ids = self.index.search(query_vector, min(self.top_k, self.index.ntotal))

            # 准备结果
            results = []
            for score, idx in zip(scores[0], ids[0]):
                if idx >= 0:  # 有效的索引
                    doc_info = self.id_to_text[idx]
                    result = {
                        "text": doc_info["text"],
                        "score": float(score),
                        "source": doc_info["source"],
                        "metadata": doc_info.get("metadata", {}),
                        "chunk_size": doc_info.get("chunk_size", 0)
                    }
                    results.append(result)

            output = {
                "query": query,
                "results": results,
                "total_docs": self.index.ntotal
            }

            logger.info(f"检索完成: 找到 {len(results)} 个相关文档")
            self.save_output(output)
            return output, None

        except Exception as e:
            error_msg = f"检索时出错: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}, None

    def _handle_storage(self,
                        embeddings: List[Dict],
                        model_name: str,
                        original_stats: Dict) -> Tuple[Dict, Optional[str]]:
        """处理存储请求"""
        if not embeddings:
            return {"status": "error", "message": "没有要存储的向量"}, None

        try:
            vectors = []
            current_id = len(self.id_to_text)

            # 收集向量和文本
            for item in embeddings:
                if not all(k in item for k in ["text", "embedding", "source"]):
                    raise ValueError(f"无效的embedding数据格式: {item.keys()}")

                # 存储向量
                vectors.append(item["embedding"])

                # 存储完整的文档信息
                self.id_to_text[current_id] = {
                    "text": item["text"],
                    "source": item["source"],
                    "metadata": item.get("metadata", {}),
                    "chunk_size": item.get("chunk_size", 0),
                }
                current_id += 1

            # 添加向量到索引
            vectors_array = np.array(vectors).astype('float32')
            faiss.normalize_L2(vectors_array)
            self.index.add(vectors_array)

            # 保存索引和映射
            self._save_index()

            output = {
                "status": "success",
                "model_name": model_name,
                "vectors_added": len(vectors),
                "total_vectors": self.index.ntotal,
                "original_stats": original_stats,
                "index_file": self.index_file_path,
                "mapping_file": self.mapping_file_path
            }

            logger.info(f"向量存储完成:")
            logger.info(f"- 新增向量: {len(vectors)}")
            logger.info(f"- 总向量数: {self.index.ntotal}")
            logger.info(f"- 索引文件: {self.index_file_path}")

            self.save_output(output)
            return output, None

        except Exception as e:
            error_msg = f"存储向量时出错: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}, None

    def _initialize_index(self):
        """初始化或加载现有索引"""
        try:
            if os.path.exists(self.index_file_path) and os.path.exists(self.mapping_file_path):
                logger.info("加载现有索引和映射文件")
                self.index = faiss.read_index(self.index_file_path)
                with open(self.mapping_file_path, 'rb') as f:
                    self.id_to_text = pickle.load(f)
            else:
                logger.info(f"创建新的FAISS索引 (维度: {self.dimension})")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.id_to_text = {}
        except Exception as e:
            logger.error(f"初始化索引时出错: {str(e)}")
            raise

    def _save_index(self):
        """保存索引和映射到文件"""
        try:
            os.makedirs(os.path.dirname(self.index_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.mapping_file_path), exist_ok=True)

            faiss.write_index(self.index, self.index_file_path)
            with open(self.mapping_file_path, 'wb') as f:
                pickle.dump(self.id_to_text, f)
            logger.info("索引和映射文件已保存")
        except Exception as e:
            logger.error(f"保存索引文件时出错: {str(e)}")
            raise