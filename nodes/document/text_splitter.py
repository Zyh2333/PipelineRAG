from typing import List, Dict, Optional, Tuple, Any
import logging
from nodes.base import BaseComponent
from storage.base import BaseStorage

logger = logging.getLogger(__name__)


class TextSplitterNode(BaseComponent):
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separator: str = "\n",
                 storage: Optional[BaseStorage] = None):
        super().__init__(storage=storage)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def run(self,
            chunks: List[Dict[str, Any]] = None,
            processed_files: List[str] = None,
            failed_files: List[tuple] = None,
            total_files: int = 0,
            successful_files: int = 0,
            failed_count: int = 0,
            **kwargs) -> Tuple[Dict, Optional[str]]:
        """
        处理来自PDF处理器的输入并生成文本块

        Args:
            chunks: PDF处理器输出的原始文档块列表，每个块包含：
                - content: 文本内容
                - source: 来源文件
                - metadata: 文档元数据
            processed_files: 成功处理的文件列表
            failed_files: 处理失败的文件和错误信息
            total_files: 总文件数
            successful_files: 成功处理的文件数
            failed_count: 失败的文件数
            **kwargs: 其他参数

        Returns:
            Tuple[Dict, Optional[str]]: (输出字典, None)
        """
        if not chunks:
            logger.error("没有收到有效的chunks数据")
            raise ValueError("没有收到有效的chunks数据")

        logger.info(f"开始处理文档块，输入统计:")
        logger.info(f"- 总文件数: {total_files}")
        logger.info(f"- 成功文件: {successful_files}")
        logger.info(f"- 失败文件: {failed_count}")

        all_chunks = []
        processed_docs = 0
        total_chunks = 0

        # 处理每个文档块
        for doc in chunks:
            try:
                # 提取文档信息
                content = doc.get('content', '')
                source = doc.get('source', 'unknown')
                metadata = doc.get('metadata', {})

                # 分割文本
                text_chunks = self._split_text(content)

                # 为每个块添加元数据
                for chunk_text in text_chunks:
                    chunk = {
                        'text': chunk_text,
                        'source': source,
                        'metadata': metadata,
                        'chunk_size': len(chunk_text),
                        'original_file': source
                    }
                    all_chunks.append(chunk)

                total_chunks += len(text_chunks)
                processed_docs += 1
                logger.debug(f"成功处理文档: {source}, 生成 {len(text_chunks)} 个文本块")

            except Exception as e:
                logger.error(f"处理文档时出错 {source}: {str(e)}")
                continue

        # 准备输出
        output = {
            "chunks": all_chunks,
            "total_chunks": total_chunks,
            "processed_documents": processed_docs,
            "chunk_stats": {
                "avg_chunk_size": sum(len(c['text']) for c in all_chunks) / len(all_chunks) if all_chunks else 0,
                "max_chunk_size": max(len(c['text']) for c in all_chunks) if all_chunks else 0,
                "min_chunk_size": min(len(c['text']) for c in all_chunks) if all_chunks else 0
            },
            "processing_stats": {
                "input_files": total_files,
                "successful_files": successful_files,
                "failed_files": failed_count,
                "processed_files": processed_files
            }
        }

        logger.info(f"文本分割完成:")
        logger.info(f"- 处理文档数: {processed_docs}")
        logger.info(f"- 生成文本块: {total_chunks}")
        logger.info(f"- 平均块大小: {output['chunk_stats']['avg_chunk_size']:.2f} 字符")

        # 保存输出
        self.save_output(output)

        return output, None

    def _split_text(self, text: str) -> List[str]:
        """文本分块的具体实现"""
        if len(text) <= self.chunk_size:
            return [text]

        segments = text.split(self.separator)
        current_chunk = []
        current_length = 0
        chunks = []

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            segment_length = len(segment)

            if segment_length > self.chunk_size:
                if current_chunk:
                    chunks.append(self.separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                sub_chunks = self._split_large_segment(segment)
                chunks.extend(sub_chunks)
                continue

            if current_length + segment_length > self.chunk_size:
                chunks.append(self.separator.join(current_chunk))
                current_chunk = [segment]
                current_length = segment_length
            else:
                current_chunk.append(segment)
                current_length += segment_length

        if current_chunk:
            chunks.append(self.separator.join(current_chunk))

        # 添加重叠
        final_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_start = max(0, len(prev_chunk) - self.chunk_overlap)
                chunk = prev_chunk[overlap_start:] + self.separator + chunk
            final_chunks.append(chunk)

        return final_chunks

    def _split_large_segment(self, segment: str) -> List[str]:
        """处理超长段落"""
        import re
        sentences = re.split('([。！？.!?])', segment)
        current_chunk = []
        current_length = 0
        chunks = []

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            sentence_length = len(sentence)

            if sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                for j in range(0, len(sentence), self.chunk_size):
                    chunks.append(sentence[j:j + self.chunk_size])
                continue

            if current_length + sentence_length > self.chunk_size:
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(''.join(current_chunk))

        return chunks