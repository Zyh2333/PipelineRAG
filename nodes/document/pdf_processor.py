# nodes/document/pdf_processor.py
import os
import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from nodes.base import BaseComponent
from storage.base import BaseStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessorNode(BaseComponent):
    """增强的PDF处理节点"""

    def __init__(self,
                 storage: Optional[BaseStorage] = None,
                 min_line_length: int = 10,
                 encoding_check: bool = True):
        """
        初始化PDF处理节点

        Args:
            storage: 可选的存储实例
            min_line_length: 最小行长度，小于此长度的行会被过滤
            encoding_check: 是否检查和处理编码问题
        """
        super().__init__(storage=storage)
        self.min_line_length = min_line_length
        self.encoding_check = encoding_check

    def run(self, file_paths: List[str], **kwargs) -> Tuple[Dict, Optional[str]]:
        """
        处理PDF文件列表

        Args:
            file_paths: PDF文件路径列表

        Returns:
            Tuple[Dict, Optional[str]]: (输出字典, None)
        """
        all_texts = []
        processed_files = []
        failed_files = []

        total_files = len(file_paths)
        for index, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"处理文件 [{index}/{total_files}]: {Path(file_path).name}")

                # 检查文件是否存在和是否为PDF
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"找不到文件: {file_path}")

                if not file_path.lower().endswith('.pdf'):
                    logger.warning(f"跳过非PDF文件: {file_path}")
                    continue

                # 处理PDF文件
                text, metadata = self._process_pdf(file_path)
                if text:
                    # 添加文件源信息
                    processed_text = {
                        'content': text,
                        'source': str(Path(file_path).name),
                        'metadata': metadata
                    }
                    all_texts.append(processed_text)
                    processed_files.append(file_path)
                    logger.info(f"成功处理文件: {Path(file_path).name}")

            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {str(e)}")
                failed_files.append((file_path, str(e)))
                continue

        # 准备输出
        output = {
            "chunks": all_texts,
            "processed_files": processed_files,
            "failed_files": failed_files,
            "total_files": total_files,
            "successful_files": len(processed_files),
            "failed_count": len(failed_files)
        }

        # 输出处理统计
        logger.info(f"\n处理完成:")
        logger.info(f"- 总文件数: {total_files}")
        logger.info(f"- 成功处理: {len(processed_files)}")
        logger.info(f"- 处理失败: {len(failed_files)}")

        if failed_files:
            logger.info("\n失败的文件:")
            for file_path, error in failed_files:
                logger.info(f"- {Path(file_path).name}: {error}")

        # 保存输出
        self.save_output(output)

        return output, None

    def _process_pdf(self, file_path: str) -> Tuple[str, Dict]:
        """
        处理单个PDF文件

        Args:
            file_path: PDF文件路径

        Returns:
            Tuple[str, Dict]: (提取的文本内容, 元数据)
        """
        doc = fitz.open(file_path)
        text_parts = []

        try:
            # 提取元数据
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
                'page_count': len(doc),
                'file_name': Path(file_path).name
            }

            for page_num, page in enumerate(doc, 1):
                try:
                    # 提取文本
                    text = page.get_text()

                    # 处理编码问题（如果启用）
                    if self.encoding_check:
                        text = self._handle_encoding(text)

                    # 处理文本
                    lines = [line.strip() for line in text.split('\n')]
                    lines = [line for line in lines if len(line) >= self.min_line_length]

                    if lines:
                        # 添加页码信息
                        text_parts.append(f"[第{page_num}页]\n" + '\n'.join(lines))

                except Exception as e:
                    logger.warning(f"处理第 {page_num} 页时出错: {str(e)}")
                    continue

        finally:
            doc.close()

        return '\n\n'.join(text_parts), metadata

    def _handle_encoding(self, text: str) -> str:
        """
        处理文本编码问题

        Args:
            text: 输入文本

        Returns:
            str: 处理后的文本
        """
        try:
            # 处理常见的编码问题
            if not isinstance(text, str):
                text = text.decode('utf-8', errors='ignore')

            # 替换特殊字符
            text = text.replace('\x00', '')  # 删除空字符
            text = text.replace('\ufeff', '')  # 删除BOM

            return text.strip()

        except Exception as e:
            logger.warning(f"处理编码时出错: {str(e)}")
            return text.strip()