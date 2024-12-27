# main.py
import os
from pathlib import Path
from nodes.document.pdf_processor import PDFProcessorNode
from nodes.document.text_splitter import TextSplitterNode
from nodes.embedding.text_embedding import TextEmbeddingNode
from nodes.vector.vector_store_node import VectorStoreNode
from nodes.llm.openai import GPT3Node
from storage.excel_storage import ExcelStorage
from pipelines.base import Pipeline
from nodes.prompt.prompt_node import PromptNode
import gradio as gr


def get_all_pdf_files(root_dir: str) -> list:
    """递归获取所有PDF文件

    Args:
        root_dir: 根目录路径

    Returns:
        list: PDF文件路径列表
    """
    pdf_files = []
    root_path = Path(root_dir)

    for file_path in root_path.rglob("*.pdf"):
        try:
            # 检查文件是否可以访问
            if file_path.is_file():
                # 将路径转换为规范形式并添加到列表
                pdf_files.append(str(file_path.resolve()))
        except (PermissionError, OSError) as e:
            print(f"无法访问文件 {file_path}: {str(e)}")
            continue

    return pdf_files


def process_documents():
    """处理文档的索引pipeline"""
    # 创建索引pipeline
    indexing_pipeline = Pipeline()

    # 配置节点
    pdf_processor = PDFProcessorNode(
        # storage=ExcelStorage("data/pdf_processing.xlsx")
    )

    text_splitter = TextSplitterNode(
        chunk_size=512,
        chunk_overlap=50,
        # storage=ExcelStorage("data/text_chunks.xlsx")
    )

    embedding_node = TextEmbeddingNode(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
        # storage=ExcelStorage("data/embeddings.xlsx")
    )

    vector_store = VectorStoreNode(
        index_file_path="data/faiss_index.bin",
        mapping_file_path="data/text_mapping.pkl",
        dimension=384
    )

    # 添加节点到pipeline
    indexing_pipeline.add_node(component=pdf_processor, name="PDFProcessor", inputs=["File"])
    indexing_pipeline.add_node(component=text_splitter, name="TextSplitter", inputs=["PDFProcessor"])
    indexing_pipeline.add_node(component=embedding_node, name="TextEmbedding", inputs=["TextSplitter"])
    indexing_pipeline.add_node(component=vector_store, name="VectorStore", inputs=["TextEmbedding"])

    return indexing_pipeline


def query_documents():
    """创建文档查询pipeline"""
    query_pipeline = Pipeline()

    embedding_node = TextEmbeddingNode(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
    )
    vector_store = VectorStoreNode(
        index_file_path="data/faiss_index.bin",
        mapping_file_path="data/text_mapping.pkl",
        dimension=384,
    )
    prompt_node = PromptNode(
        # storage=ExcelStorage("data/prompts.xlsx")
    )
    gpt_node = GPT3Node(
        # api_key="Bearer 默认api2d对应api key", #默认采用api2d对应api key
        api_key="sk-uz2K0lhafPFynE77BBX5v0adDXDzxJWir05jrlPCPQoMslp9",
        model="gpt-3.5-turbo"#默认采用api2d对应api key
        # storage=ExcelStorage("data/gpt_responses.xlsx")
    )

    # 添加节点到查询pipeline
    query_pipeline.add_node(component=embedding_node, name="TextEmbedding", inputs=["Query"])
    query_pipeline.add_node(component=vector_store, name="VectorRetrieval", inputs=["TextEmbedding"])
    query_pipeline.add_node(component=prompt_node, name="PromptBuilder", inputs=["VectorRetrieval"])
    query_pipeline.add_node(component=gpt_node, name="GPTNode", inputs=["PromptBuilder"])

    return query_pipeline


def answer_question(query: str, history: list) -> tuple:
    """给定一个问题，返回AI的回答并更新历史记录"""
    if history is None:
        history = []
    try:
        result = query_pipeline.run(query=query)
        response = result['response']
        history.append((query, response))
        return history, history  # 返回更新后的对话历史两次，以匹配输入输出
    except Exception as e:
        error_message = f"处理查询时出错: {str(e)}"
        history.append((query, error_message))
        return history, history  # 同样地，这里也返回两次

def main():
    # 设置基础存储目录
    os.makedirs("data", exist_ok=True)

    # 获取所有PDF文件
    pdf_dir = "doc"  # 你的文档根目录
    pdf_files = get_all_pdf_files(pdf_dir)

    if not pdf_files:
        print(f"在 {pdf_dir} 目录下未找到PDF文件")
        return

    print(f"找到 {len(pdf_files)} 个PDF文件:")

    # 处理文档
    print("\n开始处理文档...")
    indexing_pipeline = process_documents()
    indexing_pipeline.run(file_paths=pdf_files)
    print("文档处理完成！\n")

    global query_pipeline
    query_pipeline = query_documents()

    # 创建Gradio接口
    iface = gr.Interface(
        fn=answer_question,
        inputs=[gr.Textbox(lines=2, placeholder="输入您的问题..."), "state"],
        outputs=["chatbot", "state"],
        title="智能问答系统",
        description="输入您的问题，获取关于已处理文档的信息。",
    )

    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
    finally:
        print("\n程序已退出")