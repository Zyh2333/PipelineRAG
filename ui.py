import streamlit as st
import os
from pathlib import Path
from nodes.document.pdf_processor import PDFProcessorNode
from nodes.document.text_splitter import TextSplitterNode
from nodes.embedding.text_embedding import TextEmbeddingNode
from nodes.vector.vector_store_node import VectorStoreNode
from nodes.llm.openai import GPT3Node
from nodes.prompt.prompt_node import PromptNode
from pipelines.base import Pipeline

# 设置页面配置
st.set_page_config(
    page_title="文档问答系统",
    page_icon="📚",
    layout="wide"
)

# 初始化会话状态
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'indexing_pipeline' not in st.session_state:
    st.session_state.indexing_pipeline = None
if 'query_pipeline' not in st.session_state:
    st.session_state.query_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'question_asked' not in st.session_state:
    st.session_state.question_asked = False


def initialize_indexing_pipeline():
    """初始化文档处理pipeline"""
    pipeline = Pipeline()

    pdf_processor = PDFProcessorNode()
    text_splitter = TextSplitterNode(
        chunk_size=512,
        chunk_overlap=50,
    )
    embedding_node = TextEmbeddingNode(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_store = VectorStoreNode(
        index_file_path="data/faiss_index.bin",
        mapping_file_path="data/text_mapping.pkl",
        dimension=384
    )

    pipeline.add_node(component=pdf_processor, name="PDFProcessor", inputs=["File"])
    pipeline.add_node(component=text_splitter, name="TextSplitter", inputs=["PDFProcessor"])
    pipeline.add_node(component=embedding_node, name="TextEmbedding", inputs=["TextSplitter"])
    pipeline.add_node(component=vector_store, name="VectorStore", inputs=["TextEmbedding"])

    return pipeline


def initialize_query_pipeline():
    """初始化问答pipeline"""
    pipeline = Pipeline()

    embedding_node = TextEmbeddingNode(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
    )
    vector_store = VectorStoreNode(
        index_file_path="data/faiss_index.bin",
        mapping_file_path="data/text_mapping.pkl",
        dimension=384,
    )
    prompt_node = PromptNode()
    gpt_node = GPT3Node(
        api_key="Bearer fk220173-cGW7fQeV2jkHUDlbePBEd1YyhXVWEFnh",
    )

    pipeline.add_node(component=embedding_node, name="TextEmbedding", inputs=["Query"])
    pipeline.add_node(component=vector_store, name="VectorRetrieval", inputs=["TextEmbedding"])
    pipeline.add_node(component=prompt_node, name="PromptBuilder", inputs=["VectorRetrieval"])
    pipeline.add_node(component=gpt_node, name="GPTNode", inputs=["PromptBuilder"])

    return pipeline


def main():
    st.title("📚 智能文档问答系统")

    # 创建数据目录
    os.makedirs("data", exist_ok=True)

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")

        # 创建文件上传器
        uploaded_files = st.file_uploader(
            "上传PDF文件",
            type=['pdf'],
            accept_multiple_files=True
        )

        if uploaded_files:
            # 保存上传的文件
            pdf_files = []
            for uploaded_file in uploaded_files:
                file_path = f"doc/{uploaded_file.name}"
                os.makedirs("doc", exist_ok=True)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_files.append(file_path)

            st.session_state.processed_files = pdf_files
            st.success(f"上传了 {len(pdf_files)} 个PDF文件")

        # 显示上传的文件列表
        if st.session_state.processed_files:
            st.write("已上传的文件:")
            for file in st.session_state.processed_files:
                st.text(f"📄 {Path(file).name}")

        # 处理文档按钮
        if st.button("处理文档", disabled=not st.session_state.processed_files):
            with st.spinner("正在处理文档..."):
                try:
                    if not st.session_state.indexing_pipeline:
                        st.session_state.indexing_pipeline = initialize_indexing_pipeline()
                    st.session_state.indexing_pipeline.run(file_paths=st.session_state.processed_files)
                    st.success("文档处理完成！")

                    # 初始化查询pipeline
                    if not st.session_state.query_pipeline:
                        st.session_state.query_pipeline = initialize_query_pipeline()
                except Exception as e:
                    st.error(f"处理文档时出错: {str(e)}")

    # 主界面
    st.header("💬 问答系统")

    # 检查是否已处理文档
    if not os.path.exists("data/faiss_index.bin"):
        st.warning("请先上传并处理文档后再开始提问")
        return

    # 问答区域
    if not st.session_state.query_pipeline:
        st.session_state.query_pipeline = initialize_query_pipeline()

    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for qa in st.session_state.chat_history:
            st.write(f"👤 **问题**: {qa['question']}")
            st.write(f"🤖 **回答**: {qa['answer']}")
            st.write("---")

    # 问题输入区域
    with st.form(key='qa_form'):
        query = st.text_input("请输入您的问题:", placeholder="例如：产品有什么特点？")
        submit_button = st.form_submit_button("发送问题")

        if submit_button and query:
            try:
                with st.spinner("AI思考中..."):
                    result = st.session_state.query_pipeline.run(query=query)
                    answer = result.get('response', '抱歉，我无法回答这个问题。')

                    # 添加到聊天历史
                    st.session_state.chat_history.append({
                        'question': query,
                        'answer': answer
                    })

                st.rerun()

            except Exception as e:
                st.error(f"处理查询时出错: {str(e)}")


if __name__ == "__main__":
    main()