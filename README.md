# Pipeline-RAG: 基于Pipeline架构的RAG系统框架

Pipeline-RAG是一个用于构建检索增强生成(RAG)系统的框架，采用高度模块化的Pipeline架构设计。该框架允许你通过组合不同的功能节点来构建自定义的RAG应用，具有高度的灵活性和可扩展性。

## 核心特点

- **Pipeline架构**: 基于有向无环图(DAG)的pipeline设计，支持灵活的节点编排和流程控制
- **模块化设计**: 所有功能都被封装为独立的Node，可以根据需求自由组合
- **严格的参数传递**: 基于显式参数声明的节点间数据传递机制
- **可扩展性**: 提供BaseComponent接口，轻松开发新的功能节点
- **完整状态管理**: 包含Pipeline状态管理和错误处理机制

## 架构设计

### Pipeline核心架构
```
Pipeline
├── Graph Management (DiGraph)
│   ├── Node Registration
│   ├── Edge Management 
│   └── Topology Sort
├── Flow Control
│   ├── Sequential Processing
│   ├── Parallel Processing
│   └── Conditional Branching
└── State Management
    ├── Input/Output Handling
    └── Error Management
```

### Node基础框架
```
BaseComponent
├── Interface
│   ├── run()
│   └── _dispatch_run()
├── Configuration
│   ├── set_config()
│   └── get_config()
└── Storage Interface
    ├── save_output()
    └── load_output()
```

## 节点定义和参数传递

### 1. 基本节点结构

所有节点必须继承自`BaseComponent`并实现`run`方法：

```python
from nodes.base import BaseComponent
from typing import Dict, Optional, Tuple

class CustomNode(BaseComponent):
    def __init__(self, storage: Optional[BaseStorage] = None, **kwargs):
        super().__init__(storage=storage)
        self.kwargs = kwargs
    
    def run(self, param1: str, param2: list, **kwargs) -> Tuple[Dict, Optional[str]]:
        # 处理逻辑
        result = {
            "output_key1": value1,
            "output_key2": value2
        }
        return result, None
```

### 2. 参数传递机制

上下游节点间的参数传递需要严格匹配：

```python
# 上游节点
class UpstreamNode(BaseComponent):
    def run(self, input_text: str, **kwargs) -> Tuple[Dict, Optional[str]]:
        result = {
            "processed_text": f"Processed: {input_text}",
            "metadata": {"timestamp": "2024-01-01"}
        }
        return result, None

# 下游节点 - 必须显式声明需要使用的参数
class DownstreamNode(BaseComponent):
    def run(self, 
            processed_text: str,    # 匹配上游输出的processed_text
            metadata: Dict,         # 匹配上游输出的metadata
            **kwargs) -> Tuple[Dict, Optional[str]]:
        return {"result": f"Final: {processed_text}"}, None
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 创建自定义Node

```python
from nodes.base import BaseComponent

class TextProcessingNode(BaseComponent):
    def __init__(self, storage: Optional[BaseStorage] = None):
        super().__init__(storage=storage)
        
    def run(self, text: str, **kwargs) -> Tuple[Dict, Optional[str]]:
        processed_text = text.upper()
        return {"processed_text": processed_text}, None
```

### 3. 构建Pipeline

```python
from pipelines.base import Pipeline

# 初始化pipeline
pipeline = Pipeline()

# 添加节点
pipeline.add_node(component=pdf_processor, name="PDFProcessor", inputs=["File"])
pipeline.add_node(component=text_splitter, name="TextSplitter", inputs=["PDFProcessor"])
pipeline.add_node(component=embedding_node, name="TextEmbedding", inputs=["TextSplitter"])
```

### 4. 运行Pipeline

```python
# 运行pipeline
result = pipeline.run(file_paths=["your_doc.pdf"])
```

## 内置节点类型

### 1. 文档处理节点
```python
class PDFProcessorNode(BaseComponent):
    def run(self, file_paths: List[str], **kwargs) -> Tuple[Dict, Optional[str]]:
        result = {
            "chunks": processed_chunks,
            "metadata": file_metadata
        }
        return result, None
```

### 2. 文本分割节点
```python
class TextSplitterNode(BaseComponent):
    def run(self, chunks: List[Dict], metadata: Dict, **kwargs) -> Tuple[Dict, Optional[str]]:
        result = {
            "text_chunks": split_chunks,
            "chunk_count": len(split_chunks)
        }
        return result, None
```

### 3. 向量化节点
```python
class TextEmbeddingNode(BaseComponent):
    def run(self, text_chunks: List[str], chunk_count: int, **kwargs) -> Tuple[Dict, Optional[str]]:
        result = {
            "embeddings": embedded_vectors,
            "dimension": vector_dim
        }
        return result, None
```

## 示例应用

### RAG系统示例

1. 索引Pipeline
```python
# 文档索引pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=pdf_processor, name="PDFProcessor", inputs=["File"])
indexing_pipeline.add_node(component=text_splitter, name="TextSplitter", inputs=["PDFProcessor"])
indexing_pipeline.add_node(component=embedding_node, name="TextEmbedding", inputs=["TextSplitter"])
indexing_pipeline.add_node(component=vector_store, name="VectorStore", inputs=["TextEmbedding"])
```

2. 查询Pipeline
```python
# 查询pipeline
query_pipeline = Pipeline()
query_pipeline.add_node(component=embedding_node, name="TextEmbedding", inputs=["Query"])
query_pipeline.add_node(component=vector_store, name="VectorRetrieval", inputs=["TextEmbedding"])
query_pipeline.add_node(component=prompt_node, name="PromptBuilder", inputs=["VectorRetrieval"])
query_pipeline.add_node(component=gpt_node, name="GPTNode", inputs=["PromptBuilder"])
```

## 高级特性

### 1. 动态Pipeline构建
```python
pipeline_config = {
    "nodes": [
        {"name": "node1", "type": "ProcessorA", "inputs": ["input"]},
        {"name": "node2", "type": "ProcessorB", "inputs": ["node1"]}
    ]
}
pipeline = Pipeline.load_from_config(pipeline_config)
```

### 2. 条件分支
```python
pipeline.add_conditional_node(
    component=decision_node,
    conditions={"condition_a": "path_a", "condition_b": "path_b"}
)
```

### 3. 并行处理
```python
pipeline.add_parallel_nodes([node1, node2, node3])
```

## 项目结构

```
pipeline-rag/
├── nodes/                # 节点实现
│   ├── base.py          # 基础组件接口
│   ├── document/        # 文档处理节点
│   ├── embedding/       # 向量化节点
│   ├── llm/            # LLM接口节点
│   ├── prompt/         # 提示词管理节点
│   └── vector/         # 向量存储节点
├── pipelines/           # Pipeline核心实现
│   ├── base.py         # Pipeline基类
│   └── config.py       # 配置管理
├── storage/            # 存储实现
│   ├── base.py        # 存储基类
│   └── excel_storage.py # Excel存储实现
└── examples/           # 示例应用
    ├── demo1_rag_main.py # 命令行示例
    └── ui.py           # Web界面示例
```

## 最佳实践

### 1. 节点设计原则
- 单一职责
- 明确的输入输出接口
- 完善的参数类型提示
- 适当的错误处理

### 2. 参数传递规范
- 显式声明所需参数
- 使用类型提示
- 参数验证
- 清晰的命名

### 3. Pipeline设计模式
- 合理的节点粒度
- 清晰的数据流向
- 有效的错误传播
- 适当的并行处理

### 4. 代码规范
```python
class WellDesignedNode(BaseComponent):
    """节点功能描述
    
    Args:
        param1 (str): 参数1的描述
        param2 (List[str]): 参数2的描述
    
    Returns:
        Tuple[Dict, Optional[str]]: 返回值描述
    """
    def run(self, param1: str, param2: List[str], **kwargs) -> Tuple[Dict, Optional[str]]:
        # 参数验证
        if not isinstance(param1, str):
            raise TypeError("param1 must be string")
        
        # 处理逻辑
        result = self._process_data(param1, param2)
        
        return {"output": result}, None
```

## 注意事项

1. 参数传递
   - 确保上下游节点参数名称匹配
   - 注意参数类型一致性
   - 处理可选参数的默认值

2. 错误处理
   - 适当的异常捕获和传播
   - 清晰的错误信息
   - 完整的日志记录

3. 性能优化
   - 合理使用并行处理
   - 优化数据传递
   - 注意内存管理

## 许可证

Apache 2.0 License

## 贡献指南

1. Fork该项目
2. 创建功能分支
3. 提交代码
4. 创建Pull Request

## 联系方式

- GitHub Issues
- Email: liuyuforwh@gmail