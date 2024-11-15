from abc import ABC
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from networkx import DiGraph
import networkx as nx
import logging
from nodes.base import BaseComponent
from pipelines.config import (
    get_component_definitions,
    get_pipeline_definition,
    read_pipeline_config_from_yaml,
)
logger = logging.getLogger(__name__)

ROOT_NODE_TO_PIPELINE_NAME = {"query": "query", "file": "indexing"}
CODE_GEN_DEFAULT_COMMENT = "This code has been generated."


class RootNode(BaseComponent):
    """根节点类,用于初始化pipeline的输入"""
    outgoing_edges = 1

    def run(self, query: Optional[str] = None, **kwargs):
        """运行根节点"""
        return {"query": query}, "output_1"

    def run_batch(self):
        raise NotImplementedError


class BasePipeline:
    """Pipeline基类"""

    def run(self, **kwargs):
        raise NotImplementedError

    def get_config(self, return_defaults: bool = False) -> dict:
        raise NotImplementedError

    @classmethod
    def load_from_config(cls, pipeline_config: Dict, pipeline_name: Optional[str] = None,
                         overwrite_with_env_variables: bool = True):
        """从配置加载pipeline"""
        pipeline_definition = get_pipeline_definition(
            pipeline_config=pipeline_config,
            pipeline_name=pipeline_name
        )
        component_definitions = get_component_definitions(
            pipeline_config=pipeline_config,
            overwrite_with_env_variables=overwrite_with_env_variables
        )

        pipeline = cls()
        components: dict = {}

        for node in pipeline_definition["nodes"]:
            name = node["name"]
            component = cls._load_or_get_component(
                name=name,
                definitions=component_definitions,
                components=components
            )
            pipeline.add_node(
                component=component,
                name=name,
                inputs=node.get("inputs", [])
            )

        return pipeline

    @classmethod
    def _load_or_get_component(cls, name: str, definitions: dict, components: dict):
        """加载或获取组件"""
        if name in components:
            return components[name]

        component_def = definitions[name]
        component_type = component_def["type"]
        component_params = component_def.get("params", {})

        # 处理组件参数中的组件引用
        for key, value in component_params.items():
            if isinstance(value, str) and value in definitions:
                if value not in components:
                    components[value] = cls._load_or_get_component(
                        value, definitions, components
                    )
                component_params[key] = components[value]

        instance = BaseComponent.load_from_args(
            component_type=component_type,
            **component_params
        )
        components[name] = instance
        return instance


class Pipeline(BasePipeline):
    """Pipeline实现类"""

    def __init__(self):
        self.graph = DiGraph()
        self.root_node = None

    def add_node(self, component: BaseComponent, name: str, inputs: List[str]):
        """添加节点到pipeline

        Args:
            component: 组件实例
            name: 节点名称
            inputs: 输入节点列表
        """
        # 处理根节点
        if self.root_node is None:
            root_node = inputs[0]
            if root_node in ["Query", "File"]:
                self.root_node = root_node
                self.graph.add_node(root_node, component=RootNode())
            else:
                raise KeyError(f"无效的根节点类型: {root_node}")

        # 添加组件节点
        component.name = name
        self.graph.add_node(name, component=component)

        # 处理节点连接
        if len(self.graph.nodes) == 2:
            self.graph.add_edge(self.root_node, name, label="output_1")
            return

        for input_node in inputs:
            if "." in input_node:
                input_node_name, output_label = input_node.split(".")
                self.graph.add_edge(input_node_name, name, label=output_label)
            else:
                self.graph.add_edge(input_node, name)

    @property
    def components(self):
        """获取所有非根节点组件"""
        return {
            name: attrs["component"]
            for name, attrs in self.graph.nodes.items()
            if not isinstance(attrs["component"], RootNode)
        }

    def get_node(self, name: str) -> Optional[BaseComponent]:
        """获取指定名称的节点"""
        if name in self.graph.nodes:
            return self.graph.nodes[name]["component"]
        return None

    def get_nodes_by_class(self, class_type) -> List[BaseComponent]:
        """获取指定类型的所有节点"""
        return [
            node["component"]
            for node in self.graph.nodes.values()
            if isinstance(node["component"], class_type)
        ]

    def run(self, **kwargs) -> Dict[str, Any]:
        """运行pipeline

        Args:
            **kwargs: 运行参数

        Returns:
            Dict[str, Any]: 运行结果
        """
        node_output = None
        queue = {
            self.root_node: {
                "root_node": self.root_node,
                "query": kwargs.get("query"),
                "file_path": kwargs.get("file_path"),
                "params": kwargs
            }
        }
        processed_nodes = set()

        while queue:
            # 找到可以执行的节点
            node_id = None
            for potential_node in queue:
                predecessors = set(self.graph.predecessors(potential_node))
                if predecessors.issubset(processed_nodes):
                    node_id = potential_node
                    break

            if node_id is None:
                raise RuntimeError("Pipeline执行顺序错误")

            try:
                # 获取节点输入
                node_input = queue.pop(node_id)
                node_input["node_id"] = node_id

                # 执行节点
                component = self.graph.nodes[node_id]["component"]
                node_output, stream_id = component._dispatch_run(**node_input)

                # 标记节点为已处理
                processed_nodes.add(node_id)

                # 处理后续节点
                next_nodes = self._get_next_nodes(node_id, stream_id)
                for successor in next_nodes:
                    if successor in queue:
                        existing_input = queue[successor]
                        if "inputs" not in existing_input:
                            queue[successor] = {
                                "inputs": [existing_input, node_output],
                                "params": kwargs
                            }
                        else:
                            existing_input["inputs"].append(node_output)
                    else:
                        queue[successor] = node_output

            except Exception as e:
                logger.error(f"节点 {node_id} 执行失败: {str(e)}")
                raise RuntimeError(f"节点 {node_id} 执行失败: {str(e)}")

        return node_output

    def _get_next_nodes(self, node_id: str, stream_id: Optional[str]) -> List[str]:
        """获取下一个要执行的节点列表"""
        edges = self.graph.edges(node_id, data=True)
        return [
            next_node for _, next_node, data in edges
            if not stream_id or
               "label" not in data or
               data["label"] == stream_id or
               stream_id == "output_all"
        ]

    def get_node_names_in_order(self) -> List[str]:
        """获取按拓扑顺序排序的节点名称"""
        return list(nx.topological_sort(self.graph))

    def get_config(self, return_defaults: bool = False) -> dict:
        """获取pipeline配置"""
        pipeline_name = ROOT_NODE_TO_PIPELINE_NAME[self.root_node.lower()]

        pipeline_def = {
            "name": pipeline_name,
            "type": self.__class__.__name__,
            "nodes": []
        }

        components = {}
        for node in self.graph.nodes:
            if node == self.root_node:
                continue

            component = self.graph.nodes[node]["component"]
            component_config = component.get_config()

            components[node] = {
                "name": node,
                "type": component_config["type"],
                "params": component_config.get("params", {})
            }

            pipeline_def["nodes"].append({
                "name": node,
                "inputs": list(self.graph.predecessors(node))
            })

        return {
            "components": list(components.values()),
            "pipelines": [pipeline_def],
            "version": "ignore"
        }
