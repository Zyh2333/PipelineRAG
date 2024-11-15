# nodes/base.py
from __future__ import annotations
import inspect
import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple
from storage.base import BaseStorage

logger = logging.getLogger(__name__)


class BaseComponent:
    outgoing_edges: int = 1
    subclasses: dict = {}
    pipeline_config: dict = {}
    name: Optional[str] = None

    def __init__(self, storage: Optional[BaseStorage] = None, **kwargs):
        """
        初始化基础组件

        Args:
            storage: 可选的存储实例
            **kwargs: 其他参数
        """
        self.storage = storage
        self.kwargs = kwargs

    def save_output(self, output: Any):
        """保存节点输出"""
        if self.storage:
            try:
                self.storage.save(output)
            except Exception as e:
                logger.error(f"存储输出失败: {str(e)}")

    def __init_subclass__(cls, **kwargs):
        """注册子类"""
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def get_subclass(cls, component_type: str):
        """获取对应的子类

        Args:
            component_type: 组件类型名称
        """
        if component_type not in cls.subclasses:
            raise Exception(f"找不到名为 '{component_type}' 的组件。")
        return cls.subclasses[component_type]

    @classmethod
    def load_from_args(cls, component_type: str, **kwargs):
        """从参数创建组件实例

        Args:
            component_type: 组件类型名称
            **kwargs: 组件参数
        """
        storage = kwargs.pop('storage', None)  # 提取存储参数
        subclass = cls.get_subclass(component_type)
        return subclass(storage=storage, **kwargs)

    @classmethod
    def load_from_pipeline_config(cls, pipeline_config: dict, component_name: str):
        """从pipeline配置加载组件

        Args:
            pipeline_config: pipeline配置字典
            component_name: 组件名称
        """
        if pipeline_config:
            all_component_configs = pipeline_config["components"]
            all_component_names = [comp["name"] for comp in all_component_configs]
            component_config = next(comp for comp in all_component_configs if comp["name"] == component_name)
            component_params = component_config["params"]

            # 处理存储配置
            storage_config = component_params.pop("storage", None)
            if storage_config and isinstance(storage_config, dict):
                storage_type = storage_config.pop("type", None)
                if storage_type:
                    from storage.base import BaseStorage
                    storage = BaseStorage.get_subclass(storage_type)(**storage_config)
                else:
                    storage = None
            else:
                storage = None

            # 处理其他参数
            component_params = {
                key: cls.load_from_pipeline_config(pipeline_config, value) if value in all_component_names else value
                for key, value in component_params.items()
            }

            # 创建组件实例
            component_instance = cls.load_from_args(
                component_config["type"],
                storage=storage,
                **component_params
            )
        else:
            component_instance = cls.load_from_args(component_name)
        return component_instance

    @abstractmethod
    def run(self, **kwargs) -> Tuple[Dict, Optional[str]]:
        """运行组件的核心逻辑

        Returns:
            Tuple[Dict, Optional[str]]: (输出字典, 可选的流ID)
        """
        pass

    def _dispatch_run(self, **kwargs) -> Tuple[Dict, Optional[str]]:
        """分发运行请求"""
        return self._dispatch_run_general(self.run, **kwargs)

    def _dispatch_run_batch(self, **kwargs):
        """分发批处理运行请求"""
        return self._dispatch_run_general(self.run_batch, **kwargs)

    def _dispatch_run_general(self, run_method: Callable, **kwargs) -> Tuple[Dict, Optional[str]]:
        """通用运行分发逻辑

        Args:
            run_method: 要运行的方法
            **kwargs: 运行参数
        """
        params = deepcopy(kwargs.get("params") or {})

        # 获取方法参数
        run_signature_args = inspect.signature(run_method).parameters.keys()

        # 构建运行参数
        run_params = {
            key: value for key, value in params.items()
            if key == self.name and isinstance(value, dict)
        }
        run_params.update({
            key: value for key, value in params.items()
            if key in run_signature_args and key not in kwargs
        })

        # 构建输入参数
        run_inputs = {
            key: value for key, value in kwargs.items()
            if key in run_signature_args
        }

        # 运行方法
        output, stream = run_method(**run_inputs, **run_params)

        # 处理输出
        output["params"] = params
        for k, v in kwargs.items():
            if k not in output and k != "inputs":
                output[k] = v

        # 保存输出
        self.save_output(output)

        return output, stream

    def set_config(self, **kwargs):
        """设置组件配置"""
        if not self.pipeline_config:
            self.pipeline_config = {
                "params": {},
                "type": type(self).__name__
            }

            # 处理存储配置
            if self.storage:
                self.pipeline_config["params"]["storage"] = {
                    "type": type(self.storage).__name__,
                    **self.storage.__dict__
                }

            # 处理其他参数
            for k, v in kwargs.items():
                if isinstance(v, BaseComponent):
                    self.pipeline_config["params"][k] = v.pipeline_config
                elif v is not None:
                    self.pipeline_config["params"][k] = v