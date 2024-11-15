# 1. 新增 storage/base.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List


class BaseStorage(ABC):
    """存储基类"""

    @abstractmethod
    def save(self, data: Any):
        """保存数据"""
        pass

    @abstractmethod
    def load(self) -> Any:
        """加载数据"""
        pass

    @abstractmethod
    def close(self):
        """关闭存储"""
        pass