# nodes/llm/openai.py
import json
import http.client
from typing import Dict, Optional, Tuple
import time
import logging
from nodes.base import BaseComponent
from storage.base import BaseStorage

logger = logging.getLogger(__name__)


class GPT3Node(BaseComponent):
    """OpenAI GPT-3 节点"""

    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4o-2024-08-06",
                 temperature: float = 0.5,
                 top_p: float = 1.0,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 storage: Optional[BaseStorage] = None):
        """
        初始化GPT-3节点

        Args:
            api_key: API密钥
            model: 模型名称
            temperature: 温度参数(0-1)
            top_p: 核采样参数
            max_retries: 最大重试次数
            retry_delay: 重试延迟(秒)
            storage: 可选的存储实例
        """
        super().__init__(storage=storage)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 初始化HTTP连接
        self.conn = http.client.HTTPSConnection("oa.api2d.net")
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'{self.api_key}'
        }

    def run(self, prompt: str, **kwargs) -> Tuple[Dict, Optional[str]]:
        """
        运行GPT-3查询

        Args:
            prompt: 查询文本

        Returns:
            Tuple[Dict, Optional[str]]: (输出字典, None)
        """
        # 准备请求数据
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p
        }

        # 尝试发送请求
        for attempt in range(self.max_retries):
            try:
                response_data = self._make_request(payload)

                if "choices" in response_data and len(response_data["choices"]) > 0:
                    response_content = response_data["choices"][0]["message"]["content"]
                    output = {
                        "query": prompt,
                        "response": response_content,
                        "model": self.model,
                        "finish_reason": response_data["choices"][0].get("finish_reason")
                    }

                    # 保存输出
                    self.save_output(output)

                    return output, None
                else:
                    raise ValueError("API响应中没有有效的选择")

            except Exception as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # 指数退避
                else:
                    raise RuntimeError(f"达到最大重试次数，最后的错误: {str(e)}")

    def _make_request(self, payload: Dict) -> Dict:
        """
        发送API请求

        Args:
            payload: 请求数据

        Returns:
            Dict: API响应
        """
        try:
            # 发送请求
            self.conn.request("POST", "/v1/chat/completions",
                              json.dumps(payload), self.headers)

            # 获取响应
            response = self.conn.getresponse()
            data = response.read()

            # 检查响应状态
            if response.status != 200:
                raise ValueError(f"API请求失败: HTTP {response.status} - {data.decode('utf-8')}")

            # 解析响应
            return json.loads(data.decode("utf-8"))

        except Exception as e:
            raise RuntimeError(f"API请求异常: {str(e)}")

    def close(self):
        """关闭连接"""
        try:
            self.conn.close()
        except:
            pass
