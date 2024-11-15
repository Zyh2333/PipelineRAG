from typing import Dict, Optional, Tuple
from nodes.base import BaseComponent
from storage.base import BaseStorage


class PromptNode(BaseComponent):
    def __init__(self,
                 template: Optional[str] = None,
                 storage: Optional[BaseStorage] = None):
        super().__init__(storage=storage)
        self.template = template or self._get_default_template()

    def _get_default_template(self) -> str:
        return """请仔细阅读以下参考信息,并基于这些信息回答问题。
如果参考信息中没有相关内容,请直接说明无法从给定信息中找到答案。
请保持回答的客观性,不要添加参考信息以外的内容。
在回答时，请使用格式"《文件名》第X页"来引用信息来源。

参考信息:
{context}

问题: {query}

请给出准确、完整且客观的回答，并在相关内容处标注信息来源。例如：
{{回答结果}}

参考来源：
[1] {{标注信息来源}}
[2] ...
"""

    def _format_contexts(self, contexts: list) -> str:
        """格式化上下文信息，按文件和页码组织"""
        # 按来源和页码分组
        source_groups = {}
        for ctx in contexts:
            source = ctx.get("source", "未知来源")
            metadata = ctx.get("metadata", {})
            page = metadata.get("page_count", "未知页码")
            text = ctx.get("text", "").strip()
            score = ctx.get("score", 0.0)

            key = (source, page)
            if key not in source_groups:
                source_groups[key] = []

            if text:
                source_groups[key].append((text, score))

        # 格式化输出
        formatted_sections = []
        for i, ((source, page), contents) in enumerate(source_groups.items(), 1):
            source_header = f"[{i}] 来源：《{source}》第{page}页 "

            # 将同一页的内容合并，用相关度最高的分数
            combined_text = "\n    ".join(text for text, _ in contents)
            max_score = max(score for _, score in contents)

            section = f"{source_header}(相关度: {max_score:.3f})\n    {combined_text}"
            formatted_sections.append(section)

        return "\n\n".join(formatted_sections)

    def run(self, query: str, results: Dict, **kwargs) -> Tuple[Dict, Optional[str]]:
        """构建提示"""
        contexts = results

        if not contexts:
            prompt = f"""无法找到与问题相关的参考信息。

问题: {query}

请告知用户无法从现有信息中找到答案。"""
        else:
            formatted_context = self._format_contexts(contexts)
            prompt = self.template.format(
                context=formatted_context,
                query=query
            )

        output = {
            "original_query": query,
            "prompt": prompt,
            "context_used": bool(contexts),
            "num_contexts": len(contexts)
        }

        self.save_output(output)
        return output, None

    def set_template(self, template: str):
        """更新提示模板"""
        self.template = template