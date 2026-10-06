"""把 BM25 关键词检索包装成 qwen-agent 可以调用的工具。

为什么需要自己写一个工具，而不是直接用 qwen-agent 内置的 `retrieval`：
1. 内置 `retrieval` 把 `files`（文件路径列表）声明成必填参数，需要 LLM 自己
   填本地绝对路径，这在真实对话里是不可行的。本工具继承
   `BaseToolWithFileAccess`（`file_access = True`），qwen-agent 在执行工具前
   会自动把当前会话涉及的文件列表注入进来，LLM 只需要给检索关键词。
2. `Assistant` 每轮开头只会检索一次；有了这个工具，模型在多轮追问时可以
   自己换关键词再检索一次，这是"多文件问答"的关键体验。

检索算法：qwen-agent 的 `HybridSearch`，内部对 `keyword_search`（BM25 + jieba
中文分词）和 `front_page_search`（关键词全不命中时兜底返回文档开头）做
RRF 式融合排序。

刻意不使用内置 `vector_search`：它依赖 `from langchain.schema import Document`，
而该模块在 langchain 1.x 中已被移除，在本环境里会直接 ImportError。
"""

import json
import os
from pathlib import Path
from typing import List

# 必须放在 qwen_agent 之前 import：
# config.py 会设置环境变量 QWEN_AGENT_DEFAULT_WORKSPACE，
# 而 qwen_agent.settings 是在 import 时读取该变量的，晚设置就无效了。
from config import RAG_CFG, WORKSPACE_DIR, list_doc_files

from qwen_agent.log import logger
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.tools.doc_parser import DocParser, Record
from qwen_agent.tools.search_tools.hybrid_search import HybridSearch
from qwen_agent.tools.simple_doc_parser import PARSER_SUPPORTED_FILE_TYPES
from qwen_agent.tools.storage import KeyNotExistsError
from qwen_agent.utils.utils import get_file_type, hash_sha256


class CachedDocParser(DocParser):
    """修掉 qwen-agent 0.0.19 的解析缓存键不一致问题。

    上游 `DocParser.call` 的行为是：
      · 查找缓存时用 key = f'{hash}_{parser_page_size}'
      · 但如果文档总 token <= max_ref_token（整篇当一个 chunk），
        写缓存时 key 会被改成 f'{hash}_without_chunking'
    于是这类"小文档"第二次进来还是查不到缓存，每次都要重新解析。
    本项目 13 个语料全部小于 max_ref_token，等于缓存完全失效——PDF 每次
    提问都要重新跑 pdfplumber。

    这里在解析完成后，把结果按"查找时使用的 key"补存一份，让缓存真正生效。
    升级 qwen-agent 后如果上游已修复，可以直接删掉这个类。
    """

    def call(self, params, **kwargs):
        record = super().call(params, **kwargs)
        url = self._verify_json_format_args(params)['url']
        lookup_key = f'{hash_sha256(url)}_{self.parser_page_size}'
        try:
            self.db.get(lookup_key)
        except KeyNotExistsError:
            self.db.put(lookup_key, json.dumps(record, ensure_ascii=False))
        return record


@register_tool('knowledge_retrieval')
class KnowledgeRetrieval(BaseToolWithFileAccess):
    """从会话关联的多个文档中检索与问题最相关的原文片段。"""

    description = (
        '从知识库文档中检索与问题最相关的原文片段，供你组织答案。'
        '当需要查阅保险条款、产品说明、保障范围、免赔约定等资料时调用它。'
        'query 填写问题的核心关键词（逗号分隔，中英文混排效果更好）；'
        '如果一次检索的内容不足以回答问题，可以换一组关键词再次调用。'
    )

    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': '检索关键词，用逗号分隔。例如：雇主责任险,团体意外险,区别',
        'required': True,
    }]

    def __init__(self, cfg=None):
        super().__init__(cfg)
        # 允许通过 cfg 覆盖检索参数，默认取 config.py 里的全局配置
        self.max_ref_token = self.cfg.get('max_ref_token', RAG_CFG['max_ref_token'])
        self.parser_page_size = self.cfg.get('parser_page_size', RAG_CFG['parser_page_size'])
        self.rag_searchers = self.cfg.get('rag_searchers', RAG_CFG['rag_searchers'])

        # 文档解析 + 切块：结果落盘缓存（CachedDocParser 修了缓存键问题）
        self.parser = CachedDocParser({
            'path': str(WORKSPACE_DIR / 'tools' / 'doc_parser'),
            'max_ref_token': self.max_ref_token,
            'parser_page_size': self.parser_page_size,
        })
        # 关键词检索：BM25 + front_page 兜底
        self.searcher = HybridSearch({
            'max_ref_token': self.max_ref_token,
            'rag_searchers': self.rag_searchers,
        })

        # 本次会话的检索痕迹，供 CLI / WebUI 展示
        self.sources: List[str] = []   # 命中的文件名
        self.warnings: List[str] = []  # 解析失败或没有文字层的文件

    def begin_turn(self):
        """开始新一轮对话前清空检索痕迹。"""
        self.sources = []
        self.warnings = []

    # ------------------------------------------------------------------
    # 工具入口
    # ------------------------------------------------------------------
    def call(self, params, files: List[str] = None, **kwargs) -> str:
        """qwen-agent 调用入口，返回 JSON 字符串。

        返回结构（供 LLM 阅读，也方便命令行/测试断言）：
            {
              "results": [{"source": "文件名", "path": "绝对路径", "content": "片段"}],
              "note": "没有文字层 / 解析失败的文件说明（可能不存在）"
            }
        这里刻意把同一文件的多个片段合并成一条，方便模型直接用文件名标注来源。
        """
        params = self._verify_json_format_args(params)
        results = self.retrieve(params.get('query', ''), files)
        return json.dumps(self._format_output(results), ensure_ascii=False, indent=2)

    def _format_output(self, results: List[dict]) -> dict:
        """把 searcher 的原生结果整理成带来源文件名的结构化输出。"""
        hits = []
        for item in results:
            chunks = [t for t in item.get('text', []) if t]
            if not chunks:
                continue
            hits.append({
                'source': Path(item['url']).name,
                'path': item['url'],
                'content': '\n\n'.join(chunks),
            })

        payload = {'results': hits}
        if hits:
            note = '以上片段已按相关度排序。'
        else:
            note = '未检索到与关键词相关的原文，请换一组关键词重试。'
        if self.warnings:
            # 解析失败 / 扫描件等需要让模型知道"有些资料没查到"
            note += f' 未参与检索的文件：{"；".join(self.warnings)}'
        payload['note'] = note
        return payload

    # ------------------------------------------------------------------
    # 供 agent / CLI / 测试调用的主逻辑
    # ------------------------------------------------------------------
    def retrieve(self, query: str, files: List[str] = None) -> List[dict]:
        """检索与 query 相关的原文片段。

        参数:
            query: 检索关键词。
            files: 待检索文件列表；None 表示使用当前工作目录 docs/ 下的全部语料。
        返回:
            qwen-agent 原生格式 [{'url': '文件路径', 'text': ['片段', ...]}, ...]
        """
        target_files = self._filter_files(files)
        if not target_files:
            target_files = self._filter_files(list_doc_files())
        if not target_files:
            self.warnings.append('没有可检索的文档')
            return []

        query = (query or '').strip()
        logger.info(f'[knowledge_retrieval] query={query!r} files={len(target_files)}')

        records, failed, blanks = [], [], []
        for f in target_files:
            try:
                record = Record(**self.parser.call({'url': f}))
            except Exception as ex:  # 单个文件解析失败不应中断整次检索
                logger.warning(f'解析文件失败 {f}: {ex}')
                failed.append(f'{Path(f).name}（解析失败：{type(ex).__name__}）')
                continue
            if sum(c.token for c in record.raw) == 0:
                # 扫描件 PDF 没有文字层，pdfplumber 提取不到任何内容
                logger.warning(f'文件没有可提取的文本（疑似扫描件），已跳过：{f}')
                blanks.append(f'{Path(f).name}（无文字层，疑似扫描件）')
                continue
            records.append(record)

        for msg in failed + blanks:
            if msg not in self.warnings:
                self.warnings.append(msg)

        if not records:
            return []

        results = self.searcher.call(
            {'query': query},
            docs=records,
            max_ref_token=self.max_ref_token,
        )

        # 记录命中的文件名，供 CLI / WebUI 展示"答案依据了哪些文件"
        for item in results:
            name = Path(item['url']).name
            if name not in self.sources:
                self.sources.append(name)

        return results

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _filter_files(files: List[str]) -> List[str]:
        """去重 + 剔除不支持的文件类型。

        `file_access=True` 的工具会拿到消息里所有附件（可能包含图片），
        以及 Memory 记录的静态知识库文件，这里统一过滤一遍。
        """
        target_files = []
        for f in (files or []):
            if f in target_files:
                continue
            if not os.path.exists(f):
                continue
            if get_file_type(f) in PARSER_SUPPORTED_FILE_TYPES:
                target_files.append(f)
        return target_files
