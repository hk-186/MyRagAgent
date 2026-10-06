"""组装问答 Agent：把语料、检索工具、LLM、提示词拼成一个可对话的助手。

整体信息流（一次提问）：
    用户问题
      └─ InsuranceQAAgent._run
           ├─ 1) 用 KnowledgeRetrieval 做一次 BM25 关键词检索
           │     结果按 qwen-agent 原生格式拼进 system prompt（参考资料）
           └─ 2) 交给 FnCallAgent 的工具循环生成答案
                 如果资料不够，模型会自己换关键词再调 knowledge_retrieval

为什么不用 Assistant 默认的 Memory 自动检索：
  Memory 内部用的是 qwen-agent 原生的 DocParser，它的磁盘缓存键有 bug
  （见 rag/knowledge_retrieval.CachedDocParser），会导致每轮提问都把 13 个
  文档、包括 6 个 PDF 重新解析一遍。这里改为统一走我们修好缓存的检索器。
"""

# 必须先 import config：它会设置 QWEN_AGENT_DEFAULT_WORKSPACE 环境变量
from config import LLM_CFG, RAG_CFG, SYSTEM_PROMPT, list_doc_files

from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import USER
from qwen_agent.utils.utils import extract_files_from_messages, extract_text_from_message

from rag import KnowledgeRetrieval

AGENT_NAME = '保险条款问答助手'

AGENT_DESC = (
    '基于 docs/ 目录下的保险产品资料进行多文件问答，'
    '答案会标注依据的文件来源。'
)

# WebUI 首页的推荐问题
PROMPT_SUGGESTIONS = [
    '雇主责任险和团体意外险有什么区别？',
    '施工保的保障范围包括哪些？',
    '平安装修保适合什么场景投保？',
    '财产一切险的除外责任有哪些？',
]


class InsuranceQAAgent(Assistant):
    """检索增强的保险条款问答 Agent。"""

    def __init__(self, files=None, **kwargs):
        # 检索器要在 super().__init__ 之前建好，随后通过 function_list 交给 LLM
        self.retriever = KnowledgeRetrieval()
        super().__init__(
            llm=kwargs.pop('llm', LLM_CFG),
            system_message=kwargs.pop('system_message', SYSTEM_PROMPT),
            name=kwargs.pop('name', AGENT_NAME),
            description=kwargs.pop('description', AGENT_DESC),
            # 静态知识库：每次提问都会在这些文件里做检索
            files=files if files is not None else list_doc_files(),
            # BM25 关键词检索配置（见 config.RAG_CFG）
            rag_cfg=kwargs.pop('rag_cfg', RAG_CFG),
            # 允许模型在多轮追问时自己换关键词再检索
            function_list=[self.retriever],
            **kwargs,
        )

    def _run(self, messages, lang='en', **kwargs):
        # 新一轮对话开始，清空上一轮的检索痕迹
        self.retriever.begin_turn()
        return super()._run(messages=messages, lang=lang, **kwargs)

    def _prepend_knowledge_prompt(self, messages, lang='en', knowledge='', **kwargs):
        """把检索到的原文片段拼进 system prompt。

        覆写 Assistant 的同名方法：Assistant 默认在 knowledge 为空时会走
        `self.mem.run(...)`（即内置 Memory 检索），这里换成我们自己的检索器，
        这样既统一了文档解析缓存，也能把命中来源记录下来给命令行展示。
        knowledge 非空时（外部直接注入）沿用父类逻辑。
        """
        if not knowledge:
            knowledge = self._retrieve(messages)
        return super()._prepend_knowledge_prompt(
            messages=messages, lang=lang, knowledge=knowledge, **kwargs)

    def _retrieve(self, messages) -> list:
        """按最后一个用户问题检索，返回 qwen-agent 原生格式的检索结果。"""
        query = ''
        if messages and messages[-1].role == USER:
            query = extract_text_from_message(messages[-1], add_upload_info=False)
        if not query:
            return []

        # 静态知识库 + 用户在界面上临时上传的文件
        files = extract_files_from_messages(messages, include_images=False) + self.mem.system_files
        return self.retriever.retrieve(query, files)

    @property
    def last_sources(self):
        """最近一轮检索命中的文件名（供命令行展示）。"""
        return list(self.retriever.sources)

    @property
    def last_warnings(self):
        """最近一轮检索中遇到的问题（解析失败 / 无文字层的文件）。"""
        return list(self.retriever.warnings)


def build_agent(files=None):
    """创建问答 Agent。

    参数:
        files: 参与问答的文件列表；默认取 docs/ 下的全部语料。
    返回:
        InsuranceQAAgent 实例。
    """
    doc_files = files if files is not None else list_doc_files()
    if not doc_files:
        raise RuntimeError('没有找到任何语料文件，请把资料放进 docs/ 目录后重试。')
    return InsuranceQAAgent(files=doc_files)


if __name__ == '__main__':
    # 快速自检：python agent.py
    agent = build_agent()
    print(f'Agent 名称：{agent.name}')
    print(f'知识库文件数：{len(agent.mem.system_files)}')
    print(f'可用工具：{list(agent.function_map.keys())}')
    print(f'检索器：{RAG_CFG["rag_searchers"]}')
