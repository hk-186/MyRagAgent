"""全局配置：语料路径、LLM 参数、RAG 参数、系统提示词。

所有模块都从这里取配置，改模型 / 改检索策略只需要动这一个文件。
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DOC_DIR = BASE_DIR / 'docs'          # 语料目录（多文件问答的知识库）
WORKSPACE_DIR = BASE_DIR / 'workspace'  # qwen-agent 的解析缓存 / 中间产物

# 把 qwen-agent 的工作目录固定到项目下，
# 避免它使用相对路径 "workspace" 时跟着当前工作目录乱跑。
# 注意：必须在 import qwen_agent.settings 之前设置。
os.environ.setdefault('QWEN_AGENT_DEFAULT_WORKSPACE', str(WORKSPACE_DIR))

# 可解析的文件后缀（与 qwen_agent 的 PARSER_SUPPORTED_FILE_TYPES 对齐）
SUPPORTED_SUFFIXES = ('.txt', '.pdf', '.docx', '.pptx', '.html', '.csv', '.tsv', '.xlsx', '.xls')


def list_doc_files():
    """递归收集 docs/ 下所有可解析的语料文件，返回绝对路径列表。

    返回:
        List[str]: 按文件名排序的文件绝对路径。
    """
    if not DOC_DIR.exists():
        return []
    files = [
        p for p in DOC_DIR.rglob('*')
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return [str(p) for p in sorted(files)]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# 通过 dashscope 调用，API Key 自动读取环境变量 DASHSCOPE_API_KEY。
# 想换成更强的模型只需把 model 改成 qwen-max（更贵、引用更准）。
LLM_CFG = {
    'model': 'qwen-plus',
    'generate_cfg': {
        # 单次请求给模型的输入上限，超出会按「系统消息 + 最近一条用户消息」截断。
        # 这个值必须同时容下两份参考资料：
        #   1) Agent 每轮开头注入 system prompt 的检索片段（≤ max_ref_token）
        #   2) 模型自己调用 knowledge_retrieval 拿到的检索结果（≤ max_ref_token）
        # 两份加起来会到 40000 tokens 左右，只留 30000 会让请求直接报
        # 「input messages exceed the maximum context length」。
        # qwen-plus 的上下文窗口是 128K，这里给到 60K，留足多轮对话的余量。
        'max_input_tokens': 60000,
    },
}

# ---------------------------------------------------------------------------
# RAG（检索增强）
# ---------------------------------------------------------------------------
RAG_CFG = {
    # 检索回来后塞进 prompt 的参考资料 token 预算
    'max_ref_token': 20000,
    # 文档切块的块大小（token）
    'parser_page_size': 500,
    # 只用关键词检索：
    #   keyword_search     —— BM25 + jieba 中文分词
    #   front_page_search  —— 关键词全不命中时兜底返回文档开头
    # 刻意不使用 qwen-agent 内置的 vector_search，因为它依赖已被
    # langchain 1.x 移除的 langchain.schema 模块，在这个环境里会直接报错。
    'rag_searchers': ['keyword_search', 'front_page_search'],
    # 把用户问题改写成关键词再检索（中英文均可提升 BM25 命中率）
    'rag_keygen_strategy': 'GenKeyword',
}

# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一名专业的保险条款问答助手，服务对象是保险经纪人和投保企业客户。

你的知识库是一批保险产品资料（产品说明、条款、常见问答），涵盖雇主责任险、
团体意外险、商业综合责任险、财产一切险、施工保、装修保等险种。

回答要求：
1. 严格依据检索到的资料作答，不要凭常识补充资料里没有的内容。
2. 每条结论后面用【来源：文件名】标注出处，例如【来源：2-雇主责任险.txt】。
3. 如果检索到的资料不足以回答问题，直接说明"现有资料中未找到相关内容"，
   并告诉用户还可以换个什么角度提问，禁止编造条款、保额、费率。
4. 涉及多个险种对比时，用表格或分点罗列，便于阅读。
5. 用中文回答，语言简洁专业。"""


if __name__ == '__main__':
    # 直接运行本文件可以快速检查配置是否生效：
    #   python config.py
    docs = list_doc_files()
    print(f'语料目录：{DOC_DIR}')
    print(f'工作目录：{WORKSPACE_DIR}')
    print(f'发现 {len(docs)} 个语料文件：')
    for d in docs:
        print(f'  - {Path(d).name}')
    print(f'LLM：{LLM_CFG["model"]}')
    print(f'检索器：{RAG_CFG["rag_searchers"]}')
    print(f'DASHSCOPE_API_KEY 已配置：{bool(os.getenv("DASHSCOPE_API_KEY"))}')
