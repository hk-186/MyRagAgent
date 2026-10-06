"""命令行入口：多文件保险条款问答。

用法：
    python cli.py                       # 进入交互式多轮问答
    python cli.py "施工保保额是多少？"   # 单次提问后退出（方便脚本化调用）

交互模式下支持的命令：
    /sources  查看上一轮检索命中的文件
    /clear    清空对话历史
    /exit     退出
"""

import json
import sys

# 必须先 import config（设置 QWEN_AGENT_DEFAULT_WORKSPACE）
from config import RAG_CFG, list_doc_files

from qwen_agent.llm.schema import ASSISTANT, Message

from agent import AGENT_NAME, build_agent

HELP_TEXT = """可用命令：
  /sources  查看上一轮检索命中的文件
  /clear    清空对话历史
  /exit     退出
"""


def print_sources(agent):
    """打印本轮答案依据了哪些文件。"""
    sources = agent.last_sources
    if sources:
        print(f'[检索命中] {"、".join(sources)}')
    else:
        print('[检索命中] 本轮没有检索到文档片段')
    for warning in agent.last_warnings:
        print(f'[跳过] {warning}')
    print()


def _parse_args(arguments):
    """把工具调用的参数解析成 dict。

    流式输出时参数是一段段拼出来的，中途不是合法 JSON，这里返回 None 表示
    "还没拼完"，调用方先不打印提示。
    """
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _format_args(args):
    """把工具参数显示成人话：检索关键词直接显示，其它参数退回 JSON。"""
    if 'query' in args:
        return str(args['query'])
    return json.dumps(args, ensure_ascii=False)


def ask(agent, messages, question, show_sources=True):
    """提问并流式打印答案，返回本轮完整响应。"""
    messages.append(Message(role='user', content=question))

    printed = 0        # 已打印字符数，用于把累积式流输出转成增量打印
    printed_call = None  # 已提示过的工具调用参数，避免流式分片重复打印
    response = []
    for response in agent.run(messages):
        if not response:
            continue
        last = response[-1]

        # 模型决定调用工具：等参数拼成合法 JSON 后再提示一次，并把打印位置归零
        if last.function_call:
            printed = 0
            args = _parse_args(last.function_call.arguments)
            if args is not None and args != printed_call:
                printed_call = args
                print(f'（检索中：{_format_args(args)}）')
            continue

        if last.role == ASSISTANT and isinstance(last.content, str):
            if len(last.content) > printed:
                sys.stdout.write(last.content[printed:])
                sys.stdout.flush()
                printed = len(last.content)

    if response:
        messages.extend(response)  # 保留工具调用消息，维持多轮上下文
    print()
    if show_sources:
        print_sources(agent)
    return response


def main():
    doc_files = list_doc_files()
    print(f'加载 {len(doc_files)} 个语料文件，检索方式：{"+".join(RAG_CFG["rag_searchers"])}')
    print('正在初始化 Agent（首次运行需要解析 PDF，请稍候）...')
    agent = build_agent()
    print(f'就绪！我是「{AGENT_NAME}」，直接输入问题即可。输入 /exit 退出。\n')

    # -------- 单次提问模式：python cli.py "问题" --------
    if len(sys.argv) > 1:
        ask(agent, [], ' '.join(sys.argv[1:]).strip())
        return

    # -------- 交互模式 --------
    messages = []
    print(HELP_TEXT)
    while True:
        try:
            question = input('你 > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n再见！')
            break

        if not question:
            continue
        if question in ('/exit', '/quit'):
            print('再见！')
            break
        if question == '/clear':
            messages.clear()
            print('对话历史已清空。\n')
            continue
        if question == '/sources':
            print_sources(agent)
            continue

        ask(agent, messages, question)


if __name__ == '__main__':
    main()
