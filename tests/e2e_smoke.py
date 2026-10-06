"""端到端冒烟：真实调用 qwen-plus 跑两轮对话，验证完整链路。

和 tests/test_rag.py 的区别：
  test_rag.py  离线，只测解析 / 检索 / 工具输出，不花 token；
  本脚本       在线，会把 13 个语料 + 检索结果真的发给模型，会消耗 token。

运行（需要 DASHSCOPE_API_KEY）：
    python tests/e2e_smoke.py

它验证的是「多轮」这条最容易坏的路径：
  第 1 轮的 system prompt 注入 + 工具检索结果会留在历史里，
  第 2 轮再叠加一份注入资料和一份工具结果，容易顶到 max_input_tokens，
  所以这里连问两轮，确认第二轮既没报错、也还能正常给出带来源的答案。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 必须先 import config（设置 QWEN_AGENT_DEFAULT_WORKSPACE）
from config import LLM_CFG, list_doc_files  # noqa: F401

from cli import ask
from agent import build_agent

QUESTIONS = [
    '雇主责任险和团体意外险有什么区别？',   # 第一轮：冷启动，会触发工具检索
    '那这两种保险可以同时投保吗？',         # 第二轮：带历史追问，最容易被 token 上限卡住
]


def main():
    if not os.getenv('DASHSCOPE_API_KEY'):
        raise SystemExit('未配置 DASHSCOPE_API_KEY，无法做在线冒烟测试。')

    print(f'模型：{LLM_CFG["model"]}，语料 {len(list_doc_files())} 个文件')
    print('正在初始化 Agent ...')
    agent = build_agent()

    messages = []
    for i, question in enumerate(QUESTIONS, 1):
        print(f'\n=== 第 {i} 轮 ===')
        print(f'你 > {question}')
        response = ask(agent, messages, question)

        answer = ''.join(
            m.content for m in response
            if m.role == 'assistant' and isinstance(m.content, str)
        )
        assert answer.strip(), f'第 {i} 轮没有拿到任何回答'
        assert agent.last_sources, f'第 {i} 轮没有检索到任何文件'
        # 提示词要求标注来源，这里确认模型真的照做了
        assert '来源' in answer, f'第 {i} 轮的回答里没有标注来源：{answer[:200]}'

    print('\n多轮冒烟通过 ✅')


if __name__ == '__main__':
    main()
