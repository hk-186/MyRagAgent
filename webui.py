"""Gradio WebUI 入口（基于 qwen-agent 自带的 WebUI）。

用法：
    python webui.py                 # 默认 http://127.0.0.1:7860
    python webui.py 7861            # 指定端口

界面上可以直接把新的 pdf/txt 拖进输入框，当轮问答会额外检索这个文件。
"""

import sys

# 必须先 import config（设置 QWEN_AGENT_DEFAULT_WORKSPACE）
from config import list_doc_files  # noqa: F401

from qwen_agent.gui import WebUI

from agent import AGENT_NAME, PROMPT_SUGGESTIONS, build_agent

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 7860


def main():
    agent = build_agent()
    chatbot_config = {
        'user.name': '我',
        'input.placeholder': '问点保险条款相关的问题，例如：雇主责任险和团体意外险有什么区别？',
        'prompt.suggestions': PROMPT_SUGGESTIONS,
        'verbose': False,
    }
    print(f'启动「{AGENT_NAME}」WebUI：http://127.0.0.1:{PORT}')
    WebUI(agent, chatbot_config=chatbot_config).run(
        server_name='127.0.0.1',
        server_port=PORT,
    )


if __name__ == '__main__':
    main()
