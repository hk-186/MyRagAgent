# MyRagAgent —— 保险条款问答 Agent

基于 [qwen-agent](https://github.com/QwenLM/Qwen-Agent) 的多文件检索增强问答 Agent：
把 `docs/` 目录下的一批保险产品资料（产品说明、条款、常见问答）当作知识库，
回答问题并标注依据的原文来源。

## 功能

- **多文件问答**：13 份语料（txt + pdf）统一解析、切块、建索引。
- **中文 BM25 检索**：`jieba` 分词 + BM25 排序，关键词全不命中时兜底返回文档开头。
- **模型自主检索**：资料不够时，模型会在多轮里自己换关键词再调一次 `knowledge_retrieval`。
- **来源可追溯**：答案用【来源：文件名】标注出处；命令行还会列出本轮命中的文件与跳过的文件。
- **解析结果落盘缓存**：第二次提问不再重复跑 pdfplumber（见下方「踩过的坑」）。
- **两种界面**：命令行 `cli.py` 与 Gradio 网页 `webui.py`。

## 目录结构

```
config.py                  全局配置：语料路径、LLM 参数、RAG 参数、系统提示词
agent.py                   InsuranceQAAgent：把语料 + 检索工具 + LLM + 提示词拼成助手
cli.py                     命令行入口（交互模式 / 单次提问）
webui.py                   Gradio WebUI 入口
rag/
  knowledge_retrieval.py   knowledge_retrieval 工具 + 修好缓存的 CachedDocParser
docs/                      知识库语料（txt / pdf）
tests/test_rag.py          离线冒烟测试（不消耗 token）
tests/e2e_smoke.py         在线端到端冒烟（真实调用 LLM 连问两轮）
workspace/                 qwen-agent 的解析缓存（自动生成，可删）
```

## 安装

```bash
# Python 3.11
pip install -r requirements.txt
```

需要配置 DashScope API Key（阿里云百炼）：

```bash
set DASHSCOPE_API_KEY=sk-xxxxxxxx     # Windows cmd
export DASHSCOPE_API_KEY=sk-xxxxxxxx  # Linux / macOS
```

## 使用

```bash
python config.py                      # 自检：语料数量、模型、Key 是否就位
python cli.py                         # 交互式多轮问答
python cli.py "施工保的保障范围是什么？"  # 单次提问后退出
python webui.py                       # 网页界面 http://127.0.0.1:7860
```

交互模式下的命令：`/sources` 看上一轮命中哪些文件、`/clear` 清空历史、`/exit` 退出。

## 测试

4 个用例全部离线运行，不调用 LLM、不消耗 token：

```bash
python -m pytest tests/test_rag.py -q
python tests/test_rag.py        # 也可以直接跑（带详细打印）
```

覆盖：

| 用例 | 验证内容 |
| --- | --- |
| `test_docs_are_parseable` | 每个语料都能解析出非空内容（扫描件会在这里暴露） |
| `test_parse_result_is_cached` | 切块结果落到 `workspace/`，二次解析不再走 pdfplumber |
| `test_bm25_ranking` | 典型问题能被 BM25 排到正确的文件上 |
| `test_tool_returns_json_with_sources` | 工具返回结构化 JSON，含 `source` / `content` / `note` |

另外有一个**在线**端到端冒烟脚本，会真实调用 qwen-plus 连问两轮（验证多轮历史不会顶到
`max_input_tokens`），会消耗 token，所以没放进 pytest：

```bash
python tests/e2e_smoke.py
```

## 一次问答的信息流

```
用户问题
  └─ InsuranceQAAgent._run
       ├─ 1) KnowledgeRetrieval 先做一次检索，结果拼进 system prompt（参考资料）
       └─ 2) FnCallAgent 的工具循环生成答案
              └─ 资料不足时，模型自己换关键词再调 knowledge_retrieval
```

注意两份参考资料（system prompt 注入的 + 工具返回的）各占最多 `max_ref_token`，
所以 `LLM_CFG['generate_cfg']['max_input_tokens']` 必须留够余量，
否则会报 `input messages exceed the maximum context length`。

## 关键配置（`config.py`）

| 配置 | 作用 |
| --- | --- |
| `LLM_CFG['model']` | 默认 `qwen-plus`；换成 `qwen-max` 引用更准但更贵 |
| `LLM_CFG['generate_cfg']['max_input_tokens']` | 单次请求输入上限，默认 60000 |
| `RAG_CFG['max_ref_token']` | 每次检索塞进 prompt 的 token 预算，默认 20000 |
| `RAG_CFG['parser_page_size']` | 文档切块大小（token），默认 500 |
| `RAG_CFG['rag_searchers']` | 检索器组合，默认 `keyword_search + front_page_search` |
| `RAG_CFG['rag_keygen_strategy']` | 检索前把问题改写成关键词，默认 `GenKeyword` |
| `SYSTEM_PROMPT` | 回答规范：必须标注来源、资料不足时明说、禁止编造条款 |

## 踩过的坑（都有注释写在代码里）

1. **解析缓存键不一致**（qwen-agent 0.0.19）：查缓存用 `{hash}_{page_size}`，
   但小文档写缓存时用 `{hash}_without_chunking`，导致缓存永远不命中、每次提问都重跑
   pdfplumber。`rag/knowledge_retrieval.py:CachedDocParser` 补存了一份。
2. **不用内置 `vector_search`**：它 `from langchain.schema import Document`，
   而该模块在 langchain 1.x 已被移除，本环境会直接 ImportError。因此只用关键词检索。
3. **不依赖 Assistant 默认的 Memory 检索**：上游用 `DocParser` 会踩坑 1，
   改为统一走自己修过缓存的检索器（覆写 `_prepend_knowledge_prompt`）。
4. **`QWEN_AGENT_DEFAULT_WORKSPACE`** 必须在 `import qwen_agent.settings` 之前设置，
   所以各入口文件都是先 `from config import ...`。

## 已知限制

- `docs/平安商业综合责任保险（亚马逊）.pdf` 是 4 页纯扫描件，没有文字层，
  提取不到任何字符，检索时会跳过并在 `note` / 命令行里点名；同内容的文字版由
  `docs/1-平安商业综合责任保险（亚马逊）.txt` 提供。要支持它需要接 OCR。
- 只做关键词检索，没有向量召回；同义改写的问法可能命中率偏低，
  可以把 `rag_searchers` 换成可用的向量检索实现来补强。
- 语料目录固定为 `docs/`；WebUI 里临时拖入的文件只对当轮问答生效。
