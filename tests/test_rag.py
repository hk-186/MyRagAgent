"""离线冒烟测试：验证文档解析与 BM25 关键词检索，不消耗 token。

运行：
    python -m pytest tests/test_rag.py -q      # 用 pytest
    python tests/test_rag.py                   # 也可以直接跑

覆盖三件事：
1. docs/ 下每个语料文件都能解析出非空 chunk（扫描件 PDF 会在这里暴露）；
2. 分块结果会落盘缓存，二次解析不再走 pdfplumber；
3. BM25 检索能把典型问题排到对应的文件上。
"""

import json
import os
import sys
from types import SimpleNamespace as _SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 必须先 import config（设置 QWEN_AGENT_DEFAULT_WORKSPACE）
from config import RAG_CFG, WORKSPACE_DIR, list_doc_files

from qwen_agent.tools.doc_parser import DocParser, Record
from qwen_agent.tools.search_tools.keyword_search import KeywordSearch
from qwen_agent.tools.storage import KeyNotExistsError
from qwen_agent.utils.utils import hash_sha256

DOC_FILES = list_doc_files()

# 已知没有文字层、解析不出内容的文件。
# '平安商业综合责任保险（亚马逊）.pdf' 是 4 页纯扫描件（pdfplumber 提取到 0 个字符），
# 需要 OCR 才能用；同内容的整理版已由 '1-平安商业综合责任保险（亚马逊）.txt' 提供，
# 所以不影响问答效果。若以后补了带文字层的版本，请把这里删掉。
KNOWN_BLANK_FILES = {'平安商业综合责任保险（亚马逊）.pdf'}

# 问题 -> 期望排在前三位的文件名
RETRIEVAL_CASES = [
    ('雇主责任险和团体意外险有什么区别', '2-雇主责任险.txt'),
    ('雇主责任险A款B款94版有什么区别', '2-雇主责任险.txt'),
    ('财产一切险的保障范围是什么', '6-财产一切险.txt'),
    ('平安装修保是什么', '7-平安装修保.txt'),
    ('施工保的保险责任', '5-施工保.txt'),
]


def _parse_all(parser):
    """解析全部语料，返回 {文件名: (chunk 数, token 数)} 与 Record 列表。"""
    stats, records = {}, []
    for f in DOC_FILES:
        rec = Record(**parser.call({'url': f}))
        records.append(rec)
        stats[os.path.basename(f)] = (len(rec.raw), sum(c.token for c in rec.raw))
    return stats, records


def test_docs_are_parseable():
    """每个语料文件都应解析出非空内容，除非它本来就没有文字层。"""
    parser = DocParser({'path': str(WORKSPACE_DIR / 'tools' / 'doc_parser')})
    stats, _ = _parse_all(parser)

    print(f'\n共 {len(stats)} 个文件：')
    for name, (chunks, tokens) in stats.items():
        flag = '  <-- 无文字层' if tokens == 0 else ''
        print(f'  {name}: {chunks} 块 / {tokens} tokens{flag}')

    blank = {name for name, (_, tokens) in stats.items() if tokens == 0}
    unexpected = sorted(blank - KNOWN_BLANK_FILES)
    assert not unexpected, f'以下文件没解析出任何内容（疑似扫描件缺文字层）：{unexpected}'

    fixed = sorted(KNOWN_BLANK_FILES - blank)
    if fixed:
        print(f'提示：{fixed} 现在能解析出内容了，可以从 KNOWN_BLANK_FILES 里删掉')

    # 确认语料总量超过 max_ref_token —— 说明必须走检索，而不是整篇塞进上下文
    total = sum(t for _, t in stats.values())
    print(f'语料总 tokens：{total}')
    assert total > RAG_CFG['max_ref_token']


def test_parse_result_is_cached():
    """分块结果要落盘；缓存命中时不应再调用文档解析器。"""
    target = next(f for f in DOC_FILES if f.endswith('7-平安装修保.txt'))
    parser = DocParser({
        'path': str(WORKSPACE_DIR / 'tools' / 'doc_parser'),
        'max_ref_token': RAG_CFG['max_ref_token'],
        'parser_page_size': RAG_CFG['parser_page_size'],
    })
    first = parser.call({'url': target})

    # 1) 缓存文件确实写到了磁盘
    keys = [
        f'{hash_sha256(target)}_without_chunking',
        f'{hash_sha256(target)}_{RAG_CFG["parser_page_size"]}',
    ]
    cached = None
    for key in keys:
        try:
            cached = json.loads(parser.db.get(key))
            break
        except KeyNotExistsError:
            continue
    assert cached is not None, f'分块结果没有落盘，尝试过的 key：{keys}'

    # 2) 让底层解析器直接报错：能正常返回就说明这次是读的缓存
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError('缓存未命中，又重新解析了一次文档')

    parser.doc_extractor = _SimpleNamespace(call=_must_not_be_called)
    second = parser.call({'url': target})

    assert json.dumps(second, ensure_ascii=False) == json.dumps(first, ensure_ascii=False), \
        '两次解析结果不一致'


def test_bm25_ranking():
    """BM25 应把典型问题排到正确的文件上。"""
    parser = DocParser({'path': str(WORKSPACE_DIR / 'tools' / 'doc_parser')})
    _, records = _parse_all(parser)
    searcher = KeywordSearch()

    failures = []
    for query, expected in RETRIEVAL_CASES:
        ranked = searcher.sort_by_scores(query=query, docs=records)
        top = [os.path.basename(doc_id) for doc_id, _, _ in ranked[:3]]
        ok = expected in top
        print(f'  [{"OK " if ok else "NG "}] {query} -> {top}')
        if not ok:
            failures.append((query, expected, top))

    assert not failures, f'以下问题的检索结果不含期望文件：{failures}'


def test_tool_returns_json_with_sources():
    """knowledge_retrieval 工具的返回值应是可解析的 JSON，且带来源文件名。"""
    from rag import KnowledgeRetrieval

    tool = KnowledgeRetrieval()
    raw = tool.call({'query': '雇主责任险'}, files=DOC_FILES)
    data = json.loads(raw)
    assert 'results' in data
    assert data['results'], '检索结果不应为空'
    for item in data['results']:
        assert item['source'] and item['content']
    # 扫描件应在 note 里被点名，而不是静默消失
    assert '无文字层' in data.get('note', ''), data.get('note')


if __name__ == '__main__':
    for fn in [test_docs_are_parseable, test_parse_result_is_cached,
               test_bm25_ranking, test_tool_returns_json_with_sources]:
        print(f'\n=== {fn.__name__} ===')
        fn()
        print('PASS')
    print('\n全部通过 ✅')
