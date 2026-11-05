"""web_search_pdf -- serper 搜索 filetype:pdf。"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Any, Dict
import aiohttp
import requests
from settings import load_settings
from tools.file_utils import download_from_url
from rle_open_sdk.sdk import Environment

_SERPER_API_KEY = load_settings()["serper_api_key"]
_SERPER_URL = "https://google.serper.dev/search"
_PROXY = load_settings().get("proxy") or None
# 默认最多成功下载的文件数
DEFAULT_MAX_DOWNLOAD_COUNT = 1

WEB_SEARCH_PDF_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search_pdf",
        "description": "Search the web for PDF files. Automatically adds filetype:pdf to query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "sandbox_dir": {"type": "string", "description": "Absolute path in sandbox to upload downloaded PDFs."}
            },
            "required": ["query", "sandbox_dir"],
        },
    },
}


def _search_pdf(query_with_type: str) -> list:
    """搜索PDF文件（使用requests库，3次指数退避重试）"""
    import time
    api_key = _SERPER_API_KEY
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = json.dumps({"q": query_with_type, "autocorrect": False})

    for attempt in range(3):
        try:
            response = requests.post(
                _SERPER_URL,
                headers=headers,
                data=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            results = []
            for r in data.get('organic', [])[:5]:
                results.append({
                    "query": query_with_type,
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "snippet": r.get("snippet", "")
                })
            return results
        except Exception as e:
            print(f"[搜索] 失败 (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return []


async def _search_pdf_async(query_with_type: str) -> list:
    """搜索PDF文件，异步版本（使用aiohttp，3次指数退避重试）"""
    import os
    api_key = _SERPER_API_KEY
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = json.dumps({"q": query_with_type, "autocorrect": False})

    # 获取代理设置（aiohttp 不会自动使用系统代理）
    proxy = (
        os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY') or
        os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY') or
        _PROXY
    )

    for attempt in range(3):
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    _SERPER_URL,
                    headers=headers,
                    data=payload,
                    proxy=proxy,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    results = []
                    for r in data.get('organic', [])[:5]:
                        results.append({
                            "query": query_with_type,
                            "title": r.get("title", ""),
                            "url": r.get("link", ""),
                            "snippet": r.get("snippet", "")
                        })
                    return results
        except Exception as e:
            print(f"[搜索] 失败 (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    return []


async def _download_and_upload(file_url: str, tmp_dir: Path, sandbox_dir: str, session_id, env, page_limit: int = 40, size_limit: int = 2) -> bool:
    """下载并上传单个文件，只尝试一次

    Args:
        page_limit: 页数上限，超过则跳过
        size_limit: 文件大小上限，单位MB
    """
    # 检查URL后缀是否符合要求
    from urllib.parse import urlparse
    import os

    parsed = urlparse(file_url)
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext != '.pdf':
        print(f"跳过非PDF文件: {file_url}")
        return False

    try:
        local_path = await download_from_url(file_url, str(tmp_dir), max_size=size_limit * 1024 * 1024)
        if not local_path:
            print(f"[下载] 失败")
            return False

        # 检查页数
        try:
            from pypdf import PdfReader
            reader = PdfReader(local_path)
            num_pages = len(reader.pages)
            if num_pages > page_limit:
                print(f"⚠️ 页数过多 ({num_pages}页)，超过阈值 ({page_limit}页)，跳过上传")
                os.remove(local_path)
                return False
        except Exception as e:
            print(f"⚠️ 无法读取PDF页数: {e}，跳过上传")
            os.remove(local_path)
            return False

        file_name = Path(local_path).name
        sandbox_path = f"{sandbox_dir}/{file_name}"
        await env.session_upload(
            session_id=session_id,
            local_path=local_path,
            remote_path=sandbox_path,
        )
        print(f"✅ 上传成功: {sandbox_path}")
        return True
    except Exception as e:
        print(f"[下载] 失败: {e}")
        return False


async def web_search_pdf(query: str, sandbox_dir: str = None, session_id=None, env=None, output_tmp_dir=None, num: int = None, search_mode: str = "sync", page_limit: int = 40, size_limit: int = 2, top: int = 5) -> Dict[str, Any]:
    """搜索PDF文件并可选上传到sandbox

    Args:
        query: 搜索关键词
        sandbox_dir: sandbox中上传文件的绝对路径
        session_id: sandbox会话ID
        env: Environment对象
        output_tmp_dir: 本地临时目录
        num: 最多下载的文件数量，默认3
        search_mode: "async" 异步并发 | "sync" 同步顺序
        page_limit: 页数上限，超过则跳过，默认40
        top: 搜索结果只看前几条，默认5

    Returns:
        搜索结果字典
    """
    query_with_type = f"{query} filetype:pdf"

    # 搜索
    if search_mode == "async":
        results = await _search_pdf_async(query_with_type)
    else:
        results = _search_pdf(query_with_type)

    # 只看前top条结果
    results = results[:top]

    # 如果提供了session_id、env、output_tmp_dir和sandbox_dir，则下载文件并上传到sandbox
    if session_id and env and output_tmp_dir and sandbox_dir:
        tmp_dir = Path(output_tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 确保sandbox目标目录存在
        await env.session_exec(
            session_id=session_id,
            command=f"mkdir -p {sandbox_dir}",
            combine_output=True,
        )

        uploaded = []
        titles = []
        snippets = []
        urls = []
        download_count = 0  # 成功下载计数器
        max_download = num if num is not None else DEFAULT_MAX_DOWNLOAD_COUNT

        if search_mode == "async":
            # 异步模式：并发下载所有候选文件
            tasks = []
            for r in results:
                file_url = r.get("url")
                if file_url:
                    tasks.append(_download_and_upload(file_url, tmp_dir, sandbox_dir, session_id, env, page_limit, size_limit))

            if tasks:
                outcomes = await asyncio.gather(*tasks, return_exceptions=True)
                for r, outcome in zip(results, outcomes):
                    if download_count >= max_download:
                        break
                    if isinstance(outcome, Exception) or not outcome:
                        continue
                    file_url = r.get("url", "")
                    file_name = Path(file_url.split('/')[-1]).name
                    sandbox_path = f"{sandbox_dir}/{file_name}"
                    r["sandbox_path"] = sandbox_path
                    uploaded.append(sandbox_path)
                    titles.append(r.get("title", ""))
                    snippets.append(r.get("snippet", ""))
                    urls.append(file_url)
                    download_count += 1
        else:
            # 同步模式：顺序下载
            for r in results:
                # 最多下载指定数量的成功文件
                if download_count >= max_download:
                    break

                file_url = r.get("url")
                if not file_url:
                    continue

                success = await _download_and_upload(file_url, tmp_dir, sandbox_dir, session_id, env, page_limit, size_limit)
                if success:
                    file_name = Path(file_url.split('/')[-1]).name
                    sandbox_path = f"{sandbox_dir}/{file_name}"
                    r["sandbox_path"] = sandbox_path
                    uploaded.append(sandbox_path)
                    titles.append(r.get("title", ""))
                    snippets.append(r.get("snippet", ""))
                    urls.append(file_url)
                    download_count += 1

    # 根据下载结果返回对应格式
    if uploaded:
        return {"status": "success", "file_path": uploaded, "title": titles, "snippet": snippets, "url": urls}
    elif not results:
        return {"status": "error", "content": "该query查找失败"}
    else:
        return {"status": "error", "content": "查找到的文件链接下载失败"}


if __name__ == "__main__":
    import asyncio

    env=Environment(api_key=load_settings()["sandbox_api_key"], tool_type="aio_sandbox", env="test",
                       log_file="sandbox.log", log_level="INFO")

    async def test():
        # 测试搜索功能
        result = await web_search_pdf(
            query="election law statutes historical case law compilation",
            sandbox_dir="/home/workspace/search_pdfs",
            session_id="0fe2a892bd4041",
            env=env,
            output_tmp_dir="tools/tmp"
        )
        print("Search results:")
        for r in result.get("search_results", [])[:3]:
            print(f"  - {r.get('title')}: {r.get('url')}")

    asyncio.run(test())