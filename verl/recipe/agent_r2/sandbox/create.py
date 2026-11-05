"""沙盒 session 创建 -- 仅创建 session，保存到 unchecked 目录。

SDK 已升级为异步（create_image_session、session_exec 均为 async）。
"""
from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path
from typing import List

from rle_open_sdk.sdk import Environment
from recipe.agent_r2._settings import load_settings

logger = logging.getLogger(__name__)

SANDBOX_API_KEY = load_settings()["sandbox_api_key"]
SANDBOX_IMAGE = "rle/aio-sandbox:terminal-0.0.1"
SANDBOX_LOG = str(Path(__file__).parent / "sandbox.log")
SESSION_DIR = Path(__file__).parent / "sandbox_sessions"
SESSION_UNCHECKED = SESSION_DIR / "unchecked"
SESSION_CHECKED = SESSION_DIR / "checked"

_INIT_DIRS = "mkdir -p /mnt/user-data/uploads /home/workspace /mnt/user-data/outputs"


def _create_env() -> Environment:
    return Environment(api_key=SANDBOX_API_KEY, tool_type="aio_sandbox", env="test",
                       log_file=SANDBOX_LOG, log_level="INFO")


def _register(session_id: str):
    """注册 session 到 unchecked 目录"""
    SESSION_UNCHECKED.mkdir(parents=True, exist_ok=True)
    (SESSION_UNCHECKED / f"{session_id}.id").write_text(session_id + "\n")


async def _create_one_session(image: str = SANDBOX_IMAGE,
                              max_retries: int = 3,
                              retry_delay: float = 5.0) -> str:
    """创建单个 session，带重试。"""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            env = _create_env()
            res = await env.create_image_session(
                image_path=image, container_command="bash /opt/gem/run.sh",
                timeout=300, session_timeout=120, cpu_cores=2, memory=4)
            sid = str(res["session_id"])
            await env.session_exec(session_id=sid, command=_INIT_DIRS,
                                   exec_dir="/", combine_output=True)
            return sid
        except Exception as e:
            last_exc = e
            logger.warning("session creation attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
    raise last_exc


async def create_sessions(count: int, *, max_concurrency: int = 4,
                          image: str = SANDBOX_IMAGE) -> List[str]:
    """异步并发创建 count 个 session，返回 session_id 列表。"""
    sem = asyncio.Semaphore(min(count, max_concurrency))

    async def _one():
        async with sem:
            sid = await _create_one_session(image)
            _register(sid)
            return sid

    tasks = [asyncio.create_task(_one()) for _ in range(count)]
    results = []
    last_exc: Exception | None = None
    for fut in asyncio.as_completed(tasks):
        try:
            results.append(await fut)
        except Exception as e:
            last_exc = e
            logger.exception("session creation failed")
    if not results and last_exc is not None:
        raise last_exc
    return results


def list_unchecked_sessions() -> List[str]:
    if not SESSION_UNCHECKED.is_dir():
        return []
    return [p.read_text().strip() for p in sorted(SESSION_UNCHECKED.glob("*.id"))]


def list_checked_sessions() -> List[str]:
    if not SESSION_CHECKED.is_dir():
        return []
    return [p.read_text().strip() for p in sorted(SESSION_CHECKED.glob("*.id"))]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="创建沙盒 session（仅创建，不安装依赖）")
    parser.add_argument("--count", type=int, default=1, help="创建数量")
    args = parser.parse_args()
    sids = asyncio.run(create_sessions(args.count))
    print(json.dumps(sids, indent=2))
