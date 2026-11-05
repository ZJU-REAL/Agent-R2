"""沙盒 session 创建。"""
from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path
from typing import List

from rle_open_sdk.sdk import Environment
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Agent-R2/，用于导入 settings
from settings import load_settings

logger = logging.getLogger(__name__)

SANDBOX_API_KEY = load_settings()["sandbox_api_key"]
SANDBOX_IMAGE = "rle/aio-sandbox:terminal-0.0.1"
SANDBOX_LOG = str(Path(__file__).parent / "sandbox.log")


def _create_env() -> Environment:
    return Environment(api_key=SANDBOX_API_KEY, tool_type="aio_sandbox", env="test",
                       log_file=SANDBOX_LOG, log_level="INFO")


_INIT_DIRS = "mkdir -p /mnt/user-data/uploads /home/workspace /mnt/user-data/outputs"


async def _create_one_session(image: str = SANDBOX_IMAGE,
                              max_retries: int = 3,
                              retry_delay: float = 5.0) -> str:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            env = _create_env()
            res = await env.create_image_session(
                image_path=image, container_command="bash /opt/gem/run.sh",
                timeout=300, session_timeout=600, cpu_cores=4, memory=4)
            sid = str(res["session_id"])
            await env.session_exec(session_id=sid, commands=["bash", "-c", _INIT_DIRS], combine_output=True)
            return sid
        except Exception as e:
            last_exc = e
            logger.warning("session creation attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
    raise last_exc


async def create_sessions(count: int, *, max_concurrency: int = 32,
                          image: str = SANDBOX_IMAGE) -> List[str]:
    """异步并发创建 count 个 session，返回 session_id 列表。"""
    sem = asyncio.Semaphore(min(count, max_concurrency))

    async def _one():
        async with sem:
            return await _create_one_session(image)

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="创建沙盒 session")
    parser.add_argument("--count", type=int, default=1, help="创建数量")
    args = parser.parse_args()
    sids = asyncio.run(create_sessions(args.count))
    print(json.dumps(sids, indent=2))
