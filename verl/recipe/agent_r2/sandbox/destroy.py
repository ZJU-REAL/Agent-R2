"""销毁已注册的沙盒 session，从 checked 或 unchecked 目录删除。

SDK 已升级为异步（session_destroy 为 async）。
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
from typing import List

from recipe.agent_r2.sandbox.create import _create_env, SESSION_CHECKED, SESSION_UNCHECKED


def _list_sessions(dir_path: Path) -> List[str]:
    if not dir_path.is_dir():
        return []
    return [p.read_text().strip() for p in sorted(dir_path.glob("*.id"))]


async def destroy_session(session_id: str):
    env = _create_env()
    try:
        await env.session_destroy(session_id)
        print(f"[OK] destroyed {session_id}")
    except Exception as e:
        print(f"[ERR] {session_id}: {e}")

    # 从 checked 目录删除
    checked_file = SESSION_CHECKED / f"{session_id}.id"
    if checked_file.is_file():
        checked_file.unlink()
        print(f"[DEL] {session_id} from checked")

    # 从 unchecked 目录删除
    unchecked_file = SESSION_UNCHECKED / f"{session_id}.id"
    if unchecked_file.is_file():
        unchecked_file.unlink()
        print(f"[DEL] {session_id} from unchecked")


async def destroy_all():
    """销毁所有 session（checked 和 unchecked）"""
    for sid in _list_sessions(SESSION_CHECKED):
        await destroy_session(sid)
    for sid in _list_sessions(SESSION_UNCHECKED):
        await destroy_session(sid)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        async def _main():
            for sid in sys.argv[1:]:
                await destroy_session(sid.strip())
        asyncio.run(_main())
    else:
        asyncio.run(destroy_all())
