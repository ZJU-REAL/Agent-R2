"""销毁已注册的沙盒 session，从 checked 或 unchecked 目录删除。"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from sandbox.create import _create_env

async def destroy_session(session_id: str):
    env = _create_env()
    try:
        await env.session_destroy(session_id)
        print(f"[OK] destroyed {session_id}")
    except Exception as e:
        print(f"[ERR] {session_id}: {e}")

