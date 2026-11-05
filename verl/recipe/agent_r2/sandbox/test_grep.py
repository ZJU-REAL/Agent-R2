"""测试 grep 命令的 return_code"""
import asyncio
import json
from pathlib import Path

from rle_open_sdk.sdk import Environment
from recipe.agent_r2._settings import load_settings

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


def _unregister(session_id: str):
    """从 unchecked 目录删除 session 注册"""
    unchecked_file = SESSION_UNCHECKED / f"{session_id}.id"
    if unchecked_file.is_file():
        unchecked_file.unlink()


async def test_grep_return_code():
    # 1. 创建 session
    print("[1] 创建 session...")
    env = _create_env()
    res = await env.create_image_session(
        image_path=SANDBOX_IMAGE,
        container_command="bash /opt/gem/run.sh",
        timeout=300,
        session_timeout=120,
        cpu_cores=2,
        memory=4
    )
    sid = str(res["session_id"])
    print(f"    session_id: {sid}")

    # 初始化目录
    await env.session_exec(session_id=sid, command=_INIT_DIRS,
                           exec_dir="/", combine_output=True)

    # 2. 执行 grep 命令（返回空时 - 搜索不存在的字符串）
    print("[2] 执行 grep 命令（搜索不存在的字符串）...")
    result = await env.session_exec(
        session_id=sid,
        command="grp 'THIS_STRING_DOES_NOT_EXIST_12345' /etc/passwd",
        exec_dir="/",
        combine_output=True
    )
    print(f"    return_code: {result.get('return_code')}")
    print(f"    stdout: {result.get('stdout')[:200] if result.get('stdout') else 'None'}...")
    print(f"    stderr: {result.get('stderr')}")
    print(result)
    # 3. 销毁 session
    print("[3] 销毁 session...")
    await env.session_destroy(sid)
    _unregister(sid)
    print(f"    已销毁 {sid}")

    return result.get('return_code')


if __name__ == "__main__":
    return_code = asyncio.run(test_grep_return_code())
    print(f"\n最终 return_code: {return_code}")