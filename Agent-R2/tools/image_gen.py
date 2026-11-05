"""image_gen -- 文生图工具。"""
from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from settings import load_settings
from tools.file_utils import download_file_from_file_id

logger = logging.getLogger(__name__)

_IMAGE_GEN_CFG = load_settings()["image_gen"]
_IMAGE_GEN_URL = _IMAGE_GEN_CFG["url"]
_AK = _IMAGE_GEN_CFG["ak"]
_SK = _IMAGE_GEN_CFG["sk"]
_APPID = _IMAGE_GEN_CFG["appid"]

IMAGE_GEN_TOOL = {
    "type": "function",
    "function": {
        "name": "image_gen",
        "description": "Text-to-image generation. Returns sandbox paths for generated images.",
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {"type": "string", "description": "Natural language description of the image."},
                "image_num": {"type": "integer", "description": "Number of images (default 1).", "default": 1},
                "sandbox_path": {"type": "string", "description": "Absolute target path in sandbox."},
            },
            "required": ["instruction", "sandbox_path"],
        },
    },
}


def _gen_auth() -> str:
    ts = str(int(round(time.time() * 1000)))
    token = _AK + "/" + ts + "/" + _SK
    sign = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return _AK + "/" + ts + "/" + sign


def _image_gen_request(instruction: str, image_num: int = 1) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "authorization": _gen_auth(), "X-Bd-Assistant-Appid": _APPID}
    data = {"instruction": instruction, "image_num": int(image_num)}
    for attempt in range(3):
        try:
            with requests.post(_IMAGE_GEN_URL, headers=headers, data=json.dumps(data), stream=True, timeout=300) as resp:
                if resp.status_code != 200:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    continue
                lines = [l.decode("utf-8", errors="replace").strip() for l in resp.iter_lines()
                         if l and not l.decode("utf-8", errors="replace").strip().startswith("event:")]
                if not lines:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    continue
                last = lines[-1]
                payload = json.loads(last[5:] if last.startswith("data:") else last)
                files = payload.get("data", {}).get("prompt", {}).get("files", [])
                return {"files": [{"file_id": f.get("file_id")} for f in files if isinstance(f, dict) and f.get("file_id")]}
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return {"error": "image_gen request failed"}


def _resolve_sandbox_image_targets(sandbox_path: str, n: int) -> List[str]:
    """解析sandbox目标路径"""
    sp = str(sandbox_path).strip()
    if not sp:
        return []

    if sp.endswith("/"):
        dir_path = Path(sp)
        return [str(dir_path / f"image_{i}.png") for i in range(n)]

    p = Path(sp)
    ext = p.suffix.lower()
    supported_ext = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

    if ext not in supported_ext:
        if n == 1:
            return [sp + ".png"]
        return [f"{sp}_{i}.png" for i in range(n)]

    if n == 1:
        return [sp]

    stem = p.stem
    return [str(p.with_name(f"{stem}_{i}{ext}")) for i in range(n)]


async def image_gen(instruction: str, image_num: int, sandbox_path: str,
                    session_id=None, env=None, output_tmp_dir=None) -> Dict[str, Any]:
    """生成图片并上传到sandbox"""
    if not sandbox_path or not str(sandbox_path).strip().startswith("/"):
        return {"error": "requires absolute sandbox_path"}
    if session_id is None or env is None or not output_tmp_dir:
        return {"error": "requires session_id, env and output_tmp_dir"}

    # 调用文生图服务
    gen_ret = await asyncio.to_thread(_image_gen_request, instruction, image_num)
    if gen_ret.get("error"):
        return gen_ret

    files = gen_ret.get("files", [])
    file_ids = [f["file_id"] for f in files if f.get("file_id")]
    if not file_ids:
        return {"error": "no file_id returned"}

    # 解析目标路径
    targets = _resolve_sandbox_image_targets(sandbox_path, len(file_ids))
    if len(targets) != len(file_ids):
        return {"error": "failed to resolve sandbox upload targets"}

    # 创建临时目录
    tmp_dir = Path(output_tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 确保sandbox目标目录存在
    parents = sorted({str(Path(t).parent) for t in targets})
    for parent in parents:
        await env.session_exec(
            session_id=session_id,
            command=f"mkdir -p {parent}",
            combine_output=True,
        )

    # 下载并上传
    uploaded = []
    errors = []
    for file_id, target_remote in zip(file_ids, targets):
        try:
            local_guess = tmp_dir / f"{file_id}.png"
            downloaded_path = await download_file_from_file_id(file_id, str(local_guess))
            if not downloaded_path:
                errors.append(f"{file_id}: download failed")
                continue

            await env.session_upload(
                session_id=session_id,
                local_path=str(downloaded_path),
                remote_path=target_remote,
            )
            uploaded.append({"sandbox_path": target_remote})
        except Exception as e:
            logger.exception("image_gen upload failed: %s", e)
            errors.append(f"{file_id}: {e}")

    out = {"files": uploaded}
    if errors:
        out["errors"] = errors
    return out


if __name__ == "__main__":
    import tempfile

    async def test():
        # 创建测试图片用于测试上传
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"fake png data for test")
            tmp_path = tmp.name

        print(f"Test file: {tmp_path}")
        print("Note: 需要真实file_id才能测试完整流程")

    asyncio.run(test())