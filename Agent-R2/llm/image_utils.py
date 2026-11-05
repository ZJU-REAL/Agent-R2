"""图片 data URL 解析（不含 resize，resize 移到工具层）。"""
from __future__ import annotations
import base64
import copy
import mimetypes
from pathlib import Path
from typing import Any, Dict, Tuple


def mime_and_base64_from_image_url(url: str) -> Tuple[str, str]:
    if not isinstance(url, str) or not url:
        raise ValueError("empty image url")
    if url.startswith("data:"):
        mime, _, rest = url.partition(";base64,")
        if not rest:
            raise ValueError("invalid data URL (missing base64 payload)")
        return mime.replace("data:", "", 1), rest
    if url.startswith("http://") or url.startswith("https://"):
        raise ValueError("remote http(s) image URLs are not loaded here")
    path = Path(url).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"image path does not exist: {url}")
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return mime, b64


def openai_image_part_with_data_url(item: Dict[str, Any]) -> Dict[str, Any]:
    """本地路径 image_url → data URL（供 OpenAI 兼容 API）。"""
    if not isinstance(item, dict) or item.get("type") != "image_url":
        return item
    inner = item.get("image_url")
    if not isinstance(inner, dict):
        return item
    url = inner.get("url", "")
    if not isinstance(url, str) or not url:
        return item
    if url.startswith("data:") or url.startswith("http://") or url.startswith("https://"):
        return item
    try:
        mime, b64 = mime_and_base64_from_image_url(url)
    except (ValueError, FileNotFoundError, OSError):
        return item
    out = copy.deepcopy(item)
    out["image_url"] = {"url": f"data:{mime};base64,{b64}"}
    out.pop("image_source_path", None)
    return out
