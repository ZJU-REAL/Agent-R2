"""view_image (multimodal) -- 从沙盒下载图片返回 base64，含 resize 逻辑。"""
from __future__ import annotations
import base64
import io
import logging
import os
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
MAX_IMAGE_SIZE = 1800

VIEW_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "view_image",
        "description": "View image files in the sandbox. Pass filenames as an array. Returns base64 image data.",
        "parameters": {
            "type": "object",
            "properties": {
                "filenames": {"type": "array", "items": {"type": "string"},
                              "description": "Image paths in sandbox (basename resolves to /home/gem/)."}
            },
            "required": ["filenames"],
        },
    },
}


def _resize_image_if_needed(img_data: bytes, mime: str) -> tuple:
    """Resize 图片使长宽不超过 MAX_IMAGE_SIZE。返回 (new_mime, b64_str, was_resized)。"""
    try:
        from PIL import Image
    except ImportError:
        return mime, base64.b64encode(img_data).decode("utf-8"), False
    img = Image.open(io.BytesIO(img_data))
    w, h = img.size
    if w <= MAX_IMAGE_SIZE and h <= MAX_IMAGE_SIZE:
        return mime, base64.b64encode(img_data).decode("utf-8"), False
    if w > h:
        nw, nh = MAX_IMAGE_SIZE, int(h * MAX_IMAGE_SIZE / w)
    else:
        nh, nw = MAX_IMAGE_SIZE, int(w * MAX_IMAGE_SIZE / h)
    img_resized = img.resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO()
    fmt = mime.split("/")[-1].upper()
    if fmt not in ("JPEG", "PNG", "GIF", "WEBP"):
        fmt = "PNG"
    try:
        img_resized.save(buf, format=fmt)
    except Exception:
        img_resized.save(buf, format="PNG")
        fmt = "PNG"
    new_mime = f"image/{fmt.lower()}"
    return new_mime, base64.b64encode(buf.getvalue()).decode("utf-8"), True


def _normalize_sandbox_path(filename: str) -> str:
    p = str(filename).strip()
    if not p:
        return ""
    return p if p.startswith("/") else f"/home/gem/{p}"


def _download_sandbox_image(env, session_id, remote_path, tmp_dir):
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    want_name = Path(remote_path).name
    dl = env.session_download(session_id=session_id, paths=[remote_path], auto_extract=True) or {}
    urls = list(dl.get("bos_urls") or [])
    files = list(dl.get("files") or [])
    if not urls and dl.get("bos_url"):
        urls = [str(dl["bos_url"])]
        files = [str(dl.get("file_name") or want_name)]
    if urls:
        for idx, url in enumerate(urls):
            name = Path(files[idx] if idx < len(files) else want_name).name
            dest = tmp_dir / name
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            if name == want_name:
                return dest
        return tmp_dir / (files[0] if files else want_name) if len(urls) == 1 else None
    # fallback: tar.gz
    dl2 = env.session_download(session_id=session_id, paths=[remote_path], auto_extract=False) or {}
    bos_url = dl2.get("bos_url") or (dl2.get("bos_urls") or [None])[0]
    if not bos_url:
        return None
    fname = str(dl2.get("file_name") or "view_image.tar.gz")
    tar_path = tmp_dir / fname
    r = requests.get(bos_url, timeout=120)
    r.raise_for_status()
    tar_path.write_bytes(r.content)
    extracted = []
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=tmp_dir)
        for m in tar.getmembers():
            if m.isfile():
                extracted.append(tmp_dir / m.name)
    tar_path.unlink(missing_ok=True)
    for p in extracted:
        if p.name == want_name:
            return p
    return extracted[0] if len(extracted) == 1 else None


async def view_image(filenames: List[str], *, session_id=None, env=None, output_tmp_dir=None) -> Dict[str, Any]:
    results, errors = [], []
    if not filenames or not any(str(f).strip() for f in filenames):
        return {"type": "image", "images": results, "errors": ["empty filenames"]}
    if session_id is None or env is None or not output_tmp_dir:
        return {"type": "image", "images": results, "errors": ["view_image requires session_id, env, output_tmp_dir"]}
    tmp_dir = Path(output_tmp_dir)
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp"}
    for filename in filenames:
        filename = str(filename).strip()
        if not filename:
            errors.append("empty filename")
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            errors.append(f"'{filename}': unsupported format '{ext}'")
            continue
        remote = _normalize_sandbox_path(filename)
        try:
            filepath = _download_sandbox_image(env, session_id, remote, tmp_dir)
        except Exception as e:
            errors.append(f"'{filename}': download failed ({e})")
            continue
        if filepath is None or not filepath.is_file():
            errors.append(f"'{filename}': file not found in sandbox")
            continue
        try:
            img_data = filepath.read_bytes()
            mime = mime_map.get(ext, "image/png")
            new_mime, b64, resized = _resize_image_if_needed(img_data, mime)
            results.append({"filename": filename, "filepath": str(filepath),
                            "mime_type": new_mime, "base64_data": b64, "resized": resized})
        except Exception as e:
            errors.append(f"'{filename}': read failed ({e})")
    out = {"type": "image", "images": results}
    if errors:
        out["errors"] = errors
    return out
