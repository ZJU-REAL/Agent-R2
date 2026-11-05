"""view_image (multimodal) -- 从沙盒下载图片返回 base64，含 resize 逻辑。"""
from __future__ import annotations
import base64
import io
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_sythesis.sandbox.file_management import download_file_from_sandbox

logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
MAX_IMAGE_SIZE = 1800

VIEW_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "view_image",
        "description": "View an image file in the sandbox. Returns base64 image data.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string",
                             "description": "Image path in sandbox (e.g., /home/workspace/image.png). Only supports .png, .jpg, .jpeg, .gif, .bmp, .webp"}
            },
            "required": ["filename"],
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


async def view_image(filename: str, *, session_id=None, env=None, output_tmp_dir=None) -> Dict[str, Any]:
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp"}
    filename = str(filename).strip()
    if not filename:
        return {"error": "empty filename"}
    if session_id is None or not output_tmp_dir:
        return {"error": "view_image requires session_id, env, output_tmp_dir"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        return {"error": f"unsupported format '{ext}'"}
    try:
        filepath = await download_file_from_sandbox(env, session_id, filename, Path(output_tmp_dir) / Path(filename).name)
    except Exception as e:
        return {"error": f"download failed ({e})"}
    if filepath is not None:
        filepath = Path(filepath)
    if filepath is None or not filepath.is_file():
        return {"error": f"file not found in sandbox: {filename}"}
    try:
        img_data = filepath.read_bytes()
        mime = mime_map.get(ext, "image/png")
        new_mime, b64, resized = _resize_image_if_needed(img_data, mime)
        return {"type": "image", "filename": filename, "filepath": str(filepath),
                "mime_type": new_mime, "base64_data": b64, "resized": resized}
    except Exception as e:
        return {"error": f"read failed ({e})"}


if __name__ == "__main__":
    """本地测试：直接 mock download_file_from_sandbox，验证 view_image 基本流程。"""
    import asyncio
    import tempfile
    import unittest.mock as mock

    image_path = Path(__file__).resolve().parent.parent / "gdpval/reference_files/3f7e829bf6b5d99feac00f56d19ea7c4/Rural Area Clinic.png"
    if not image_path.exists():
        print(f"[SKIP] 测试图片不存在: {image_path}")
        raise SystemExit(0)

    async def _mock_download(env, session_id, remote_path, output_dir):
        import shutil
        dest = os.path.join(output_dir, os.path.basename(remote_path))
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(str(image_path), dest)
        return dest

    with tempfile.TemporaryDirectory() as tmp_dir:
        with mock.patch("tools.view_image_multimodal.download_file_from_sandbox", side_effect=_mock_download):
            result = asyncio.run(
                view_image(str(image_path), session_id="test-session", env=None, output_tmp_dir=tmp_dir)
            )
    print(result)
