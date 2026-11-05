"""view_image_text -- 基于LLM多模态能力的图片理解工具。"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，以便直接运行时能找到 llm 模块
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import Any, Dict

import yaml

from llm import call_model
from settings import load_settings
from llm.image_utils import mime_and_base64_from_image_url

# 加载配置
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data_sythesis" / "config.yaml"


def _load_config() -> Dict[str, Any]:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


VIEW_IMAGE_TEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "view_image_text",
        "description": "Understand an image using AI vision. Pass a query and a sandbox image path. Returns a text description of the image.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you want to understand about the image."},
                "image_path": {"type": "string", "description": "Image path in sandbox (e.g., /home/workspace/image.png). Only support .png, .jpg and .jpeg"},
            },
            "required": ["query", "image_path"],
        },
    },
}


async def view_image_text(
    query: str,
    image_path: str,
    *,
    family: str = None,
    model_id: str = None,
    sk: str = None,
    api_url: str = None,
) -> Dict[str, Any]:
    if family is None or model_id is None or sk is None:
        config = _load_config()
        family = family or config["vision_model"]["family"]
        model_id = model_id or config["vision_model"]["model_id"]
        sk = sk or load_settings()["llm_api_key"]
        api_url = api_url or load_settings()["llm_api_url"]

    # 将本地图片编码为 data URL
    try:
        mime, b64 = mime_and_base64_from_image_url(image_path)
    except (ValueError, FileNotFoundError, OSError) as e:
        return {"error": f"Failed to read image: {e}"}

    data_url = f"data:{mime};base64,{b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    try:
        result = await call_model(
            family=family,
            model_id=model_id,
            sk=sk,
            messages=messages,
            tools=[],
            api_url=api_url,
        )
        if result.get("error"):
            return {"error": result["error"]}
        return {"result": result.get("content", "")}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import asyncio

    image_path = str(Path(_PROJECT_ROOT) / "gdpval/reference_files/3f7e829bf6b5d99feac00f56d19ea7c4/Rural Area Clinic.png")
    query = "请描述这张图片的内容"

    result = asyncio.run(view_image_text(query, image_path))
    print(result)
