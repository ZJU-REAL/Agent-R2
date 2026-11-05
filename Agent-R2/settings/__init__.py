"""settings -- 全局唯一的外部服务配置：读取同目录下的 settings.yaml。

Agent-R2/ 下各模块和 verl/recipe/agent_r2 都从这里取 LLM 网关、沙盒、Serper、代理等配置。
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
import yaml

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings() -> Dict[str, Any]:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
