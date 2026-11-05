"""_settings.py -- 复用 Agent-R2/settings/settings.yaml，URL / key / 代理只在那里填一次。

launch 脚本已把 Agent-R2/ 加入 PYTHONPATH，Ray job 通过 runtime_env 的 py_modules 上传 settings；
其他情况下（如直接运行 sandbox 脚本）按仓库结构定位 Agent-R2/。
"""
import sys
from pathlib import Path

try:
    from settings import load_settings
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Agent-R2"))
    from settings import load_settings

__all__ = ["load_settings"]
