"""沙盒 session 依赖安装 -- 对 unchecked 目录中的 session 安装依赖。

SDK 已升级为异步（session_exec、session_upload 均为 async）。
"""
from __future__ import annotations
import asyncio
import os
from pathlib import Path
from typing import List

from rle_open_sdk.sdk import Environment
from recipe.agent_r2._settings import load_settings

SANDBOX_API_KEY = load_settings()["sandbox_api_key"]
SANDBOX_LOG = str(Path(__file__).parent / "sandbox.log")
SESSION_UNCHECKED = Path(__file__).parent / "sandbox_sessions" / "unchecked"


def _create_env() -> Environment:
    return Environment(api_key=SANDBOX_API_KEY, tool_type="aio_sandbox", env="test",
                       log_file=SANDBOX_LOG, log_level="INFO")


async def _install_deps(env: Environment, sid: str, pip_packages: str = "",
                        apt_packages: str = "", npm_packages: str = "",
                        install_libreoffice: bool = False):
    """在已创建的 session 中安装依赖。"""
    print("\n" + "=" * 60)
    print(f"开始为 session 安装依赖：{sid}")
    print("=" * 60)

    if pip_packages:
        print("\n📦 安装 pip 包...")
        res = await env.session_exec(session_id=sid,
                                     command=f"pip install {pip_packages}",
                                     exec_dir="/", combine_output=True)
        print(res.get("output", ""))

    if apt_packages:
        print("\n📦 apt-get update...")
        res = await env.session_exec(session_id=sid, command="apt-get update",
                                     exec_dir="/", combine_output=True)
        print(res.get("output", ""))

        print("\n📦 安装 apt 包...")
        res = await env.session_exec(session_id=sid,
                                     command=f"apt-get install -y {apt_packages}",
                                     exec_dir="/", combine_output=True)
        print(res.get("output", ""))

    if npm_packages:
        print("\n📦 安装 npm 包...")
        res = await env.session_exec(session_id=sid,
                                     command=f"npm install -g {npm_packages}",
                                     exec_dir="/", combine_output=True)
        print(res.get("output", ""))

    if install_libreoffice:
        print("\n安装 LibreOffice...")
        local_file_path = "data_sythesis/sandbox/LibreOffice_26.2.1_Linux_x86-64_deb.tar.gz"
        remote_tar = "/tmp/libreoffice.tar.gz"
        dpkg_cmd = "dpkg -i /tmp/LibreOffice_26.2.1.2_Linux_x86-64_deb/DEBS/*.deb"
        soffice_target = "/opt/libreoffice26.2/program/soffice"

        print("  上传安装包...")
        await env.session_upload(session_id=sid, local_path=local_file_path,
                                  remote_path=remote_tar)
        print("  上传完成")

        base = os.path.basename(remote_tar)
        res = await env.session_exec(session_id=sid,
                                     command=f"cd /tmp && tar -zxvf {base}",
                                     exec_dir="/tmp", combine_output=True)
        print(res.get("output", ""))

        res = await env.session_exec(session_id=sid, command=dpkg_cmd,
                                     exec_dir="/", combine_output=True)
        print(res.get("output", ""))

        await env.session_exec(session_id=sid,
                                command=f"ln -sf {soffice_target} /usr/bin/soffice",
                                exec_dir="/", combine_output=True)

        res = await env.session_exec(session_id=sid, command="soffice --version",
                                     exec_dir="/", combine_output=True)
        print(res.get("output", ""))

    print("\n所有依赖安装完成！")


async def update_sessions(pip_packages: str = "", apt_packages: str = "",
                          npm_packages: str = "", install_libreoffice: bool = False) -> List[str]:
    """对 unchecked 目录中的所有 session 安装依赖。"""
    if not SESSION_UNCHECKED.is_dir():
        print(f"目录不存在: {SESSION_UNCHECKED}")
        return []

    session_ids = [p.read_text().strip() for p in sorted(SESSION_UNCHECKED.glob("*.id"))]
    if not session_ids:
        print(f"没有待安装的 session: {SESSION_UNCHECKED}")
        return []

    print(f"找到 {len(session_ids)} 个待安装 session")
    env = _create_env()
    for sid in session_ids:
        try:
            await _install_deps(env, sid, pip_packages, apt_packages, npm_packages,
                                install_libreoffice)
        except Exception as e:
            print(f"[ERROR] session {sid} 安装失败: {e}")
    return session_ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="为 unchecked 目录中的 session 安装依赖")
    parser.add_argument("--pip", type=str,
                        default="markitdown Pillow numpy pandas openpyxl scikit-learn pypdf pdfplumber reportlab pytesseract pdf2image")
    parser.add_argument("--apt", type=str, default="",
                        help="apt 包（镜像已包含 poppler-utils, qpdf, pandoc）")
    parser.add_argument("--npm", type=str,
                        default="pptxgenjs docx react react-dom react-icons sharp")
    parser.add_argument("--libreoffice", action="store_true",
                        help="安装 LibreOffice（镜像已包含，通常不需要）")
    args = parser.parse_args()
    asyncio.run(update_sessions(
        pip_packages=args.pip, apt_packages=args.apt,
        npm_packages=args.npm, install_libreoffice=args.libreoffice,
    ))
