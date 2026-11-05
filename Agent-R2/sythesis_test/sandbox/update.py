"""沙盒 session 依赖安装 -- 对 unchecked 目录中的 session 安装依赖。"""
from __future__ import annotations
import asyncio
import os
from pathlib import Path
from typing import List

from rle_open_sdk.sdk import Environment
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Agent-R2/，用于导入 settings
from settings import load_settings

SANDBOX_API_KEY = load_settings()["sandbox_api_key"]
SANDBOX_LOG = str(Path(__file__).parent / "sandbox.log")
SESSION_UNCHECKED = Path(__file__).parent / "sandbox_sessions" / "unchecked"


def _create_env() -> Environment:
    return Environment(api_key=SANDBOX_API_KEY, tool_type="aio_sandbox", env="test",
                       log_file=SANDBOX_LOG, log_level="INFO")


async def _install_deps(env: Environment, sid: str, pip_packages: str = "", apt_packages: str = "",
                        npm_packages: str = "", install_libreoffice: bool = False):
    """在已创建的 session 中安装依赖。"""
    print("\n" + "=" * 60)
    print(f"🚀 开始为 session 安装依赖：{sid}")
    print("=" * 60)

    if pip_packages:
        print("\n" + "=" * 60)
        print("🚀 开始预装 pip 依赖（分步执行）")
        print(f"会话 ID: {sid}")
        print("=" * 60)

        print("\n📦 步骤 2/2：安装 pip 包...")
        res = await env.session_exec(session_id=sid, commands=["bash", "-c", f"pip install {pip_packages}"],
                                     combine_output=True)
        print(res.get("output", ""))

    if apt_packages:
        print("\n" + "=" * 60)
        print("🚀 开始全自动安装 apt 依赖")
        print(f"会话 ID: {sid}")
        print("=" * 60)

        print("\n📦 步骤 1/2：apt-get update...")
        res = await env.session_exec(session_id=sid, commands=["bash", "-c", "apt-get update"], combine_output=True)
        print(res.get("output", ""))

        print("\n📦 步骤 2/2：安装 apt 包...")
        res = await env.session_exec(session_id=sid, commands=["bash", "-c", f"apt-get install -y {apt_packages}"],
                                     combine_output=True)
        print(res.get("output", ""))

    if npm_packages:
        print("\n" + "=" * 60)
        print("🚀 开始预装 npm 依赖")
        print(f"会话 ID: {sid}")
        print("=" * 60)

        print("\n📦 步骤 2/2：安装 npm 包...")
        res = await env.session_exec(session_id=sid, commands=["bash", "-c", f"npm install -g {npm_packages}"],
                                     combine_output=True)
        print(res.get("output", ""))

    if install_libreoffice:
        print("\n" + "=" * 60)
        print("🚀 开始安装 LibreOffice + 配置全局 soffice 命令")
        print(f"会话 ID: {sid}")
        print("=" * 60)

        local_file_path = "sythesis_test/sandbox/LibreOffice_26.2.1_Linux_x86-64_deb.tar.gz"
        remote_tar = "/tmp/libreoffice.tar.gz"
        dpkg_cmd = "dpkg -i /tmp/LibreOffice_26.2.1.2_Linux_x86-64_deb/DEBS/*.deb"
        soffice_target = "/opt/libreoffice26.2/program/soffice"

        print("\n📥 步骤 1/5: 上传本地 LibreOffice 安装包...")
        await env.session_upload(
            session_id=sid,
            local_path=local_file_path,
            remote_path=remote_tar,
        )
        print("✅ 上传完成！")

        print("\n🔍 步骤 2/5: 检查上传文件是否正常...")
        res = await env.session_exec(
            session_id=sid,
            commands=["bash", "-c", f"ls -lh {remote_tar}"],
            combine_output=True,
        )
        print("📌 文件信息：")
        print(res.get("output", ""))

        print("\n📦 步骤 3/5: 解压安装包...")
        base = os.path.basename(remote_tar)
        res = await env.session_exec(
            session_id=sid,
            commands=["bash", "-c", f"cd /tmp && tar -zxvf {base}"],
            combine_output=True,
        )
        print("✅ 解压完成！输出：")
        print(res.get("output", ""))

        print("\n📂 检查解压后的目录结构：")
        res = await env.session_exec(
            session_id=sid,
            commands=["bash", "-c", "ls -lh /tmp/"],
            combine_output=True,
        )
        print(res.get("output", ""))

        print("\n🔧 步骤 4/5: 安装 LibreOffice...")
        res = await env.session_exec(
            session_id=sid,
            commands=["bash", "-c", dpkg_cmd],
            combine_output=True,
        )
        print("✅ 安装完成！输出：")
        print(res.get("output", ""))

        print("\n🔗 步骤 5/5: 创建全局软链接...")
        await env.session_exec(
            session_id=sid,
            commands=["bash", "-c", f"ln -sf {soffice_target} /usr/bin/soffice"],
            combine_output=True,
        )

        print("\n🎉 最终测试：soffice --version")
        res = await env.session_exec(
            session_id=sid,
            commands=["bash", "-c", "soffice --version"],
            combine_output=True,
        )
        print(res.get("output", ""))

    print("\n✅ 所有依赖安装完成！")


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
            await _install_deps(env, sid, pip_packages, apt_packages, npm_packages, install_libreoffice)
        except Exception as e:
            print(f"[ERROR] session {sid} 安装失败: {e}")
    return session_ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="为 unchecked 目录中的 session 安装依赖")
    parser.add_argument("--pip", type=str, default="markitdown Pillow numpy pandas openpyxl scikit-learn pypdf pdfplumber reportlab pytesseract pdf2image")
    parser.add_argument("--apt", type=str, default="", help="apt 包（镜像已包含 poppler-utils, qpdf, pandoc）")
    parser.add_argument("--npm", type=str, default="pptxgenjs docx react react-dom react-icons sharp")
    parser.add_argument("--libreoffice", action="store_true", help="安装 LibreOffice（镜像已包含，通常不需要）")
    args = parser.parse_args()
    asyncio.run(update_sessions(pip_packages=args.pip, apt_packages=args.apt, npm_packages=args.npm,
                                install_libreoffice=args.libreoffice))
