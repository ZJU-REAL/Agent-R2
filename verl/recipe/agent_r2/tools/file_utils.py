"""file_utils -- 文件下载工具。"""
from __future__ import annotations
import asyncio
import os
import traceback
from codecs import encode
from urllib.parse import unquote
from typing import Optional


def _sign_file(ak: str, sk: str) -> str:
    """生成文件上传授权签名"""
    import time
    current_time = int(time.time() * 1000)
    sign_key_info = f"{ak}/{current_time}/{sk}"
    import hashlib
    hash_value = hashlib.sha256(sign_key_info.encode('utf-8')).digest()
    signature = encode(hash_value, 'hex').decode('utf-8')
    return f"{ak}/{current_time}/{signature}"


async def download_file_from_file_id(file_id: str, output_path: str, file_server_url: str,
                                     ak: str = "", sk: str = "") -> Optional[str]:
    """
    从文件服务器下载文件（根据file_id）

    Args:
        file_id: 文件ID
        output_path: 文件保存的目标路径
        file_server_url: 文件服务地址（.../v1/files）
        ak, sk: 文件服务签名凭证

    Returns:
        下载文件的最终路径，失败返回None
    """
    import aiohttp

    try:
        url = f"{file_server_url}/{file_id}/download"

        headers = {
            "X-BD-LOGID": "3333",
            "Authorization": _sign_file(ak, sk)
        }

        # 连接超时3秒，下载不限时
        timeout = aiohttp.ClientTimeout(connect=3)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    content_disposition = response.headers.get('Content-Disposition', '')
                    if 'filename=' in content_disposition:
                        file_name = content_disposition.split('filename=')[1]
                        file_name = unquote(file_name).strip('"')
                    else:
                        file_name = os.path.basename(output_path)

                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)

                    final_output_path = os.path.join(output_dir, file_name) if file_name else output_path

                    with open(final_output_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)

                    return final_output_path
                else:
                    print(f"Failed to download file {file_id}: HTTP status {response.status}")
                    return None
    except Exception as e:
        print(f"Error downloading file {file_id}: {e}")
        traceback.print_exc()
        return None


# 多个 User-Agent 轮换，失败时自动切换重试
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "BaiduResearch contact@example.com",
]


# 默认文件大小上限: 1MB
DEFAULT_MAX_FILE_SIZE = 1 * 1024 * 1024


async def _check_file_size(url: str, max_size: Optional[int] = DEFAULT_MAX_FILE_SIZE) -> int | None:
    """用 HEAD 请求检查文件大小，返回 Content-Length 或 None。
    如果超过 max_size 则抛出 ValueError。"""
    import aiohttp

    # 获取代理设置（aiohttp 不会自动使用系统代理）
    proxy = (
        os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY') or
        os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
    )

    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.head(url, allow_redirects=True, proxy=proxy) as resp:
                if resp.status == 200:
                    content_length = resp.headers.get("Content-Length")
                    if content_length:
                        size = int(content_length)
                        if max_size is not None and size > max_size:
                            raise ValueError(
                                f"文件过大 ({size / 1024 / 1024:.1f}MB)，"
                                f"超过阈值 ({max_size / 1024 / 1024:.0f}MB)，跳过下载"
                            )
                        return size
    except ValueError:
        raise
    except Exception:
        pass
    return None


async def _curl_download(url: str, final_path: str, env: dict) -> bool:
    """使用curl下载文件，返回是否成功"""
    import asyncio
    import subprocess

    # --connect-timeout: 连接超时3秒，下载不限时
    cmd = ["curl", "-o", final_path, "-s", "-f", "--connect-timeout", "3", url]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
        )
        _, _ = await proc.communicate()

        if proc.returncode == 0 and os.path.exists(final_path) and os.path.getsize(final_path) > 0:
            return True
        return False
    except Exception:
        return False


async def download_from_url(url: str, output_dir: str, max_size: Optional[int] = DEFAULT_MAX_FILE_SIZE) -> Optional[str]:
    """
    从URL下载文件（使用curl）
    策略：先用HEAD请求检查文件大小，超过max_size则跳过；
    再正常下载（系统默认代理），失败后取消代理重试

    Args:
        url: 文件URL
        output_dir: 本地保存目录
        max_size: 文件大小上限（字节），超过则跳过下载，默认50MB
    """
    import subprocess

    os.makedirs(output_dir, exist_ok=True)

    # 先用 HEAD 请求检查文件大小
    try:
        await _check_file_size(url, max_size)
    except ValueError as e:
        print(f"⚠️ {e}")
        return None

    file_name = url.split('/')[-1].split('?')[0] or "downloaded_file"
    final_path = os.path.join(output_dir, file_name)

    # 不含代理的环境变量
    env_no_proxy = os.environ.copy()
    for proxy_key in ['https_proxy', 'http_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        env_no_proxy.pop(proxy_key, None)

    # 先正常下载（使用系统默认代理）
    if await _curl_download(url, final_path, os.environ.copy()):
        print(f"✅ 下载成功: {final_path}")
        return final_path

    # 失败后尝试不用代理
    print(f"[代理] 失败，尝试不用代理下载...")
    if await _curl_download(url, final_path, env_no_proxy):
        print(f"✅ 不用代理下载成功: {final_path}")
        return final_path

    print(f"❌ 下载失败: {url}")
    # 清理失败的临时文件
    if os.path.exists(final_path):
        try:
            os.remove(final_path)
        except Exception:
            pass
    return None


if __name__ == "__main__":
    # 测试下载 PDF
    url = "https://www.elections.virginia.gov/media/grebhandbook/2024-updates/15_Recounts_and_Contests_2024_FINAL.pdf"
    output_dir = "tools/tmp"

    result = asyncio.run(download_from_url(url, output_dir))
    print(f"测试结果: {result}")