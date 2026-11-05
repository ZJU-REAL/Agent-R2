"""沙盒文件操作：上传目录、下载 finish 产物。"""
from __future__ import annotations
import asyncio
import logging
import os
import tarfile
import aiohttp
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


async def _retry_session_upload(env: Any, session_id: str, local_path: str, remote_path: str) -> None:
    for attempt in range(3):
        try:
            await env.session_upload(session_id=session_id, local_path=local_path, remote_path=remote_path)
            if attempt > 0:
                logger.info("session_upload succeeded on attempt %d: %s -> %s", attempt + 1, local_path, remote_path)
            return
        except Exception as e:
            logger.warning("session_upload attempt %d/3 failed: %s", attempt + 1, e)
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)


async def _retry_session_download(env: Any, session_id: str, paths: List[str], auto_extract: bool) -> Dict:
    for attempt in range(3):
        try:
            result = await env.session_download(session_id=session_id, paths=paths, auto_extract=auto_extract) or {}
            if attempt > 0:
                logger.info("session_download succeeded on attempt %d: %s", attempt + 1, paths)
            return result
        except Exception as e:
            logger.warning("session_download attempt %d/3 failed: %s", attempt + 1, e)
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)


async def _retry_session_exec(env: Any, session_id: str, command: str, exec_dir: str, combine_output: bool) -> Dict[str, Any]:
    for attempt in range(3):
        try:
            result = await env.session_exec(session_id=session_id, command=command,
                                            exec_dir=exec_dir, combine_output=combine_output)
            if attempt > 0:
                logger.info("session_exec succeeded on attempt %d: %s", attempt + 1, command)
            return result
        except Exception as e:
            logger.warning("session_exec attempt %d/3 failed: %s", attempt + 1, e)
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)

async def upload_file_to_sandbox(env: Any, session_id: str, local_file: str,
                                 remote_path: str) -> str:
    """上传单个文件。"""
    await _retry_session_upload(env, session_id, local_file, remote_path)
    return remote_path


async def upload_files_to_sandbox(env: Any, session_id: str, local_files: List[str],
                                  local_root: str, remote_root: str = "/home/workspace") -> List[str]:
    """上传多个文件：打包成 tgz 后上传，沙盒内解压。返回各文件在沙盒中的路径列表。"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    remote_paths = []
    with tarfile.open(tmp_path, "w:gz") as tar:
        for f in local_files:
            if os.path.isdir(local_root):
                rel = os.path.relpath(f, local_root)
            else:
                rel = os.path.basename(f)
            tar.add(f, arcname=rel)
            remote_paths.append(f"{remote_root}/{rel}")
    await upload_tgz_and_extract(env, session_id, tmp_path,
                                 remote_tgz=f"{remote_root}/_upload.tar.gz",
                                 extract_dir=remote_root)
    os.unlink(tmp_path)
    return remote_paths


async def upload_tgz_and_extract(env: Any, session_id: str, local_tgz: str,
                                 remote_tgz: str, extract_dir: str = "/home/workspace") -> Dict[str, Any]:
    """上传已打包的 tgz 并在沙盒内解压。"""
    await _retry_session_upload(env, session_id, local_tgz, remote_tgz)
    return await _retry_session_exec(env, session_id,
                                     command=f"tar xzf {remote_tgz} -C {extract_dir} && rm -f {remote_tgz}",
                                     exec_dir=extract_dir, combine_output=True)


async def download_file_from_sandbox(env: Any, session_id: str, remote_path: str,
                                     output_dir: str) -> str | None:
    """下载单个文件到 output_dir，返回本地文件路径字符串。"""
    filename = os.path.basename(remote_path) or "download"
    dl = await _retry_session_download(env, session_id, [remote_path], True)
    urls, _ = _extract_urls_and_files(dl, filename)
    if not urls:
        return None
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, filename)
    await _fetch_url(urls[0], dest)
    return dest


async def download_files_from_sandbox(env: Any, session_id: str, remote_paths: List[str],
                                      save_dir: str) -> List[str]:
    """下载多个文件：走 tar.gz 打包下载后本地解压，返回解压后的文件路径列表。"""
    os.makedirs(save_dir, exist_ok=True)

    dl = await _retry_session_download(env, session_id, remote_paths, False)
    bos_urls, files = _extract_urls_and_files(dl, "result.tar.gz")
    if not bos_urls:
        return []
    tasks = []
    for idx, url in enumerate(bos_urls):
        name = files[idx] if idx < len(files) else f"download_{idx}.tar.gz"
        if not str(name).endswith(".tar.gz"):
            name = f"{name}.tar.gz"
        tar_path = os.path.join(save_dir, name)
        tasks.append(_fetch_and_extract_tar_gz(url, tar_path, save_dir))
    results = await asyncio.gather(*tasks)
    downloaded: List[str] = []
    for r in results:
        downloaded.extend(r)
    return downloaded


def _extract_urls_and_files(dl: Dict, default_name: str) -> Tuple[List[str], List[str]]:
    urls = list(dl.get("bos_urls") or [])
    files = list(dl.get("files") or [])
    if not urls and dl.get("bos_url"):
        urls = [str(dl["bos_url"])]
        files = [str(dl.get("file_name") or default_name)]
    return urls, files


async def _fetch_url(url: str, dest: str) -> str:
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        f.write(await resp.read())
            return dest
        except Exception as e:
            logger.warning("_fetch_url attempt %d/3 failed: %s", attempt + 1, e)
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)


async def _fetch_and_extract_tar_gz(url: str, tar_path: str, extract_dir: str) -> List[str]:
    await _fetch_url(url, tar_path)
    extracted = []
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        for m in tar.getmembers():
            if m.isfile():
                extracted.append(os.path.join(extract_dir, m.name))
    os.unlink(tar_path)
    return extracted


def bundle_dir_to_tgz(src_dir: str, dest_tgz: str, arcname: str = "skills") -> str:
    os.makedirs(os.path.dirname(dest_tgz), exist_ok=True)
    with tarfile.open(dest_tgz, "w:gz") as tar:
        tar.add(src_dir, arcname=arcname)
    return dest_tgz


def load_skill_files(skills_root: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    names, sections = [], []
    root_name = os.path.basename(skills_root)
    if os.path.isdir(skills_root):
        for skill_md in sorted(
            os.path.join(dirpath, "SKILL.md")
            for dirpath, _, filenames in os.walk(skills_root)
            if "SKILL.md" in filenames
        ):
            rel = os.path.relpath(os.path.dirname(skill_md), skills_root)
            has_scripts = os.path.isdir(os.path.join(os.path.dirname(skill_md), "scripts"))
            names.append((rel, has_scripts))
            with open(skill_md, encoding="utf-8", errors="replace") as fh:
                sections.append(f"--- {rel}/SKILL.md ---\n{fh.read()}\n")
    name_lines = []
    for n, has_scripts in names:
        line = f"- {n}（SKILL.md路径：/home/workspace/{root_name}/{n}/SKILL.md"
        if has_scripts:
            line += f"，scripts路径：/home/workspace/{root_name}/{n}/scripts/"
        line += "）"
        name_lines.append(line)
    out["SKILL_NAMES"] = "\n".join(name_lines)
    out["ALL_SKILLS"] = "\n".join(sections)
    return out
