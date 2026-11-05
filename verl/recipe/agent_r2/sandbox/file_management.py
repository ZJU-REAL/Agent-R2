"""沙盒文件操作：上传目录、下载 finish 产物。

函数签名和行为与 project/gdpval/sandbox/file_management.py 保持一致。
额外包含 sandbox session 管理：init_sandbox / create_session / SandboxSession。
"""
from __future__ import annotations
import asyncio
import logging
import os
import tarfile
import aiohttp
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Low-level retry helpers
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Upload helpers
# ─────────────────────────────────────────────────────────────

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
            local_size = os.path.getsize(f) if os.path.exists(f) else -1
            logger.warning(
                "[upload_files_to_sandbox] packing: local=%s (size=%d) -> remote=%s/%s",
                f, local_size, remote_root, rel,
            )
            tar.add(f, arcname=rel)
            remote_paths.append(f"{remote_root}/{rel}")
    tgz_size = os.path.getsize(tmp_path)
    logger.warning(
        "[upload_files_to_sandbox] uploading tgz: %s (size=%d) -> session=%s remote_tgz=%s/_upload.tar.gz",
        tmp_path, tgz_size, session_id, remote_root,
    )
    try:
        result = await upload_tgz_and_extract(env, session_id, tmp_path,
                                              remote_tgz=f"{remote_root}/_upload.tar.gz",
                                              extract_dir=remote_root)
        logger.warning(
            "[upload_files_to_sandbox] upload+extract done | session=%s remote_root=%s "
            "files=%s result=%s",
            session_id, remote_root, remote_paths, result,
        )
    except Exception as e:
        logger.warning(
            "[upload_files_to_sandbox] upload+extract FAILED | session=%s remote_root=%s error=%s",
            session_id, remote_root, e,
        )
        raise
    finally:
        os.unlink(tmp_path)
    return remote_paths


async def upload_tgz_and_extract(env: Any, session_id: str, local_tgz: str,
                                 remote_tgz: str, extract_dir: str = "/home/workspace") -> Dict[str, Any]:
    """上传已打包的 tgz 并在沙盒内解压。"""
    await _retry_session_upload(env, session_id, local_tgz, remote_tgz)
    return await _retry_session_exec(env, session_id,
                                     command=f"tar xzf {remote_tgz} -C {extract_dir} && rm -f {remote_tgz}",
                                     exec_dir=extract_dir, combine_output=True)


# ─────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Sandbox session management (init_sandbox / create_session / SandboxSession)
# ─────────────────────────────────────────────────────────────

from pathlib import Path
from typing import Optional

logger_session = logging.getLogger(__name__ + ".session")
logger_session.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SANDBOX_LOG_FILE = os.environ.get(
    "SANDBOX_LOG_FILE",
    str(Path(__file__).resolve().parent.parent / "sandbox.log"),
)
RUN_LOG_FILE = os.environ.get(
    "RUN_LOG_FILE",
    str(Path(__file__).resolve().parent.parent / "run.log"),
)

DEFAULT_SANDBOX_IMAGE = os.environ.get(
    "SANDBOX_IMAGE",
    "rle/aio-sandbox:terminal-0.0.1",
)
DEFAULT_SANDBOX_TOOL_TYPE = os.environ.get("SANDBOX_TOOL_TYPE", "aio_sandbox")
DEFAULT_SANDBOX_ENV = os.environ.get("SANDBOX_ENV", "test")

_INIT_DIRS = "mkdir -p /mnt/user-data/uploads /home/workspace /mnt/user-data/outputs"

_env = None
_sandbox_config: dict = {}
_initialized: bool = False


def init_sandbox(config: Optional[dict] = None):
    """Initialize the shared Environment and logging handlers.

    Called once at class init time (RealAgentLoop.init_class).
    Safe to call multiple times — only the first call has effect.
    """
    global _initialized, _env, _sandbox_config
    if _initialized:
        return
    _initialized = True

    cfg = config or {}
    _sandbox_config = cfg

    from recipe.agent_r2._settings import load_settings
    api_key = cfg.get("api_key") or load_settings()["sandbox_api_key"]
    tool_type = cfg.get("tool_type", DEFAULT_SANDBOX_TOOL_TYPE)
    env_name = cfg.get("env", DEFAULT_SANDBOX_ENV)

    run_log = cfg.get("run_log_file", RUN_LOG_FILE)
    sandbox_log = cfg.get("log_file", SANDBOX_LOG_FILE)

    root_logger = logging.getLogger()
    if run_log and not any(
        isinstance(h, logging.FileHandler) and
        h.baseFilename == str(Path(run_log).resolve())
        for h in root_logger.handlers
    ):
        run_fh = logging.FileHandler(run_log, encoding="utf-8")
        run_fh.setLevel(logging.INFO)
        run_fh.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(run_fh)

    try:
        from rle_open_sdk.sdk import Environment
        _env = Environment(
            api_key=api_key,
            tool_type=tool_type,
            env=env_name,
            log_level="INFO",
            log_file=sandbox_log,
        )
        logger_session.warning(f"[sandbox] Environment created (tool_type={tool_type}, env={env_name})")
    except Exception as e:
        logger_session.error(f"[sandbox] Failed to create Environment: {e}")
        raise


async def create_session() -> "SandboxSession":
    """Create a fresh sandbox session for one trajectory."""
    if _env is None:
        raise RuntimeError(
            "sandbox not initialized. Call init_sandbox() at class init time."
        )

    image = _sandbox_config.get("image", DEFAULT_SANDBOX_IMAGE)
    max_retries = 3
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            res = await _env.create_image_session(
                image_path=image,
                container_command="bash /opt/gem/run.sh",
                timeout=300,
                session_timeout=_sandbox_config.get("session_timeout", 3600),
                cpu_cores=_sandbox_config.get("cpu_cores", 1),
                memory=_sandbox_config.get("memory", 4),
            )
            session_id = str(res["session_id"])
            await _env.session_exec(
                session_id=session_id,
                command=_INIT_DIRS,
                exec_dir="/",
                combine_output=True,
            )
            logger_session.warning(f"[sandbox] Created session {session_id}")
            return SandboxSession(env=_env, session_id=session_id)
        except Exception as e:
            last_exc = e
            logger_session.warning(
                f"[sandbox] create_session attempt {attempt + 1}/{max_retries} failed: {e}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(5.0 * (attempt + 1))

    raise RuntimeError(f"Failed to create sandbox session after {max_retries} attempts: {last_exc}")


class SandboxSession:
    """A sandbox session for one trajectory.

    Created at trajectory start, destroyed at trajectory end via stop().
    """

    def __init__(self, env, session_id: str):
        self.env = env
        self.session_id = session_id

    async def exec(self, command: str, timeout: float = 180.0) -> dict:
        """Execute a command in the sandbox with retries."""
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                result = await asyncio.wait_for(
                    self.env.session_exec(
                        session_id=self.session_id,
                        command=command,
                        exec_dir="/",
                        combine_output=True,
                    ),
                    timeout=timeout,
                )
                if isinstance(result, dict):
                    return_code = result.get("return_code", 0)
                    output = result.get("output", "")
                    if return_code != 0 and output:
                        logger_session.warning(
                            f"[sandbox] exec return_code={return_code}, output={str(output)[:200]}"
                        )
                        return {"output": output, "return_code": return_code}
                return {"output": result.get("output", str(result)) if isinstance(result, dict) else str(result)}
            except asyncio.TimeoutError:
                return {"error": f"Command timed out after {timeout}s"}
            except Exception as e:
                last_exc = e
                logger_session.warning(
                    f"[sandbox] session_exec attempt {attempt + 1}/3 failed: {e}"
                )
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return {"error": f"session_exec failed after 3 attempts: {last_exc}"}

    async def list_output_files(self) -> list[str]:
        """List all files in /mnt/user-data/outputs directory."""
        result = await self.exec("find /mnt/user-data/outputs -type f 2>/dev/null || true")
        if result.get("error"):
            logger_session.warning(f"[sandbox] list_output_files error: {result.get('error')}")
            return []
        output = result.get("output", "")
        if isinstance(output, dict):
            output = output.get("output", "")
        files = [f.strip() for f in str(output).strip().split("\n") if f.strip()]
        return files

    async def download_file(self, remote_path: str, local_path) -> bool:
        """Download a file from sandbox to local_path directory, with retries."""
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                dl = await self.env.session_download(
                    session_id=self.session_id,
                    paths=[remote_path],
                    auto_extract=False,
                ) or {}

                bos_urls: list[str] = []
                file_names: list[str] = []
                if dl.get("bos_url"):
                    bos_urls = [str(dl["bos_url"])]
                    file_names = [str(dl.get("file_name") or "download.tar.gz")]
                else:
                    bos_urls = list(dl.get("bos_urls") or [])
                    file_names = list(dl.get("files") or [])

                if not bos_urls:
                    return False

                for idx, url in enumerate(bos_urls):
                    name = file_names[idx] if idx < len(file_names) else f"download_{idx}.tar.gz"
                    if not str(name).endswith(".tar.gz"):
                        name = f"{name}.tar.gz"
                    tar_path = local_path / name

                    async with aiohttp.ClientSession() as http_session:
                        async with http_session.get(
                            url,
                            timeout=aiohttp.ClientTimeout(total=120),
                        ) as resp:
                            resp.raise_for_status()
                            tar_path.write_bytes(await resp.read())

                    with tarfile.open(tar_path, "r:gz") as tar:
                        tar.extractall(path=local_path)
                    tar_path.unlink(missing_ok=True)

                return True

            except Exception as e:
                last_exc = e
                logger_session.warning(
                    f"[sandbox] download_file attempt {attempt + 1}/3 failed: {e}"
                )
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        logger_session.warning(f"[sandbox] download_file failed: {last_exc}")
        return False

    async def stop(self):
        """Destroy this sandbox session."""
        if self.session_id is None:
            return
        try:
            await self.env.session_destroy(self.session_id)
            logger_session.warning(f"[sandbox] Destroyed session {self.session_id}")
        except Exception as e:
            logger_session.warning(f"[sandbox] session_destroy failed for {self.session_id}: {e}")
        finally:
            self.session_id = None
