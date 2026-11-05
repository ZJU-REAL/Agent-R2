"""测试 sandbox 各功能，记录执行时间。所有调用均为异步。"""
from __future__ import annotations
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sandbox.create import create_sessions, _create_env
from sandbox.check import check_session
from sandbox.destroy import destroy_session
from sandbox.file_management import upload_files_to_sandbox, download_files_from_sandbox

SANDBOX_IMAGE = "rle/aio-sandbox:terminal-0.0.1"
TEST_FILE = Path(__file__).parent / "LibreOffice_26.2.1_Linux_x86-64_deb.tar.gz"
DOWNLOAD_DIR = Path(__file__).parent / "test_downloads"


def _t(label: str, start: float) -> float:
    elapsed = time.time() - start
    print(f"[TIMER] {label}: {elapsed:.2f}s")
    return elapsed


async def run_tests():
    results = {}

    # ── 1. create_session ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: create_sessions")
    print("=" * 60)
    t0 = time.time()
    sids = await create_sessions(count=1, image=SANDBOX_IMAGE)
    results["create_session"] = _t("create_session", t0)
    if not sids:
        print("[FAIL] 创建 session 失败，终止测试")
        return results
    sid = sids[0]
    print(f"session_id: {sid}")

    env = _create_env()

    # ── 2. session_exec ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: session_exec")
    print("=" * 60)
    t0 = time.time()
    try:
        res = await env.session_exec(session_id=sid, command="echo hello_sandbox", combine_output=True)
        print(f"exec result: {res}")
        results["session_exec"] = _t("session_exec", t0)
    except Exception as e:
        print(f"[FAIL] session_exec: {e}")
        results["session_exec"] = _t("session_exec(FAIL)", t0)

    # ── 3. session_upload ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: session_upload")
    print("=" * 60)
    if not TEST_FILE.exists():
        print(f"[SKIP] 测试文件不存在: {TEST_FILE}")
        results["session_upload"] = 0.0
        results["verify_upload"] = 0.0
        results["session_download"] = 0.0
        results["upload_files_to_sandbox"] = 0.0
    else:
        remote_path = f"/tmp/{TEST_FILE.name}"

        t0 = time.time()
        try:
            res = await env.session_upload(session_id=sid, local_path=str(TEST_FILE), remote_path=remote_path)
            print(f"upload result: {res}")
            results["session_upload"] = _t("session_upload", t0)
        except Exception as e:
            print(f"[FAIL] session_upload: {e}")
            results["session_upload"] = _t("session_upload(FAIL)", t0)

        # ── 4. 验证上传文件 ──────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 4: verify upload via session_exec")
        print("=" * 60)
        t0 = time.time()
        try:
            res = await env.session_exec(session_id=sid, command=f"ls -lh {remote_path}", combine_output=True)
            print(f"ls result: {res}")
            results["verify_upload"] = _t("verify_upload", t0)
        except Exception as e:
            print(f"[FAIL] verify_upload: {e}")
            results["verify_upload"] = _t("verify_upload(FAIL)", t0)

        # ── 5. session_download ──────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 5: session_download")
        print("=" * 60)
        t0 = time.time()
        try:
            downloaded = await download_files_from_sandbox(env, sid, [remote_path], DOWNLOAD_DIR)
            print(f"downloaded files: {downloaded}")
            results["session_download"] = _t("session_download", t0)
        except Exception as e:
            print(f"[FAIL] session_download: {e}")
            results["session_download"] = _t("session_download(FAIL)", t0)

        # ── 6. upload_files_to_sandbox ───────────────────────────
        print("\n" + "=" * 60)
        print("STEP 6: upload_files_to_sandbox (file_management)")
        print("=" * 60)
        t0 = time.time()
        try:
            remote_paths = await upload_files_to_sandbox(
                env, sid,
                local_files=[TEST_FILE],
                local_root=TEST_FILE.parent,
                remote_root="/tmp/fm_test",
            )
            print(f"uploaded to: {remote_paths}")
            results["upload_files_to_sandbox"] = _t("upload_files_to_sandbox", t0)
        except Exception as e:
            print(f"[FAIL] upload_files_to_sandbox: {e}")
            results["upload_files_to_sandbox"] = _t("upload_files_to_sandbox(FAIL)", t0)

    # ── 7. check_session ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: check_session")
    print("=" * 60)
    t0 = time.time()
    try:
        ok = await check_session(sid, move_on_success=False)
        print(f"check result: {'PASS' if ok else 'FAIL'}")
        results["check_session"] = _t("check_session", t0)
    except Exception as e:
        print(f"[FAIL] check_session: {e}")
        results["check_session"] = _t("check_session(FAIL)", t0)

    # ── 8. destroy_session ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: destroy_session")
    print("=" * 60)
    t0 = time.time()
    try:
        await destroy_session(sid)
        results["destroy_session"] = _t("destroy_session", t0)
    except Exception as e:
        print(f"[FAIL] destroy_session: {e}")
        results["destroy_session"] = _t("destroy_session(FAIL)", t0)

    # ── 汇总 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for step, elapsed in results.items():
        status = "SKIP" if elapsed == 0.0 else "OK"
        print(f"  {step:<35} {elapsed:.2f}s  [{status}]")
    print(f"\n  TOTAL: {sum(results.values()):.2f}s")

    return results


if __name__ == "__main__":
    asyncio.run(run_tests())
