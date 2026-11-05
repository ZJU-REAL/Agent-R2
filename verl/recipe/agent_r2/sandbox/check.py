"""检查 session 依赖是否就绪，检查通过后移动到 checked 目录。

SDK 已升级为异步（session_exec 为 async）。
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
from typing import Any

from recipe.agent_r2.sandbox.create import _create_env, SESSION_UNCHECKED, SESSION_CHECKED

_PIP_CHECK = "python3 -c 'import markitdown, PIL, numpy, pandas, openpyxl, pypdf, pdfplumber, reportlab, pytesseract, pdf2image'"
_NPM_GLOBALS = ("pptxgenjs", "docx", "react", "react-dom", "react-icons", "sharp")


async def _run(env: Any, sid: str, label: str, script: str) -> bool:
    marker = "__CHECK_RC__="
    wrapped = f"( {script} ); __rc=$?; echo \"{marker}${{__rc}}\""
    res = await env.session_exec(session_id=sid, command=wrapped,
                                  exec_dir="/", combine_output=True)
    out = (res.get("output") or "").strip()

    # 提取实际输出（去掉 marker 行）
    actual_output = "\n".join(line for line in out.splitlines() if not line.startswith(marker))

    for line in out.splitlines():
        if line.startswith(marker):
            try:
                rc = int(line[len(marker):].strip())
                if rc != 0:
                    print(f"  [{label}] FAIL: exit_code={rc}")
                    if actual_output:
                        print(f"  [{label}] output: {actual_output[:500]}")
                else:
                    print(f"  [{label}] OK")
                return rc == 0
            except ValueError:
                pass

    rc = int(res.get("exit_code", 1))
    if rc != 0:
        print(f"  [{label}] FAIL: exit_code={rc}")
        if actual_output:
            print(f"  [{label}] output: {actual_output[:500]}")
    else:
        print(f"  [{label}] OK")
    return rc == 0


async def check_session(session_id: str, move_on_success: bool = True) -> bool:
    """检查 session 依赖是否就绪，成功后移动到 checked 目录"""
    print(f"\n检查 session: {session_id}")
    env = _create_env()
    ok = True
    ok &= await _run(env, session_id, "pip", _PIP_CHECK)
    ok &= await _run(env, session_id, "soffice", "soffice --version")
    ok &= await _run(env, session_id, "poppler", "pdftotext -v")
    ok &= await _run(env, session_id, "qpdf", "qpdf --version")
    ok &= await _run(env, session_id, "pandoc", "pandoc --version")
    npm_script = " ".join(f'npm list -g "{p}" --depth=0 >/dev/null || exit 1;' for p in _NPM_GLOBALS)
    ok &= await _run(env, session_id, "npm", f"set -e; {npm_script} echo ok")

    if ok and move_on_success:
        SESSION_CHECKED.mkdir(parents=True, exist_ok=True)
        unchecked_file = SESSION_UNCHECKED / f"{session_id}.id"
        checked_file = SESSION_CHECKED / f"{session_id}.id"
        if unchecked_file.exists():
            content = unchecked_file.read_text()
            checked_file.write_text(content)
            unchecked_file.unlink()
            print(f"[MOVE] {session_id} -> checked")

    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] session_id={session_id}")
    return ok


def list_unchecked() -> list[str]:
    if not SESSION_UNCHECKED.is_dir():
        return []
    return [p.read_text().strip() for p in sorted(SESSION_UNCHECKED.glob("*.id"))]


if __name__ == "__main__":
    async def _main():
        sids = list_unchecked()
        if not sids:
            print("没有待检查的 session")
            sys.exit(0)

        passed = 0
        failed = 0
        for sid in sids:
            if await check_session(sid):
                passed += 1
            else:
                failed += 1

        print(f"\n检查完成: passed={passed}, failed={failed}")
        sys.exit(0 if failed == 0 else 1)

    asyncio.run(_main())
