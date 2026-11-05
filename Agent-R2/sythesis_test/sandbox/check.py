"""检查 session 依赖是否就绪。"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from sandbox.create import _create_env

_PIP_CHECK = "python3 -c 'import markitdown, PIL, numpy, pandas, openpyxl, pypdf, pdfplumber, reportlab, pytesseract, pdf2image'"
_NPM_GLOBALS = ("pptxgenjs", "docx", "react", "react-dom", "react-icons", "sharp")


async def _run(env: Any, sid: str, label: str, script: str) -> bool:
    marker = "__CHECK_RC__="
    wrapped = f"( {script} ); __rc=$?; echo \"{marker}${{__rc}}\""
    res = await env.session_exec(session_id=sid, commands=["bash", "-lc", wrapped], combine_output=True)
    out = (res.get("output") or "").strip()

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


async def check_session(session_id: str) -> bool:
    """检查 session 依赖是否就绪"""
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

    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] session_id={session_id}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check.py <session_id> [session_id ...]")
        sys.exit(1)

    passed = failed = 0
    for sid in sys.argv[1:]:
        try:
            if asyncio.run(check_session(sid)):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[SKIP] session {sid} 检查失败: {e}")

    print(f"\n检查完成: passed={passed}, failed={failed}")
    sys.exit(0 if failed == 0 else 1)
