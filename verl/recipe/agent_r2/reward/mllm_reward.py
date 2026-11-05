"""reward/mllm_reward.py -- LLM-as-a-judge reward for sandbox output files.

Flow:
  1. Download output files from sandbox via session.download_file().
  2. If no files → reward = 0.0.
  3. Preprocess the required deliverable file:
     - pdf / docx / pptx  → render pages to images with libreoffice/pdftoppm,
       then send images + rubric to vllm.
     - xlsx               → recalculate formulas with local LibreOffice, then send
       cell values + formulas as markdown AND rendered page images + rubric.
  4. Call vllm once with all page images (or text) and the full rubric; parse
     per-item scores and return sum / max_possible as the final reward in [0, 1].

Reference files (optional, off by default):
  When ``use_reference_files`` is enabled (argument or ``reward_config``) and
  reference files are available, they are uploaded to the sandbox, converted
  with the same pipeline as the deliverable, and shown to the judge in a
  separate "[Reference Files]" section. With the switch off (default) the judge
  input is identical to the original deliverable-only behaviour.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# vllm / openai client helpers
# ---------------------------------------------------------------------------

def _make_openai_client(llm_config: dict):
    """Create an async OpenAI-compatible client from llm_config."""
    from openai import AsyncOpenAI
    from recipe.agent_r2._settings import load_settings
    settings = load_settings()
    # reward judge has its own endpoint; fall back to the shared LLM gateway when unset
    if settings.get("reward_judge_api_url"):
        base_url = settings["reward_judge_api_url"]
        api_key = settings.get("reward_judge_api_key") or "EMPTY"
    else:
        base_url = settings["llm_api_url"]
        api_key = settings["llm_api_key"]
    # OpenAI client expects base_url ending with /v1
    if base_url and not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def _get_judge_model(llm_config: dict) -> str:
    return llm_config.get("llm_model_id", "Qwen/Qwen2.5-VL-7B-Instruct")


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_message_part(path: str, max_dim: int = 1568) -> dict:
    """Return an OpenAI-style image_url content part (base64 data URI)."""
    from PIL import Image
    img = Image.open(path)
    w, h = img.size
    if w > max_dim or h > max_dim:
        scale = min(max_dim / w, max_dim / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name)
            data = _encode_image_b64(tmp.name)
        os.unlink(tmp.name)
    else:
        data = _encode_image_b64(path)

    suffix = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(suffix, "png")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{mime};base64,{data}"},
    }


# ---------------------------------------------------------------------------
# Rubric prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator. Score the submitted document against "
    "each rubric criterion. Output ONLY a JSON code block with the structure shown."
)

_OUTPUT_FORMAT = """
Output format (JSON code block, no other text):
```json
{
  "scores": [
    {"criterion_index": 0, "score": <int>, "max_score": <int>, "reasoning": "<one sentence>"},
    ...
  ]
}
```"""

_JSON_RETRY_MESSAGE = (
    "Your previous response could not be parsed as valid JSON. "
    "Please output ONLY a JSON code block in exactly the format specified, with no other text."
)


def _build_rubric_text(rubric: list[dict]) -> str:
    lines = []
    for i, item in enumerate(rubric):
        crit = item.get("criterion", item.get("description", ""))
        pts = item.get("points", item.get("score", item.get("max_score", 1)))
        lines.append(f"[{i}] (max {pts} pts) {crit}")
    return "\n".join(lines)


_REFERENCE_NOTE = (
    "The [Reference Files] section shows a reference solution for this task. Use it to "
    "check facts, numbers and required content, but score ONLY the submitted document, "
    "and do not penalize differences in wording, structure or layout unless a rubric "
    "item explicitly requires them.\n\n"
)


def _build_user_text(
    rubric: list[dict],
    question: dict | None,
    filenames: list[str] | None = None,
    ref_filenames: list[str] | None = None,
) -> str:
    rubric_text = _build_rubric_text(rubric)
    ctx = ""
    if question:
        title = question.get("title", "")
        desc = question.get("description", "")
        deliv = question.get("deliverable", {})
        ctx = f"[Task]\nTitle: {title}\nDescription: {desc[:800]}\nDeliverable: {deliv.get('type','')} / {deliv.get('filename','')}\n\n"
    file_info = ""
    if filenames:
        file_info = f"[Files Submitted]\n" + "\n".join(f"  - {f}" for f in filenames) + "\n\n"
    ref_info = ""
    if ref_filenames:
        ref_info = (
            f"[Reference Files Provided]\n" + "\n".join(f"  - {f}" for f in ref_filenames) + "\n\n"
            + _REFERENCE_NOTE
        )
    return (
        f"{file_info}"
        f"{ref_info}"
        f"{ctx}"
        f"[Rubric]\n{rubric_text}\n\n"
        f"Please evaluate the submitted document against each rubric item.\n"
        f"{_OUTPUT_FORMAT}"
    )


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def _parse_scores(content: str, rubric: list[dict]) -> tuple[float, list[dict]]:
    """Extract scores from vllm response; return (normalized reward in [0,1], per-rubric scores)."""
    try:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        raw = m.group(1) if m else content
        data = json.loads(raw)
        scores_list = data.get("scores", [])
        total_earned = 0
        total_max = 0
        rubric_scores = []
        for entry in scores_list:
            idx = entry.get("criterion_index", -1)
            earned = entry.get("score", 0)
            if 0 <= idx < len(rubric):
                max_pts = rubric[idx].get("points", rubric[idx].get("score", rubric[idx].get("max_score", 1)))
                criterion = rubric[idx].get("criterion", rubric[idx].get("description", ""))
            else:
                max_pts = entry.get("max_score", 1)
                criterion = entry.get("criterion", "")
            # Clamp
            earned = max(0, min(int(earned), int(max_pts)))
            total_earned += earned
            total_max += int(max_pts)
            rubric_scores.append({
                "criterion_index": idx,
                "criterion": criterion,
                "score": earned,
                "max_score": int(max_pts),
                "reasoning": entry.get("reasoning", ""),
            })
        if total_max == 0:
            return 0.0, rubric_scores
        return float(total_earned) / float(total_max), rubric_scores
    except Exception as e:
        logger.warning(f"[mllm_reward] score parse error: {e}, raw: {content[:200]}")
        return 0.0, []


def _fallback_max_from_rubric(rubric: list[dict]) -> int:
    return sum(
        int(item.get("points", item.get("score", item.get("max_score", 1))))
        for item in rubric
    )


# ---------------------------------------------------------------------------
# File preprocessing
# ---------------------------------------------------------------------------

async def _convert_to_images_via_sandbox(
    session,
    remote_file: str,
    local_file: str,
    eval_input_dir: str,
    suffix: str,
    request_id: str,
) -> list[str]:
    """Convert docx/pptx/pdf → images inside sandbox, download as one tar, extract locally."""
    import tarfile as _tarfile

    eval_input_dir_path = Path(eval_input_dir).resolve()
    eval_input_dir_path.mkdir(parents=True, exist_ok=True)

    sandbox_tmp = f"/tmp/mllm_eval_{request_id}"
    sandbox_images_dir = f"{sandbox_tmp}/images"
    sandbox_tar = f"{sandbox_tmp}/images.tar.gz"

    await session.exec(f"rm -rf {sandbox_tmp} && mkdir -p {sandbox_images_dir}", timeout=30.0)

    if suffix == "pdf":
        cmd = f"pdftoppm -jpeg -r 150 {remote_file} {sandbox_images_dir}/page"
        result = await session.exec(cmd, timeout=120.0)
        if result.get("error"):
            raise RuntimeError(f"pdftoppm failed in sandbox: {result['error']}")
    else:
        # docx / pptx → pdf via soffice, then pdftoppm
        soffice_cmd = (
            f"soffice -env:UserInstallation=file://{sandbox_tmp}/.soffice_profile "
            f"--headless --convert-to pdf --outdir {sandbox_tmp} {remote_file}"
        )
        result = await session.exec(soffice_cmd, timeout=120.0)
        if result.get("error"):
            raise RuntimeError(f"soffice failed in sandbox: {result['error']}")

        find_result = await session.exec(
            f"find {sandbox_tmp} -maxdepth 1 -name '*.pdf' | head -1", timeout=10.0
        )
        # exec returns {"output": ...}, not {"result": ...}
        output = find_result.get("output", "")
        if isinstance(output, dict):
            output = output.get("output", "")
        pdf_in_sandbox = str(output).strip()
        if not pdf_in_sandbox:
            raise RuntimeError("soffice produced no PDF in sandbox")

        cmd = f"pdftoppm -jpeg -r 150 {pdf_in_sandbox} {sandbox_images_dir}/page"
        result2 = await session.exec(cmd, timeout=120.0)
        if result2.get("error"):
            raise RuntimeError(f"pdftoppm failed in sandbox: {result2['error']}")

    # Verify images were produced
    count_result = await session.exec(
        f"ls {sandbox_images_dir}/page-*.jpg 2>/dev/null | wc -l", timeout=10.0
    )
    output = count_result.get("output", "")
    if isinstance(output, dict):
        output = output.get("output", "")
    count_str = str(output).strip()
    if int(count_str or "0") == 0:
        raise RuntimeError("No images produced in sandbox after conversion")

    # Pack all images into one tar in sandbox, download once, then extract locally
    await session.exec(
        f"tar -czf {sandbox_tar} -C {sandbox_images_dir} .", timeout=60.0
    )
    ok = await session.download_file(sandbox_tar, eval_input_dir_path)
    await session.exec(f"rm -rf {sandbox_tmp}", timeout=30.0)

    if not ok:
        raise RuntimeError("Failed to download images tar from sandbox")

    # download_file extracts the outer transport tar, leaving our images.tar.gz in place;
    # extract it manually, then remove it.
    local_tar = eval_input_dir_path / "images.tar.gz"
    if local_tar.exists():
        with _tarfile.open(local_tar, "r:gz") as tf:
            tf.extractall(path=eval_input_dir_path)
        local_tar.unlink(missing_ok=True)

    return sorted(str(p) for p in eval_input_dir_path.glob("page-*.jpg"))


# xlsx: recalculate with LibreOffice (openpyxl-written formulas carry no cached
# value, so a plain data_only read shows them as empty cells), then give the
# judge (a) cell values with their formulas as text and (b) rendered pages so
# charts, formatting and layout are visible.
MAX_XLSX_PAGES = int(os.environ.get("REWARD_MAX_XLSX_PAGES", "8"))  # = max sheets rendered
_SOFFICE_TIMEOUT = float(os.environ.get("REWARD_SOFFICE_TIMEOUT", "120"))
_XLSX_PDF_FILTER = 'pdf:calc_pdf_Export:{"SinglePageSheets":{"type":"boolean","value":"true"}}'

# Force "always recalculate" on load for OOXML and ODF files.
_RECALC_XCU = """<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="OOXMLRecalcMode" oor:op="fuse"><value>0</value></prop></item>
<item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="ODFRecalcMode" oor:op="fuse"><value>0</value></prop></item>
</oor:items>
"""


async def _run_cmd(cmd: list[str], timeout: float) -> bool:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.warning(f"[mllm_reward] timeout: {cmd[0]}")
        return False
    if proc.returncode != 0:
        logger.warning(f"[mllm_reward] {cmd[0]} rc={proc.returncode}: {err.decode(errors='replace')[:200]}")
        return False
    return True


async def _soffice_convert(src: Path, fmt: str, out_dir: Path, profile: Path) -> Optional[Path]:
    """soffice --convert-to <fmt>; returns the produced file or None."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        logger.warning("[mllm_reward] soffice not found on PATH")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = await _run_cmd(
        [soffice, f"-env:UserInstallation=file://{profile}", "--headless", "--norestore",
         "--convert-to", fmt, "--outdir", str(out_dir), str(src)],
        _SOFFICE_TIMEOUT,
    )
    produced = out_dir / f"{src.stem}.{fmt.split(':')[0]}"
    return produced if ok and produced.exists() else None


async def _xlsx_recalc_and_render(
    local_xlsx: Path, converted_dir: Path, request_id: str
) -> tuple[Optional[Path], list[str]]:
    """Recalculate an xlsx with LibreOffice and render it to page images.

    Returns (recalculated xlsx or None, list of page image paths).
    """
    work = Path(tempfile.mkdtemp(prefix=f"xlsx_{request_id}_"))
    profile = work / "profile"
    (profile / "user").mkdir(parents=True)
    (profile / "user" / "registrymodifications.xcu").write_text(_RECALC_XCU)
    converted_dir.mkdir(parents=True, exist_ok=True)

    images: list[str] = []
    recalculated: Optional[Path] = None
    try:
        out = await _soffice_convert(local_xlsx, "xlsx:Calc MS Excel 2007 XML", work / "recalc", profile)
        if out is not None:
            recalculated = converted_dir / f"{local_xlsx.stem}.recalc.xlsx"
            shutil.move(str(out), recalculated)
        else:
            logger.warning(f"[mllm_reward][{request_id}] xlsx recalculation failed: {local_xlsx.name}")

        # One page per sheet so charts/tables are not split by print page breaks.
        pdf = await _soffice_convert(recalculated or local_xlsx, _XLSX_PDF_FILTER, work / "pdf", profile)
        pdftoppm = shutil.which("pdftoppm")
        if pdf is not None and pdftoppm:
            prefix = converted_dir / f"xlsx_{local_xlsx.stem}"
            if await _run_cmd(
                [pdftoppm, "-jpeg", "-scale-to", "1568", "-l", str(MAX_XLSX_PAGES), str(pdf), str(prefix)],
                _SOFFICE_TIMEOUT,
            ):
                images = sorted(
                    str(p) for p in converted_dir.iterdir()
                    if p.name.startswith(f"{prefix.name}-") and p.suffix == ".jpg"
                )
        else:
            logger.warning(f"[mllm_reward][{request_id}] xlsx render failed: {local_xlsx.name}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return recalculated, images


def _xlsx_to_markdown(local_file: str) -> str:
    """Cell values as markdown tables; formula cells are shown as `value [=FORMULA]`."""
    import openpyxl
    try:
        wb_val = openpyxl.load_workbook(local_file, read_only=True, data_only=True)
        wb_fx = openpyxl.load_workbook(local_file, read_only=True, data_only=False)
    except Exception as e:
        logger.warning(f"[mllm_reward] xlsx load failed {local_file}: {e}")
        return ""
    parts = []
    for ws_val, ws_fx in zip(wb_val.worksheets, wb_fx.worksheets):
        rows_md = []
        for row_val, row_fx in zip(ws_val.iter_rows(values_only=True), ws_fx.iter_rows(values_only=True)):
            cells = []
            for v, f in zip(row_val, row_fx):
                s = "" if v is None else str(v)
                if isinstance(f, str) and f.startswith("="):
                    s = f"{s} [{f}]" if s else f"[{f}]"
                cells.append(s.replace("|", "\\|").replace("\n", " "))
            rows_md.append("| " + " | ".join(cells) + " |")
            if len(rows_md) == 1:
                rows_md.append("| " + " | ".join(["---"] * len(cells)) + " |")
        parts.extend([f"### Sheet: {ws_val.title}", ""] + rows_md + [""])
    wb_val.close()
    wb_fx.close()
    return "\n".join(parts)


async def _extract_file_content(
    session,
    remote_path: str,
    original_dir: Path,
    converted_dir: Path,
    request_id: str,
) -> tuple[list[str], list[str]]:
    """Download one sandbox file and convert it for the judge.

    Returns (texts, images). Shared by deliverables and reference files.
    """
    suffix = Path(remote_path).suffix.lower().lstrip(".")
    filename = Path(remote_path).name
    texts: list[str] = []
    images: list[str] = []

    # Always download the original file
    try:
        ok = await session.download_file(remote_path, original_dir)
        if not ok:
            logger.warning(f"[mllm_reward][{request_id}] original download failed: {remote_path}")
    except Exception as e:
        logger.warning(f"[mllm_reward][{request_id}] original download exception: {remote_path}: {e}")

    # Convert and collect judge inputs
    try:
        if suffix == "xlsx":
            local_xlsx = original_dir / filename
            if not local_xlsx.exists():
                found = list(original_dir.rglob(filename))
                local_xlsx = found[0] if found else local_xlsx
            if local_xlsx.exists():
                recalculated, imgs = await _xlsx_recalc_and_render(local_xlsx, converted_dir, request_id)
                text = _xlsx_to_markdown(str(recalculated or local_xlsx))
                if text:
                    texts.append(f"[File: {filename}]\n{text}")
                images.extend(imgs)
                logger.info(
                    f"[mllm_reward][{request_id}] xlsx converted: {filename} "
                    f"(recalculated={recalculated is not None}, pages={len(imgs)})"
                )
            else:
                logger.warning(f"[mllm_reward][{request_id}] xlsx not found locally: {filename}")
        elif suffix in {"pdf", "docx", "pptx"}:
            imgs = await _convert_to_images_via_sandbox(
                session=session,
                remote_file=remote_path,
                local_file=str(original_dir / filename),
                eval_input_dir=str(converted_dir),
                suffix=suffix,
                request_id=request_id,
            )
            if imgs:
                images.extend(imgs)
                logger.info(f"[mllm_reward][{request_id}] {len(imgs)} images from {filename}")
        elif suffix in _TEXT_SUFFIXES:
            # Read plain-text file content from the locally downloaded copy
            local_candidates = list(original_dir.rglob(filename))
            local_txt = local_candidates[0] if local_candidates else None
            if local_txt and local_txt.is_file():
                try:
                    text = local_txt.read_text(encoding="utf-8", errors="replace")
                    if text.strip():
                        texts.append(f"[File: {filename}]\n{text}")
                        logger.info(f"[mllm_reward][{request_id}] text file read: {filename}")
                except Exception as read_exc:
                    logger.warning(
                        f"[mllm_reward][{request_id}] read text file failed {filename}: {read_exc}"
                    )
            else:
                logger.warning(
                    f"[mllm_reward][{request_id}] text file not found locally after download: {filename}"
                )
        else:
            logger.info(f"[mllm_reward][{request_id}] Unsupported type {suffix}, skipping: {filename}")
    except Exception as e:
        logger.warning(f"[mllm_reward][{request_id}] Processing {remote_path} failed: {e}")
    return texts, images


# ---------------------------------------------------------------------------
# Reference files (optional)
# ---------------------------------------------------------------------------

def _rewrite_path(p: str) -> str:
    """Apply settings.yaml path_rewrites ([[old_prefix, new_prefix], ...]) to a dataset path."""
    from recipe.agent_r2._settings import load_settings
    for old, new in load_settings().get("path_rewrites") or []:
        p = p.replace(old, new)
    return p


def resolve_reference_files(extra_info: dict | None, reward_config: dict | None = None) -> list[str]:
    """Locate local reference files for one task.

    Priority:
      1. extra_info["reference_files"]: explicit list (or JSON string) of local paths.
      2. extra_info["sample_dir"] / reward_config["reference_subdir"]: every file in that
         directory (e.g. reference_subdir="claude_solver_tmp/output").
    Returns [] when nothing is found, so callers can always pass the result through.
    """
    extra_info = extra_info or {}
    cfg = reward_config or {}

    refs = extra_info.get("reference_files")
    if isinstance(refs, str):
        try:
            refs = json.loads(refs)
        except Exception:
            refs = [refs] if refs else []
    if refs:
        return [p for p in (_rewrite_path(str(r)) for r in refs) if Path(p).is_file()]

    subdir = cfg.get("reference_subdir")
    sample_dir = extra_info.get("sample_dir")
    if subdir and sample_dir:
        ref_dir = Path(_rewrite_path(str(sample_dir))) / subdir
        if ref_dir.is_dir():
            return sorted(str(p) for p in ref_dir.iterdir() if p.is_file())
    return []


async def _prepare_reference_content(
    session,
    reference_files: list[str],
    ref_original_dir: Path,
    ref_converted_dir: Path,
    request_id: str,
) -> tuple[list[str], list[str], list[str]]:
    """Upload local reference files to the sandbox and convert them like deliverables.

    Returns (texts, images, filenames).
    """
    from recipe.agent_r2.sandbox.file_management import upload_file_to_sandbox

    sandbox_ref_dir = f"/tmp/mllm_ref_{request_id}"
    await session.exec(f"rm -rf {sandbox_ref_dir} && mkdir -p {sandbox_ref_dir}", timeout=30.0)

    texts: list[str] = []
    images: list[str] = []
    filenames: list[str] = []
    for local in reference_files:
        name = Path(local).name
        remote = f"{sandbox_ref_dir}/{name}"
        try:
            await upload_file_to_sandbox(session.env, session.session_id, local, remote)
        except Exception as e:
            logger.warning(f"[mllm_reward][{request_id}] reference upload failed {local}: {e}")
            continue
        # Each reference gets its own converted dir so page-*.jpg names don't collide
        t, imgs = await _extract_file_content(
            session=session,
            remote_path=remote,
            original_dir=ref_original_dir,
            converted_dir=ref_converted_dir / Path(name).stem,
            request_id=f"{request_id}_ref",
        )
        if t or imgs:
            filenames.append(name)
            texts.extend(t)
            images.extend(imgs)
    await session.exec(f"rm -rf {sandbox_ref_dir}", timeout=30.0)
    return texts, images, filenames


# ---------------------------------------------------------------------------
# vllm judge calls
# ---------------------------------------------------------------------------

MAX_PAGES_FOR_JUDGE = int(os.environ.get("REWARD_MAX_PAGES", "20"))
MAX_REF_PAGES_FOR_JUDGE = int(os.environ.get("REWARD_MAX_REF_PAGES", "10"))

# File suffixes treated as plain-text inputs (read content directly into the prompt)
_TEXT_SUFFIXES = {
    "txt", "md", "csv", "json", "html", "htm", "xml",
    "py", "js", "ts", "sh", "bash",
    "yaml", "yml", "toml", "ini", "cfg",
    "log", "rst", "tex", "sql", "r",
    "c", "cpp", "h", "java", "go", "rs",
}


async def _call_judge(
    images: list[str],
    texts: list[str],
    rubric: list[dict],
    question: dict | None,
    client,
    model: str,
    filenames: list[str] | None = None,
    ref_images: list[str] | None = None,
    ref_texts: list[str] | None = None,
    ref_filenames: list[str] | None = None,
) -> tuple[float, list[dict], list[dict]]:
    """Unified judge call: text-only, image-only, or mixed — always one request.

    When both images and text are present the text content is prepended before
    the rubric inside the same multimodal message so the judge sees everything.
    Reference content (if any) is appended in a separate "[Reference Files]"
    section; without references the request is unchanged.
    """
    images = images[:MAX_PAGES_FOR_JUDGE]
    ref_images = (ref_images or [])[:MAX_REF_PAGES_FOR_JUDGE]
    ref_texts = ref_texts or []
    has_ref = bool(ref_images or ref_texts)
    user_text = _build_user_text(rubric, question, filenames, ref_filenames if has_ref else None)
    max_text = int(os.environ.get("REWARD_MAX_TEXT_CHARS", "16000"))
    max_ref_text = int(os.environ.get("REWARD_MAX_REF_TEXT_CHARS", "8000"))

    ref_text_block = ""
    if ref_texts:
        combined_ref = "\n\n".join(ref_texts)
        if len(combined_ref) > max_ref_text:
            combined_ref = combined_ref[:max_ref_text] + "\n...[truncated]"
        ref_text_block = f"[Reference Files Content]\n{combined_ref}\n\n"

    is_multimodal = bool(images or ref_images)

    if is_multimodal:
        content: list[dict] = []
        if has_ref:
            content.append({"type": "text", "text": "[Submitted Deliverable Pages]"})
        for img_path in images:
            try:
                content.append(_image_message_part(img_path))
            except Exception as e:
                logger.warning(f"[mllm_reward] image encode failed {img_path}: {e}")
        if ref_images:
            content.append({"type": "text", "text": "[Reference Files Pages]"})
            for img_path in ref_images:
                try:
                    content.append(_image_message_part(img_path))
                except Exception as e:
                    logger.warning(f"[mllm_reward] reference image encode failed {img_path}: {e}")

        if not any(p.get("type") == "image_url" for p in content):
            return 0.0, [], []

        if texts:
            combined = "\n\n".join(texts)
            if len(combined) > max_text:
                combined = combined[:max_text] + "\n...[truncated]"
            full_user_text = f"[Document Content]\n{combined}\n\n{ref_text_block}{user_text}"
        else:
            full_user_text = f"{ref_text_block}{user_text}"

        content.append({"type": "text", "text": full_user_text})
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
    else:
        combined = "\n\n".join(texts)
        if len(combined) > max_text:
            combined = combined[:max_text] + "\n...[truncated]"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"[Document Content]\n{combined}\n\n{ref_text_block}{user_text}"},
        ]

    # ====================== 修复部分开始 ======================
    # 配置：最大重试 5 次，max_tokens 16384
    MAX_RETRY = 5
    current_messages = messages.copy()
    final_answer = ""
    reward = 0.0
    rubric_scores = []

    for attempt in range(MAX_RETRY + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=current_messages,
                max_tokens=16384,  # 👈 改成 16384
                temperature=0.0,
            )
            answer = resp.choices[0].message.content or ""
            final_answer = answer

            # 解析分数
            reward, rubric_scores = _parse_scores(answer, rubric)
            if rubric_scores:
                logger.info(f"[mllm_reward] 解析成功！第 {attempt+1} 次调用生效")
                break  # 成功就退出循环

            # 解析失败 → 添加重试提示
            logger.warning(f"[mllm_reward] JSON解析失败，第 {attempt+1} 次重试")
            current_messages.append({"role": "assistant", "content": answer})
            current_messages.append({"role": "user", "content": _JSON_RETRY_MESSAGE})

        except Exception as e:
            logger.warning(f"[mllm_reward] judge调用异常: {e}")
            current_messages.append({"role": "assistant", "content": ""})
            current_messages.append({"role": "user", "content": _JSON_RETRY_MESSAGE})

    # 所有重试耗尽仍未解析到 scores → fallback reward = 0.5
    if not rubric_scores:
        logger.warning(f"[mllm_reward] 所有 {MAX_RETRY+1} 次尝试均未解析到scores，reward设为0.5")
        reward = 0.5

    # 构建保存的对话记录
    if is_multimodal:
        saved_content = [
            p if p.get("type") != "image_url"
            else {"type": "image_url", "image_url": {"url": "<base64_omitted>"}}
            for p in content
        ]
        saved_messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": saved_content},
            {"role": "assistant", "content": final_answer},
        ]
    else:
        saved_messages = messages + [{"role": "assistant", "content": final_answer}]

    return reward, saved_messages, rubric_scores
    # ====================== 修复部分结束 ======================

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def compute_reward(
    session,
    rubric: list[dict],
    question: dict | None = None,
    request_id: str = "",
    traj_dir: "Path | str | None" = None,
    llm_config: dict | None = None,
    reference_files: list[str] | None = None,
    use_reference_files: bool | None = None,
) -> tuple[float, dict]:
    """Convert sandbox output files in-place, download originals + converted, call judge.

    Flow:
      1. List output files in /mnt/user-data/outputs (same sandbox the agent just used).
      2. For each file: convert (pdf/pptx/docx → images inside the sandbox;
         xlsx → LibreOffice recalculation + markdown + page images locally), and
         download both the original and the converted artefacts.
      3. Call the vllm judge with the collected images / text and return reward + info.

    Directory layout written under traj_dir (when provided):
        solver/output/   ← original deliverable files
        eval/input/      ← converted images (pdf/docx/pptx/xlsx) + recalculated xlsx
        eval/reference/  ← original + converted reference files (only when enabled)
        eval/conversation.json ← judge messages + scores

    Args:
        session: SandboxSession instance (the one the agent just ran in).
        rubric: list of rubric items with 'criterion' and 'points' fields.
        question: optional question dict (title, description, deliverable).
        request_id: trajectory ID, used only for log messages.
        traj_dir: base directory for this trajectory's output. If None, uses a
            temp dir that is cleaned up afterwards.
        reference_files: optional local paths of reference files for this task
            (see resolve_reference_files). Ignored unless reference use is enabled.
        use_reference_files: show reference files to the judge. None → read
            llm_config["use_reference_files"] (default False).

    Returns:
        (reward, info_dict) where reward ∈ [0.0, 1.0].
    """
    info: dict[str, Any] = {
        "has_output": False,
        "file_type": None,
        "num_pages": 0,
        "reward": 0.0,
    }

    if not rubric:
        return 0.0, info

    # ── 1. List output files ────────────────────────────────────────────────
    files = await session.list_output_files()
    logger.warning(f"[mllm_reward][{request_id}] list_output_files: {files}")
    if not files:
        logger.info(f"[mllm_reward][{request_id}] No output files → reward=0.0")
        return 0.0, info

    info["has_output"] = True
    info["output_files"] = files

    # ── 2. Setup local directories ──────────────────────────────────────────
    _tmp_root = None
    if traj_dir is not None:
        traj_dir = Path(traj_dir)
        solver_output_dir = traj_dir / "solver" / "output"
        eval_input_dir = traj_dir / "eval" / "input"
        eval_dir = traj_dir / "eval"
        cleanup = False
    else:
        _tmp_root = Path(tempfile.mkdtemp(prefix=f"mllm_reward_{request_id}_"))
        solver_output_dir = _tmp_root / "solver" / "output"
        eval_input_dir = _tmp_root / "eval" / "input"
        eval_dir = _tmp_root / "eval"
        cleanup = True

    solver_output_dir.mkdir(parents=True, exist_ok=True)
    eval_input_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 3. For each output file: convert in sandbox, download original + converted ──
        all_texts: list[str] = []
        all_images: list[str] = []
        all_filenames: list[str] = []

        for remote_path in files:
            all_filenames.append(Path(remote_path).name)
            texts, images = await _extract_file_content(
                session=session,
                remote_path=remote_path,
                original_dir=solver_output_dir,
                converted_dir=eval_input_dir,
                request_id=request_id,
            )
            all_texts.extend(texts)
            all_images.extend(images)

        # ── 3b. Optional reference files ────────────────────────────────────
        _cfg = llm_config or {}
        if use_reference_files is None:
            use_reference_files = bool(_cfg.get("use_reference_files", False))
        ref_texts: list[str] = []
        ref_images: list[str] = []
        ref_filenames: list[str] = []
        info["use_reference_files"] = use_reference_files
        if use_reference_files and reference_files:
            ref_root = eval_dir / "reference"
            ref_texts, ref_images, ref_filenames = await _prepare_reference_content(
                session=session,
                reference_files=list(reference_files),
                ref_original_dir=ref_root / "original",
                ref_converted_dir=ref_root / "converted",
                request_id=request_id,
            )
            logger.info(
                f"[mllm_reward][{request_id}] reference files used: {ref_filenames} "
                f"(texts={len(ref_texts)} images={len(ref_images)})"
            )
        elif use_reference_files:
            logger.warning(f"[mllm_reward][{request_id}] use_reference_files=True but no reference files found")
        info["reference_files"] = ref_filenames

        # ── 4. Score all collected content ──────────────────────────────────
        if not all_texts and not all_images:
            logger.warning(f"[mllm_reward][{request_id}] No content extracted from files → reward=0.0")
            return 0.0, info

        client = _make_openai_client(_cfg)
        model = _get_judge_model(_cfg)

        info["eval_input"] = "text" if not all_images else str(eval_input_dir)
        info["num_pages"] = len(all_images)

        reward, judge_messages, rubric_scores = await _call_judge(
            images=all_images,
            texts=all_texts,
            rubric=rubric,
            question=question,
            client=client,
            model=model,
            filenames=all_filenames,
            ref_images=ref_images,
            ref_texts=ref_texts,
            ref_filenames=ref_filenames,
        )
        info["reward"] = reward

        # ── 5. Save eval/conversation.json and eval/result.json ────────────
        _save_eval_conversation(
            eval_dir=eval_dir,
            judge_messages=judge_messages,
            reward=reward,
            rubric=rubric,
            rubric_scores=rubric_scores,
            question=question,
            info=info,
        )

        logger.info(
            f"[mllm_reward][{request_id}] reward={reward:.3f} "
            f"(texts={len(all_texts)} images={len(all_images)})"
        )
        return reward, info

    finally:
        if cleanup and _tmp_root is not None:
            shutil.rmtree(str(_tmp_root), ignore_errors=True)


def _save_eval_conversation(
    eval_dir: Path,
    judge_messages: list[dict],
    reward: float,
    rubric: list[dict],
    rubric_scores: list[dict],
    question: dict | None,
    info: dict,
) -> None:
    """Write eval/conversation.json and eval/result.json with judge messages and scores."""
    try:
        record = {
            "reward": reward,
            "file_type": info.get("file_type"),
            "num_pages": info.get("num_pages", 0),
            "rubric": rubric,
            "question": question,
            "judge_conversation": judge_messages,
        }
        conv_file = eval_dir / "conversation.json"
        with open(conv_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)

        result = {
            "reward": reward,
            "rubric_scores": rubric_scores,
        }
        result_file = eval_dir / "result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.warning(f"[mllm_reward] _save_eval_conversation failed: {e}")
