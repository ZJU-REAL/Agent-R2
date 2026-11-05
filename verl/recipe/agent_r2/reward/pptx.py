"""data_sythesis/preprocess/pptx.py -- PowerPoint 文件预处理，帮助模型理解 .pptx 内容。

对应 skills/pptx 中的能力：
  - markitdown 文本提取
  - unpack.py 解析 XML
  - soffice.py + pdftoppm 转图片
  - thumbnail.py 生成缩略图网格

依赖:
    pip install "markitdown[pptx]" python-pptx Pillow
    LibreOffice（soffice）+ Poppler（pdftoppm）用于转图片
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_text_markitdown(pptx_path: str) -> str:
    """使用 markitdown 提取 .pptx 文本（含演讲者备注），转为 Markdown。

    Args:
        pptx_path: .pptx 文件路径。

    Returns:
        Markdown 格式文本字符串。
    """
    result = subprocess.run(
        ["python", "-m", "markitdown", pptx_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"markitdown 提取失败: {result.stderr}")
    return result.stdout


def extract_text_python_pptx(pptx_path: str) -> str:
    """使用 python-pptx 提取 .pptx 中所有幻灯片的文本。

    Args:
        pptx_path: .pptx 文件路径。

    Returns:
        按幻灯片编号组织的文本字符串。
    """
    from pptx import Presentation

    prs = Presentation(pptx_path)
    slides_text: List[str] = []

    for i, slide in enumerate(prs.slides):
        parts: List[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        parts.append(text)
        if parts:
            slides_text.append(f"[Slide {i + 1}]\n" + "\n".join(parts))

    return "\n\n".join(slides_text)


def extract_slide_structure(pptx_path: str) -> List[Dict[str, Any]]:
    """提取每张幻灯片的结构信息（标题、正文、备注、形状类型）。

    Returns:
        [
            {
                "index": 1,
                "title": "...",
                "body": ["...", ...],
                "notes": "...",
                "shape_types": ["TEXT_BOX", "PICTURE", ...]
            },
            ...
        ]
    """
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    prs = Presentation(pptx_path)
    slides: List[Dict[str, Any]] = []

    for i, slide in enumerate(prs.slides):
        title = ""
        body_texts: List[str] = []
        shape_types: List[str] = []

        for shape in slide.shapes:
            # 记录形状类型
            try:
                shape_types.append(shape.shape_type.name)
            except Exception:
                shape_types.append("UNKNOWN")

            if not shape.has_text_frame:
                continue

            text = shape.text_frame.text.strip()
            if not text:
                continue

            # 判断是否是标题占位符
            if shape.is_placeholder:
                ph = shape.placeholder_format
                if ph and ph.idx == 0:
                    title = text
                    continue

            body_texts.append(text)

        # 提取演讲者备注
        notes_text = ""
        if slide.has_notes_slide:
            notes_tf = slide.notes_slide.notes_text_frame
            if notes_tf:
                notes_text = notes_tf.text.strip()

        slides.append({
            "index": i + 1,
            "title": title,
            "body": body_texts,
            "notes": notes_text,
            "shape_types": shape_types,
        })

    return slides


def extract_metadata(pptx_path: str) -> Dict[str, Any]:
    """提取 .pptx 文件属性（标题、作者、幻灯片数量等）。

    Returns:
        包含文档属性的字典。
    """
    from pptx import Presentation

    prs = Presentation(pptx_path)
    props = prs.core_properties
    return {
        "title": props.title or "",
        "author": props.author or "",
        "subject": props.subject or "",
        "description": props.comments or "",
        "created": str(props.created) if props.created else "",
        "modified": str(props.modified) if props.modified else "",
        "last_modified_by": props.last_modified_by or "",
        "slide_count": len(prs.slides),
        "slide_width_emu": prs.slide_width,
        "slide_height_emu": prs.slide_height,
    }


def convert_to_images(
    pptx_path: str,
    output_dir: str,
    dpi: int = 150,
    soffice_script: Optional[str] = None,
    use_soffice: bool = True,
) -> List[str]:
    """将 .pptx 每张幻灯片转为图片。

    use_soffice=True（默认）：pptx → PDF（soffice） → 逐页图片（pdftoppm）
    use_soffice=False：pptx → PDF（python-pptx-pdf / docx2pdf）→ 图片（pymupdf）

    Args:
        pptx_path: .pptx 文件路径。
        output_dir: 图片输出目录。
        dpi: 渲染分辨率。
        soffice_script: skills/pptx/scripts/office/soffice.py 路径（use_soffice=True 时有效），
                        若为 None 则直接调用系统 soffice。
        use_soffice: 是否使用 LibreOffice（soffice）进行转换，默认 True。

    Returns:
        生成的图片路径列表（按幻灯片顺序）。
    """
    os.makedirs(output_dir, exist_ok=True)
    pptx_path = str(Path(pptx_path).resolve())

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path: Optional[str] = None

        if use_soffice:
            # --- 路径 A: soffice 转 PDF ---
            if soffice_script:
                cmd = ["python", soffice_script, "--headless", "--convert-to", "pdf",
                       "--outdir", tmp, pptx_path]
            else:
                user_install = f"file://{tmp}/.soffice_profile"
                cmd = ["soffice", f"-env:UserInstallation={user_install}",
                       "--headless", "--convert-to", "pdf",
                       "--outdir", tmp, pptx_path]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"soffice 转换失败: {result.stderr}")

            pdf_files = list(Path(tmp).glob("*.pdf"))
            if not pdf_files:
                raise RuntimeError("soffice 未生成 PDF 文件")
            pdf_path = str(pdf_files[0])

            prefix = os.path.join(output_dir, "slide")
            result2 = subprocess.run(
                ["pdftoppm", "-jpeg", "-r", str(dpi), pdf_path, prefix],
                capture_output=True,
                text=True,
            )
            if result2.returncode != 0:
                raise RuntimeError(f"pdftoppm 失败: {result2.stderr}")

            images = sorted(Path(output_dir).glob("slide-*.jpg"))
            return [str(p) for p in images]

        else:
            # --- 路径 B: docx2pdf 转 PDF，再用 pymupdf 渲染 ---
            try:
                from docx2pdf import convert as _docx2pdf_convert
                pdf_path = os.path.join(tmp, Path(pptx_path).stem + ".pdf")
                _docx2pdf_convert(pptx_path, pdf_path)
            except Exception as e:
                raise RuntimeError(f"docx2pdf 转换失败（请确认已安装 docx2pdf 且系统有 Word/LibreOffice）: {e}")

            try:
                import fitz  # pymupdf
            except ImportError:
                raise RuntimeError("pymupdf 未安装，请执行 pip install pymupdf")

            doc = fitz.open(pdf_path)
            saved: List[str] = []
            for i, page in enumerate(doc):
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                out_path = os.path.join(output_dir, f"slide_{i + 1}.png")
                pix.save(out_path)
                saved.append(out_path)
            doc.close()
            return saved


def generate_thumbnail_grid(
    pptx_path: str,
    output_path: str,
    cols: int = 3,
    thumbnail_script: Optional[str] = None,
) -> str:
    """生成幻灯片缩略图网格图片（使用 skills/pptx/scripts/thumbnail.py）。

    Args:
        pptx_path: .pptx 文件路径。
        output_path: 输出图片路径（.jpg）。
        cols: 每行缩略图数量。
        thumbnail_script: thumbnail.py 脚本路径；None 则尝试自动查找。

    Returns:
        生成的缩略图路径。
    """
    if thumbnail_script is None:
        # 尝试从 skills 目录自动定位
        skills_dir = Path(__file__).resolve().parent.parent / "skills" / "pptx" / "scripts"
        candidate = skills_dir / "thumbnail.py"
        if candidate.exists():
            thumbnail_script = str(candidate)
        else:
            raise FileNotFoundError(
                "未找到 thumbnail.py，请通过 thumbnail_script 参数指定路径"
            )

    out_prefix = str(Path(output_path).with_suffix(""))
    result = subprocess.run(
        ["python", thumbnail_script, pptx_path, out_prefix, "--cols", str(cols)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"thumbnail.py 失败: {result.stderr}")

    # thumbnail.py 默认输出 <prefix>.jpg 或 <prefix>_1.jpg
    candidates = sorted(Path(output_path).parent.glob(Path(output_path).stem + "*.jpg"))
    return str(candidates[0]) if candidates else output_path


def preprocess(pptx_path: str, output_dir: Optional[str] = None, use_soffice: bool = True) -> Dict[str, Any]:
    """一键预处理 .pptx，返回供模型使用的结构化摘要。

    Args:
        pptx_path: .pptx 文件路径。
        output_dir: 可选，图片输出目录。
        use_soffice: 是否使用 LibreOffice 转图片，默认 True。
                     False 时改用 docx2pdf + pymupdf。

    Returns:
        {
            "metadata": {...},
            "text": "...",          # markitdown 提取，失败则用 python-pptx
            "slides": [...],        # 每张幻灯片的结构信息
            "images": [...],        # 仅在 output_dir 非空时存在
        }
    """
    result: Dict[str, Any] = {}

    # 元数据
    try:
        result["metadata"] = extract_metadata(pptx_path)
    except Exception:
        result["metadata"] = {}

    # 正文文本（优先 markitdown）
    try:
        result["text"] = extract_text_markitdown(pptx_path)
    except Exception:
        try:
            result["text"] = extract_text_python_pptx(pptx_path)
        except Exception:
            result["text"] = ""

    # 幻灯片结构
    try:
        result["slides"] = extract_slide_structure(pptx_path)
    except Exception:
        result["slides"] = []

    # 图片（可选）
    if output_dir:
        try:
            result["images"] = convert_to_images(pptx_path, output_dir, use_soffice=use_soffice)
        except Exception:
            result["images"] = []

    return result
